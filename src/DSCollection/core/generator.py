import configparser
import os
import sys
import glob
import time

import cv2
import gi
import pyds
import numpy as np

from enum import IntEnum
from threading import Thread
from multiprocessing import Process, Queue
from queue import Queue as threadQueue, Empty
from typing import List, Dict, Callable, Tuple, Type

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib, GObject
from tqdm import tqdm

from ..core.dataset import DatasetType, Box, ImageLabel
from ..core.convertor import Convertor, LabelInfo
from ..utils.common import check_path, is_video, is_image
from ..utils.imgutil import ImageUtil
from .gst import build_uri_source, build_preprocess

g_count: int = 0                    # The number of consecutive frames that have been processed
g_num_srcs: int = 0                 # Number of sources already in pipeline
g_sync_num: int = 150               # Threshold trigger synchronizing buffer
g_uri_list: List[str]               # Static array containing current URIs
g_enabled_list: List[bool]          # Static array containing flags that indicate corresponding sources are enabled
g_eos_list: List[bool]              # Static array containing flags that indicate corresponding sources reached eos
g_src_list: List[Gst.Element]       # Static array containing current source elements
g_prc_list: List[Gst.Element]       # Static array containing current preprocess elements
g_cls_dict: Dict[int, List[str]]    # Map model unique-id to class names


class InputType(IntEnum):
    UNKNOWN = 0
    VIDEO = 1
    IMAGE = 2


class MemoryType(IntEnum):
    DEFAULT = 0
    CUDA_PINNED = 1
    CUDA_DEVICE = 2
    CUDA_UNIFIED = 3


class _GeneratorDaemon(Process):

    def __init__(self, index: int, queue: Queue):
        super(_GeneratorDaemon, self).__init__()
        self.index = index
        self.name = f"{index} - Subprocess"
        self.queue = queue
        self.daemon = True
        self._buffer = threadQueue()
        self._num_workers = 2
        self._worker = []

    def run(self) -> None:
        frame: np.ndarray
        lblInfo: LabelInfo
        dstImg: str
        dstLbl: str

        # Start two threads
        for i in range(self._num_workers):
            worker = _GeneratorDaemonWorker(self._buffer)
            worker.start()
            self._worker.append(worker)

        while True:
            data = self.queue.get()
            if data is None:
                for i in range(self._num_workers):
                    self._buffer.put(None)
                break
            frame, lblInfo, dstImg, dstLbl = data
            if dstLbl:
                self._buffer.put((lblInfo.data, dstLbl))
            imgExt = os.path.splitext(dstImg)[1]
            frameBytes = ImageUtil.encode(frame, imgExt)
            self._buffer.put((frameBytes, dstImg))

        for worker in self._worker:
            worker.join()


class _GeneratorDaemonWorker(Thread):

    def __init__(self, queue: threadQueue):
        super(_GeneratorDaemonWorker, self).__init__()
        self._q = queue
        self.daemon = True

    def run(self) -> None:
        while True:
            content = self._q.get()
            if content is None:
                break
            data, path = content
            with open(path, 'wb') as f:
                f.write(data)


class Generator:
    """
    Generator built on top of Gstreamer and Deepstream that generate new dataset from
    raw fisheye videos. In addition, this class is able to automate labeling using
    deepstream models.
    """

    def __init__(self, inputs: List[str], outDir: str,
                 models: List[str] = None,
                 dType: str = DatasetType.KITTI,
                 subdir: str = "",
                 imgExt: str = ".png",
                 max_srcs: int = None,
                 skip_mode: int = None,
                 interval: int = None,
                 gpu_id: int = None,
                 camera_shifts: Tuple[int, int, int, int] = None,
                 crop_size: int = None,
                 memory_type: int = None,
                 num_workers: int = None,
                 contiguous: bool = False,
                 drop_empty: bool = False,
                 save_empty_label: bool = False):

        global g_enabled_list
        global g_eos_list
        global g_uri_list
        global g_src_list
        global g_prc_list
        global g_cls_dict

        self.input_type, self.input_uris = self.check_inputs(inputs)
        try:
            cvt = Convertor.from_type(dType)
        except ValueError:
            msg = "error: unsupported dtype"
            raise RuntimeError(msg)
        self.outDir, self.imgDir, self.lblDir = self.check_output(outDir, subdir, cvt)
        self.models = self.check_models(models)
        self.imgExt = imgExt if is_image(imgExt) else ".jpg"
        self.lblExt = cvt.lblExt
        self.convert_label = cvt.convert_label
        self.max_srcs = 1 if not max_srcs else max(1, int(max_srcs))
        if self.max_srcs == 1 and len(self.input_uris) > 1:
            self.max_srcs = min(4, len(self.input_uris))
            sys.stderr.write(f"warn: max_srcs=1 while multiple sources found\n")
            sys.stderr.write(f"warn: update max_srcs={self.max_srcs}\n")

        # Initialization of global variables
        g_enabled_list = [False] * self.max_srcs
        g_eos_list = [True] * self.max_srcs
        g_uri_list = [""] * self.max_srcs
        g_src_list = [None] * self.max_srcs
        g_prc_list = [None] * self.max_srcs
        g_cls_dict = {}
        self.contiguous = contiguous
        self.drop_empty = drop_empty
        self.save_empty_label = save_empty_label
        if contiguous:
            self.imgPathFMT = os.path.join(self.imgDir, "{count:08d}" + self.imgExt)
            self.lblPathFMT = os.path.join(self.lblDir, "{count:08d}" + self.lblExt)
        else:
            self.imgPathFMT = os.path.join(self.imgDir, "{name}-{fIdx}-{cIdx}" + self.imgExt)
            self.lblPathFMT = os.path.join(self.lblDir, "{name}-{fIdx}-{cIdx}" + self.lblExt)

        # Initialization of Gstreamer
        Gst.init(None)
        self._pipeline_dot = ""
        self.pipeline: Gst.Pipeline
        self.loop: GLib.MainLoop
        self.streammux: Gst.Element
        self.probe_func: Callable
        self.state: Gst.State = None
        self.skip_mode = 0 if not skip_mode else max(0, min(2, int(skip_mode)))
        self.interval = 0 if not interval else max(0, int(interval))
        self.gpu_id = 0 if not gpu_id else max(0, int(gpu_id))
        self.cam_shifts = (0, 0, 0, 0) if not camera_shifts else tuple(int(x) for x in camera_shifts)
        self.crop_size = 1296 if not crop_size else max(100, min(1296, int(crop_size)))
        self.mem_type = MemoryType.CUDA_UNIFIED if not memory_type else max(0, min(3, int(memory_type)))

        # Initialization of thread that save img-bytes and lbl-bytes
        self.num_workers = 1 if not num_workers else max(1, min(os.cpu_count() - 1, int(num_workers)))
        self.buffer = Queue()
        self.workers = []
        for i in range(self.num_workers):
            worker = _GeneratorDaemon(i, self.buffer)
            worker.start()
            self.workers.append(worker)

        # Print info
        sys.stdout.write(f"INFO: Generator initialization complete\n")
        sys.stdout.write(f"INFO: Number of inputs: {len(self.input_uris)}\n")
        sys.stdout.write(f"INFO: Output path: {self.outDir}\n")

    def check_inputs(self, inputs: List[str]):
        """
        Generator accepts input_paths of following form:
            - A single video file path
            - A single directory path containing multiple videos
            - A single directory path containing sequentially named png/jpg images
            - Multiple directories each of which containing multiple videos/images
            - Directories include wildcard
        """
        input_uris = []

        input_type = None
        for inputPath in inputs:
            pathList = glob.glob(os.path.expanduser(inputPath))
            if not pathList:
                sys.stderr.write("warn: got invalid input path: %s\n" % inputPath)
            for path in pathList:
                path = os.path.abspath(path)
                path_type, files = self._check_input_type(path)
                if input_type is None:
                    input_type = path_type
                    if input_type == InputType.UNKNOWN:
                        msg = f"error: no video or images found under: {path}"
                        raise RuntimeError(msg)
                elif input_type != path_type:
                    msg = "error: paths mixture videos and images are not allowed\n"
                    raise RuntimeError(msg)
                input_uris.extend(files)

        if not input_uris:
            msg = "error: invalid inputs"
            raise RuntimeError(msg)
        return input_type, input_uris

    @staticmethod
    def _check_input_type(path: str) -> Tuple[InputType, List[str]]:
        if os.path.isfile(path):
            if is_video(path):
                return InputType.VIDEO, ["file://" + path]
            if is_image(path):
                return InputType.IMAGE, [path]

        if os.path.isdir(path):
            fnames = os.listdir()
            videos = list(map(is_video, fnames))
            if len(fnames) == len(videos):
                return InputType.VIDEO, ["file://" + os.path.join(path, fn) for fn in fnames]
            images = list(map(is_image, fnames))
            if len(fnames) == len(images):
                return InputType.IMAGE, [os.path.join(path, fn) for fn in fnames]
            # Check path mix videos and images
            if len(videos) and len(images):
                msg = "error: path mixture videos and image are not allowed\n"
                raise RuntimeError(msg)

        return InputType.UNKNOWN, []

    @staticmethod
    def check_output(output_dir: str, subdir: str, cvt: Convertor):
        output_dir = os.path.abspath(os.path.expanduser(output_dir))
        if os.path.exists(output_dir):
            msg = f"error: Output dataset already exist: {output_dir}"
            raise RuntimeError(msg)
        dsRoot, dsName = os.path.split(output_dir)
        if not os.path.exists(dsRoot):
            msg = f"error: Output directory not exist {dsRoot}"
            raise RuntimeError(msg)
        if subdir:
            cvt.imgDirName = os.path.join(subdir, cvt.imgDirName)
            cvt.lblDirName = os.path.join(subdir, cvt.lblDirName)
        dsRoot = cvt.create_dataset(dsRoot, dsName)
        imgDir = os.path.join(dsRoot, cvt.imgDirName)
        lblDir = os.path.join(dsRoot, cvt.lblDirName)
        return dsRoot, imgDir, lblDir

    @staticmethod
    def check_models(models: List[str] = None):
        if not models: return models

        unique_ids = []
        model_list = []
        for i, model in enumerate(models):
            try:
                model = check_path(model)
            except FileNotFoundError:
                msg = f"error: model not exist: {model}"
                raise RuntimeError(msg)
            model_ini = configparser.ConfigParser()
            if not model_ini.read(model):
                msg = f"error: model not exist: {model}"
                raise RuntimeError(msg)
            try:
                uid = int(model_ini["property"]["gie-unique-id"])
            except KeyError as e:
                msg = f"error: incorrect model configs: {e}"
                raise RuntimeError(msg)
            if uid in unique_ids:
                msg = f"error: get duplicate `gie-unique-id={uid}` in model: {model}"
                raise RuntimeError(msg)
            unique_ids.append(uid)
            model_list.append(model)

        # Unique-ids must start at 1
        models_list = list(zip(model_list, unique_ids))
        models_list.sort(key=lambda elem: elem[1])
        if models_list[0][1] != 1:
            msg = "error: there must only one model that configured `gie-unique-id=1`"
            raise RuntimeError(msg)
        return [m[0] for m in models_list]

    def save_dot(self):
        if self._pipeline_dot:
            return
        dot_data = Gst.debug_bin_to_dot_data(self.pipeline, Gst.DebugGraphDetails.ALL)
        self._pipeline_dot = os.path.join(self.outDir, "pipeline.dot")
        with open(self._pipeline_dot, 'w', encoding="utf-8") as f:
            f.write(dot_data)
        print("[Success] Save dot: %s\n" % self._pipeline_dot)

    def run_loop(self):
        self.loop = GLib.MainLoop.new(None, False)
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", bus_call, self)

        GLib.timeout_add_seconds(1, watch_dog, self)

        self.pipeline.set_state(Gst.State.PLAYING)
        try:
            self.loop.run()
        except Exception:
            pass
        self.pipeline.set_state(Gst.State.NULL)

    def add_video_sources(self):
        global g_num_srcs
        global g_enabled_list
        global g_eos_list
        global g_uri_list
        global g_src_list
        global g_prc_list

        for i in range(self.max_srcs):
            if g_enabled_list[i]:
                continue
            if not g_eos_list[i]:
                continue
            try:
                uri = self.input_uris.pop()
            except IndexError:
                return

            src_bin = build_uri_source(i, uri, self.skip_mode, self.interval, self.gpu_id)
            prc_bin = build_preprocess(i, self.cam_shifts, self.crop_size, self.gpu_id, self.mem_type)
            self.pipeline.add(src_bin)
            self.pipeline.add(prc_bin)
            src_bin.link(prc_bin)
            for j in range(2):
                srcpad = prc_bin.get_static_pad("src_%u" % j)
                sinkpad = self.streammux.get_request_pad("sink_%u" % (2 * i + j))
                if not srcpad or not sinkpad:
                    sys.stderr.write("error: unable to link source %d with streamux\n" % i)
                    sys.exit(-1)
                srcpad.link(sinkpad)

            # Sync states for dynamic pipeline
            if not src_bin.sync_state_with_parent() or \
                    not prc_bin.sync_state_with_parent():
                sys.stderr.write("warn: failed on source(%d): %s\n" % (i, uri))
                self.pipeline.remove(src_bin)
                self.pipeline.remove(prc_bin)
                src_bin.set_state(Gst.State.NULL)
                prc_bin.set_state(Gst.State.NULL)
                continue

            sys.stdout.write("add new source(%d): %s\n" % (i, uri))
            g_enabled_list[i] = True
            g_eos_list[i] = False
            g_uri_list[i] = uri
            g_src_list[i] = src_bin
            g_prc_list[i] = prc_bin
            g_num_srcs += 1

    def delete_video_sources(self):
        global g_num_srcs
        global g_enabled_list
        global g_eos_list
        global g_uri_list
        global g_src_list
        global g_prc_list

        for i in range(self.max_srcs):
            if not g_eos_list[i]:
                continue
            if not g_enabled_list[i]:
                continue
            src_bin = g_src_list[i]
            prc_bin = g_prc_list[i]
            uri = g_uri_list[i]

            # Change state to release resources
            ret = src_bin.set_state(Gst.State.NULL)
            ret = prc_bin.set_state(Gst.State.NULL)
            if ret == Gst.StateChangeReturn.FAILURE:
                sys.stderr.write("warn: unable to release source(%d): %s\n" % (i, uri))
                continue
            if ret == Gst.StateChangeReturn.ASYNC:
                src_bin.get_state(Gst.CLOCK_TIME_NONE)
                prc_bin.get_state(Gst.CLOCK_TIME_NONE)
            self.pipeline.remove(src_bin)
            self.pipeline.remove(prc_bin)
            for j in range(2):
                sinkpad = self.streammux.get_static_pad("sink_%u" % (2 * i + j))
                sinkpad.send_event(Gst.Event.new_flush_stop(False))
                self.streammux.release_request_pad(sinkpad)

            sys.stdout.write("delete source(%d): %s\n" % (i, uri))
            g_num_srcs -= 1
            g_enabled_list[i] = False
            g_uri_list[i] = ""
            g_src_list[i] = None
            g_prc_list[i] = None

    def build_video_pipeline(self, show_pipe: bool = False):
        global g_cls_dict

        pipeline = Gst.Pipeline()
        if not pipeline:
            sys.stderr.write("error: unable to create pipeline\n")
            sys.exit(-1)

        # Build deepstream pipeline start by streammux
        streammux = Gst.ElementFactory.make("nvstreammux")
        if not streammux:
            sys.stderr.write("error: unable to create streammux")
            sys.exit(-1)
        streammux.set_property("width", self.crop_size)
        streammux.set_property("height", self.crop_size)
        streammux.set_property("batched-push-timeout", 2500)
        streammux.set_property("batch-size", 2 * self.max_srcs)
        streammux.set_property("gpu_id", self.gpu_id)
        streammux.set_property("nvbuf-memory-type", self.mem_type)
        streammux.set_property("sync-inputs", False)
        pipeline.add(streammux)

        last = streammux
        for i, model in enumerate(self.models):
            pgie = Gst.ElementFactory.make("nvinfer")
            if not pgie:
                sys.stderr.write("error: unable to create model: %s\n" % model)
                sys.exit(-1)
            pgie.set_property('config-file-path', model)
            pgie.set_property("batch-size", 2 * self.max_srcs)
            pgie.set_property("gpu-id", self.gpu_id)
            pgie.set_property("interval", 0)
            unique_id = pgie.get_property("unique-id")

            pipeline.add(pgie)
            last.link(pgie)
            last = pgie

            # Read class-names from label.txt file
            modelDir = os.path.dirname(model)
            try:
                labelTxt = os.path.join(modelDir, "labels.txt")
                check_path(labelTxt)
            except FileNotFoundError:
                msg = f"error: No `label.txt` file found under model path: {modelDir}"
                raise RuntimeError(msg)

            with open(labelTxt, 'r') as f:
                clsNames = [l.strip() for l in f.readlines() if l]
            g_cls_dict[unique_id] = clsNames

        if show_pipe:
            tile = Gst.ElementFactory.make("nvmultistreamtiler")
            conv = Gst.ElementFactory.make("nvvideoconvert")
            nosd = Gst.ElementFactory.make("nvdsosd")
            sink = Gst.ElementFactory.make("nveglglessink")
            if not tile or not conv or not nosd or not sink:
                sys.stderr.write("error: unable to create nveglglessink\n")
                sys.exit(-1)

            if self.max_srcs == 1:
                rows, cols = 1, 2
                width, height = 1920, 960
            elif self.max_srcs == 2:
                rows, cols = 2, 2
                width, height = 1080, 1080
            else:
                rows, cols = 2, 4
                width, height = 1920, 960
            tile.set_property("rows", rows)
            tile.set_property("columns", cols)
            tile.set_property("width", width)
            tile.set_property("height", height)
            tile.set_property("gpu-id", self.gpu_id)
            tile.set_property("nvbuf-memory-type", self.mem_type)
            sink.set_property("sync", False)

            pipeline.add(tile)
            pipeline.add(conv)
            pipeline.add(nosd)
            pipeline.add(sink)
            last.link(tile)
            tile.link(conv)
            conv.link(nosd)
            nosd.link(sink)
        else:
            sink = Gst.ElementFactory.make("fakesink")
            if not sink:
                sys.stderr.write("error: unable to create fakesink\n")
                sys.exit(-1)
            pipeline.add(sink)
            last.link(sink)

        self.pipeline = pipeline
        self.streammux = streammux
        srcpad = last.get_static_pad("src")
        srcpad.add_probe(Gst.PadProbeType.BUFFER, self.probe_func, self)

    def synchronize(self):

        while self.state == Gst.State.PLAYING and self.buffer.qsize() > g_sync_num:
            ret = self.pipeline.set_state(Gst.State.PAUSED)
            if ret == Gst.StateChangeReturn.FAILURE:
                sys.stderr.write("[WARN]: Failed to pause pipeline, sync stopped\n")
                break
            if ret == Gst.StateChangeReturn.ASYNC:
                ret = self.pipeline.get_state(10)
                if ret == Gst.StateChangeReturn.FAILURE:
                    sys.stderr.write("[WARN]: Failed to pause pipeline, sync stopped\n")
                    break

            total = self.buffer.qsize()
            progress = tqdm(total=total, desc="Sync buffer")
            while total > 0:
                time.sleep(1.0)
                current_size = self.buffer.qsize()
                if current_size > total:
                    progress.total = total = current_size
                else:
                    progress.update(total - current_size)
                    total = current_size
            progress.update(progress.total - progress.n)
            progress.close()
            self.pipeline.set_state(Gst.State.PLAYING)

    def wait_until_finish(self):
        for _ in range(self.num_workers):
            self.buffer.put(None)
        total = self.buffer.qsize()
        if total <= 0: return
        progress = tqdm(total=total, desc="Wait until finish")
        while total > 0:
            time.sleep(1.0)
            current_size = self.buffer.qsize()
            if current_size > total:
                progress.total = total = current_size
            else:
                progress.update(total - current_size)
                total = current_size
        progress.update(progress.total - progress.n)
        progress.close()
        for worker in self.workers:
            worker.join()

    def probe_func(self, pad: Gst.Pad, info: Gst.PadProbeInfo, user_data) -> Gst.PadProbeReturn:
        global g_count
        global g_cls_dict

        contiguous = self.contiguous
        drop_empty = self.drop_empty
        save_empty_label = self.save_empty_label
        imgPathFMT = self.imgPathFMT
        lblPathFMT = self.lblPathFMT
        buffer = self.buffer
        convert_label = self.convert_label

        gst_buffer = info.get_buffer()
        if not gst_buffer:
            sys.stderr.write("warn: unable to get buffer\n")
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            frame_idx = frame_meta.frame_num
            batch_idx = frame_meta.batch_id
            stream_idx = frame_meta.pad_index
            source_idx = stream_idx // 2
            width = frame_meta.source_frame_width
            height = frame_meta.source_frame_height

            # Traverse objects detected by gie
            boxes = []
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                obj_uid = obj_meta.unique_component_id
                cls_id = obj_meta.class_id
                clsName = g_cls_dict[obj_uid][cls_id]
                rect = obj_meta.rect_params
                left, top, w, h = rect.left, rect.top, rect.width, rect.height
                bbox = Box(rect.left, rect.top, left + w, top + h, clsName)
                boxes.append(bbox)

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            if boxes or not drop_empty:
                frame = pyds.get_nvds_buf_surface(hash(gst_buffer), batch_idx)
                frame = np.array(frame, copy=True, order='C')
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                if contiguous:
                    imgPath = imgPathFMT.format(count=g_count)
                    lblPath = lblPathFMT.format(count=g_count)
                else:
                    uri_name = os.path.basename(g_uri_list[source_idx]).split('.')[0]
                    imgPath = imgPathFMT.format(name=uri_name, fIdx=frame_idx, cIdx=stream_idx % 2)
                    lblPath = lblPathFMT.format(name=uri_name, fIdx=frame_idx, cIdx=stream_idx % 2)
                if not boxes and not save_empty_label:
                    lblPath = ""

                imgName = os.path.basename(imgPath)
                imgLbl = ImageLabel(imgName, boxes, width, height)
                lblInfo = convert_label(g_count, imgLbl)
                buffer.put((frame, lblInfo, imgPath, lblPath))
                g_count += 1

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK


def bus_call(bus, message, gen: Generator):
    global g_eos_list

    t = message.type
    if t == Gst.MessageType.EOS:
        sys.stdout.write("End-of-stream\n")
        gen.pipeline.set_state(Gst.State.NULL)
        gen.loop.quit()

    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        sys.stderr.write("Warning: %s: %s\n" % (err, debug))

    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write("Error: %s: %s\n" % (err, debug))
        gen.loop.quit()

    elif t == Gst.MessageType.ELEMENT:
        struct = message.get_structure()
        if struct is not None and struct.has_name("stream-eos"):
            parsed, stream_id = struct.get_uint("stream-id")
            if parsed:
                src_id = stream_id // 2
                g_eos_list[src_id] = True
                if stream_id % 2:
                    sys.stdout.write("Got EOS from source(%d)\n" % src_id)

    elif t == Gst.MessageType.STATE_CHANGED:
        old_state, new_state, _ = message.parse_state_changed()
        if message.src == gen.pipeline:
            gen.state = new_state
            if new_state == Gst.State.PLAYING:
                gen.save_dot()
    return True


def watch_dog(gen: Generator):
    global g_num_srcs

    gen.synchronize()

    gen.delete_video_sources()

    gen.add_video_sources()

    return g_num_srcs != 0 or len(gen.input_uris) > 0
