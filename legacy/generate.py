import os
import ast
import sys
import glob
import argparse
import cv2
import gi
import pyds
import numpy as np

from collections import deque
from concurrent.futures import ProcessPoolExecutor, Future
from typing import List, Dict, Any, Callable, Optional, Deque

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib, GObject
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_tools import BBox, Dataset, get_dataset
from gst_bins import build_uri_source, build_image_source, build_preprocess

g_count: int = 0                    # The number of consecutive frames that have been processed
g_num_srcs: int = 0                 # Number of sources already in pipeline
g_uri_stack: List[str]              # Dynamic array containing all URIs to be processed
g_uri_list: List[str]               # Static array containing current URIs
g_enabled_list: List[bool]          # Static array containing flags that indicate corresponding sources are enabled
g_eos_list: List[bool]              # Static array containing flags that indicate corresponding sources reached eos
g_src_list: List[Gst.Element]       # Static array containing source element
g_prc_list: List[Gst.Element]       # Static array containing preprocess element
g_gie_probe_list: List[Callable]
g_ann_list: List[Dict[int, Dict[int, Dict[str, Any]]]]  # [ { frameIdx: { streamIdx: Dict[str, Any] } } ]
g_dataset: Dataset

pipeline: Gst.Pipeline
loop: GLib.MainLoop
streammux: Gst.Element

CLEAN_NUM: int = 1000
executor: Optional[ProcessPoolExecutor] = None
futures: Deque[Future] = deque()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--type", type=str, default="video",
                        help="Input path type, must be one of `video` or `image`.")
    parser.add_argument("-i", "--input", nargs='*', required=True, help="Input video/images or directory path.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory.")
    parser.add_argument("-n", "--name", type=str, default="GeneratedDateset", help="Name of generated dataset.")
    parser.add_argument("-s", "--show", action="store_true", help="Show pipeline.")
    parser.add_argument("-m", "--model", action="append", default=[], help="Model config file.")
    parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count(), help="Number of subprocesses to generate.")
    parser.add_argument("-c", "--contiguous", action="store_true", help="Save images and labels sequentially.")

    parser.add_argument("--ext", type=str, default=".png", help="Image file extension (default .png).")
    parser.add_argument("--format", type=str, default="kitti", help="Dataset format: 'voc' | 'kitti' (default).")
    parser.add_argument("--crop-size", type=int, default=1296, help="Center crop size (default: 1296).")
    parser.add_argument("--offset", type=int, default=0,
                        help="Offset of the shorter size for roi cropping (default: 0).")
    parser.add_argument("--camera-shifts", type=str, default="(0,0,0,0)",
                        help="Offsets of camera cropping (default (0,0,0,0)).")
    parser.add_argument("--grayscale", action="store_true", help="Store grayscale image")
    parser.add_argument("--keep-empty-label", action="store_true", help="Whether to keep label if it's empty.")
    parser.add_argument("--inplace", action="store_true", help="Merge generated dataset labels.")

    parser.add_argument("--max-srcs", type=int, default=2, help="Maximum number of sources (default: 2).")
    parser.add_argument("--skip-mode", type=int, default=0,
                        help="Frame skipping mode. [0:all(default), 1:non_ref[Jetson], 2:key]")
    parser.add_argument("--interval", type=int, default=2, help="Frame interval to decode (default: 0).")
    parser.add_argument("--mem-type", type=int, default=3, help="NvBuf memory type.")
    parser.add_argument("--gpu-id", type=int, default=0, metavar="GPU_ID")

    return parser.parse_args()


def wait_until_finish():
    global futures
    if not futures: return
    progress = tqdm(total=len(futures), desc="clean-up")
    while len(futures):
        future = futures.popleft()
        future.result()
        progress.update()


def clear_finished():
    global futures
    while len(futures):
        future = futures.popleft()
        if not future.done():
            futures.appendleft(future)
            break


class PipelineDot:
    """
    Save Gst pipeline only once.
    """
    dot_saved: bool = False

    @classmethod
    def save(cls, pipeline: Gst.Pipeline, path: str = "pipeline.dot", ):
        if cls.dot_saved: return
        cls.dot_saved = True
        dot_data = Gst.debug_bin_to_dot_data(pipeline, Gst.DebugGraphDetails.ALL)
        with open(path, 'w', encoding="utf-8") as f:
            f.write(dot_data)
        print("[Success] Save dot: %s\n" % path)


class ProbeFunction:

    def __init__(self, gie_id: int):
        self.gie_id = gie_id
        self.is_last = False
        self.is_first = False

    def __call__(self, pad: Gst.Pad, info: Gst.PadProbeInfo, args) -> Gst.PadProbeReturn:
        global g_count
        global g_dataset
        global g_ann_list
        global executor
        global futures

        buffer = info.get_buffer()
        if not buffer:
            sys.stderr.write("warn: unable to get buffer\n")
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            frame_idx = frame_meta.frame_num
            stream_idx = frame_meta.pad_index
            source_id = stream_idx // 2

            # Get or initialize annotation dict
            source_dict = g_ann_list[source_id]
            frame_dict = source_dict.setdefault(frame_idx, {})
            stream_dict = frame_dict.setdefault(frame_idx, {})
            if not stream_dict:
                uri_name = os.path.basename(os.path.splitext(g_uri_list[source_id])[0])
                stream_dict["folder"] = args.name
                stream_dict["filename"] = "{}-{}-{}{}".format(uri_name, frame_idx, stream_idx, args.ext)
                stream_dict["object"] = []
                stream_dict["size"] = size_dict = {}
                size_dict["width"] = frame_meta.source_frame_width
                size_dict["height"] = frame_meta.source_frame_height
                size_dict["depth"] = 1 if args.grayscale else 3

            # Traverse objects detected by this gie
            objects = stream_dict["object"]
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                text_params = obj_meta.text_params
                class_name = pyds.get_string(text_params.display_text)
                if obj_meta.unique_component_id == self.gie_id:
                    rect = obj_meta.rect_params
                    width, height = rect.width, rect.height
                    bbox = BBox.from_xywh(rect.left, rect.top, width, height, class_name)
                    objects.append(bbox)

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            # Save image and label if this is the last probe function
            if self.is_last:
                frame = pyds.get_nvds_buf_surface(hash(buffer), frame_meta.batch_id)
                frame = np.array(frame, copy=True, order='C')
                if args.grayscale:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

                if len(stream_dict["object"]) or args.keep_empty_label:
                    if args.contiguous:
                        stream_dict["filename"] = "{:08d}{}".format(g_count, args.ext)
                        g_count += 1
                    if executor is None:
                        g_dataset.save_all(stream_dict, frame, args.inplace)
                    else:
                        future = executor.submit(g_dataset.save_all, stream_dict, frame, args.inplace)
                        futures.append(future)

                # since annotation of this frame is already saved,
                # this annotation dict can be popped out to save memory
                g_ann_list[source_id].pop(frame_idx)

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK


def initialize(args):
    """
    Initialize Gstreamer and parse inputs to a list of URIs.

    Acceptable input formats:
       - A single video file path
       - A single image file path
       - A single directory path containing multiple videos
       - A single directory path containing sequentially named png/jpg images
       - Multiple directories echo of which containing multiple videos/images
    """
    global g_uri_stack
    global g_uri_list
    global g_enabled_list
    global g_eos_list
    global g_src_list
    global g_prc_list
    global g_dataset
    global g_ann_list
    global g_gie_probe_list
    global executor

    assert args.type in ("video", "image"), "error: type must be `video` or `image`"

    # Initialization of Gstreamer
    Gst.init(None)

    # Initialization of input path
    if args.type == "video":
        g_uri_stack = []
        for input_path in args.input:
            paths = glob.glob(input_path)
            for path in paths:
                if os.path.isfile(path):
                    g_uri_stack.append(f"file://{os.path.abspath(os.path.expanduser(path))}")
                elif os.path.isdir(path):
                    g_uri_stack.extend([f"file://{os.path.abspath(os.path.expanduser(p))}"
                                        for p in os.listdir(path)])
        if not g_uri_stack:
            sys.stderr.write("error: empty uri! check input\n")
            sys.exit(-1)
    else:
        if len(args.input) != 2:
            msg = "error: input must include sequential named pattern and start index\n\t" \
                  "eg. -i img-%08d.png 0\n"
            sys.stderr.write(msg)
            sys.exit(-1)

    if args.max_srcs == 1 and len(g_uri_stack) > 1:
        args.max_srcs = min(4, len(g_uri_stack))
        sys.stderr.write("warn: max_srcs=1 while multiple sources found\n")

    # Initialization of global states
    g_enabled_list = [False] * args.max_srcs
    g_eos_list = [True] * args.max_srcs
    g_uri_list = [""] * args.max_srcs
    g_src_list = [None] * args.max_srcs
    g_prc_list = [None] * args.max_srcs
    g_ann_list = [None] * args.max_srcs
    g_gie_probe_list = [None] * len(args.model)

    # Initialization of output path
    if args.inplace and not args.output:
        if os.path.isdir(args.input):
            args.output = args.input
        elif os.path.isfile(args.input):
            args.output = os.path.dirname(args.input)
        else:
            sys.stderr.write("error: output path not specified\n")
            sys.exit(-1)

    output_path = os.path.abspath(os.path.expanduser(args.output))
    dir_name, base_name = os.path.split(output_path)
    if not os.path.exists(dir_name):
        sys.stderr.write("error: output path not exist\n")
        sys.exit(-1)
    if os.path.exists(output_path) and base_name == args.name and not args.inplace:
        sys.stderr.write("error: output dataset realdy exist\n")
        sys.exit(-1)
    if not os.path.exists(output_path) or base_name == args.name or args.inplace:
        os.makedirs(output_path, exist_ok=True)
        args.output = dir_name
        args.name = base_name
    else:
        args.output = output_path
        os.makedirs(os.path.join(output_path, args.name), exist_ok=True)

    # Initialize dataset
    g_dataset = get_dataset(args.format, args.jobs)
    g_dataset.img_ext = args.ext
    g_dataset.create_directory(args.output, args.name)

    # Initialize multiple jobs
    if args.jobs > 1:
        executor = ProcessPoolExecutor(max_workers=args.jobs - 1)

    # Print info
    sys.stdout.write(f"Number of inputs: {len(g_uri_stack)}\n")
    sys.stdout.write(f"Output path: {args.output}/{args.name}\n")


def add_video_sources(args):
    global g_num_srcs
    global g_uri_stack
    global g_enabled_list
    global g_eos_list
    global g_uri_list
    global g_src_list
    global g_prc_list
    global g_ann_list
    global pipeline
    global streammux

    for i in range(args.max_srcs):
        if g_enabled_list[i]:
            continue
        if not g_eos_list[i]:
            continue
        try:
            uri = g_uri_stack.pop()
        except IndexError:
            return

        src_bin = build_uri_source(i, uri, args)
        prc_bin = build_preprocess(i, args)

        pipeline.add(src_bin)
        pipeline.add(prc_bin)
        src_bin.link(prc_bin)
        for j in range(2):
            srcpad = prc_bin.get_static_pad("src_%u" % j)
            sinkpad = streammux.get_request_pad("sink_%u" % (2 * i + j))
            if not srcpad or not sinkpad:
                sys.stderr.write("error: unable to link source %d with streamux\n" % i)
                sys.exit(-1)
            srcpad.link(sinkpad)

        # Sync states for dynamic pipeline
        if not src_bin.sync_state_with_parent() or \
                not prc_bin.sync_state_with_parent():
            sys.stderr.write("warn: failed on source(%d): %s\n" % (i, uri))
            pipeline.remove(src_bin)
            pipeline.remove(prc_bin)
            src_bin.set_state(Gst.State.NULL)
            prc_bin.set_state(Gst.State.NULL)
            continue

        sys.stdout.write("add new source(%d): %s\n" % (i, uri))
        g_enabled_list[i] = True
        g_eos_list[i] = False
        g_uri_list[i] = uri
        g_src_list[i] = src_bin
        g_prc_list[i] = prc_bin
        g_ann_list[i] = {}
        g_num_srcs += 1


def delete_video_sources(args):
    global g_num_srcs
    global g_enabled_list
    global g_eos_list
    global g_uri_list
    global g_src_list
    global g_prc_list
    global g_ann_list
    global pipeline
    global streammux

    for i in range(args.max_srcs):
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
        pipeline.remove(src_bin)
        pipeline.remove(prc_bin)
        for j in range(2):
            sinkpad = streammux.get_static_pad("sink_%u" % (2 * i + j))
            sinkpad.send_event(Gst.Event.new_flush_stop(False))
            streammux.release_request_pad(sinkpad)

        sys.stdout.write("delete source(%d): %s\n" % (i, uri))
        g_num_srcs -= 1
        g_enabled_list[i] = False
        g_uri_list[i] = ""
        g_src_list[i] = None
        g_prc_list[i] = None
        g_ann_list[i] = None


def add_image_source(args):
    global pipeline
    global streammux

    img_src = build_image_source()



def build_pipeline(args):

    global g_gie_probe_list
    global pipeline
    global streammux

    pipeline = Gst.Pipeline()
    if not pipeline:
        sys.stderr.write("error: unable to create pipeline\n")
        sys.exit(-1)

    # Build deepstream pipeline start by streammux
    streammux = Gst.ElementFactory.make("nvstreammux")
    if not streammux:
        sys.stderr.write("error: unable to create streammux")
        sys.exit(-1)
    streammux.set_property("width", args.crop_size)
    streammux.set_property("height", args.crop_size)
    streammux.set_property("batched-push-timeout", 2500)
    streammux.set_property("batch-size", 2 * args.max_srcs)
    streammux.set_property("sync-inputs", False)
    streammux.set_property("gpu_id", args.gpu_id)
    streammux.set_property("nvbuf-memory-type", args.mem_type)
    pipeline.add(streammux)

    last = streammux
    for i, model in enumerate(args.model):
        pgie = Gst.ElementFactory.make("nvinfer")
        if not pgie:
            sys.stderr.write("error: unable to create model: %s\n" % model)
            sys.exit(-1)
        pgie.set_property('config-file-path', model)
        pgie.set_property("batch-size", 2 * args.max_srcs)
        pgie.set_property("gpu-id", args.gpu_id)
        pgie.set_property("interval", 0)
        unique_id = pgie.get_property("unique-id")

        pipeline.add(pgie)
        last.link(pgie)
        last = pgie

        # Attach probe on this nvinfer srcpad
        probe_func = ProbeFunction(unique_id)
        probe_func.is_last = (i == len(args.model) - 1)
        probe_func.is_first = (i == 0)
        srcpad = pgie.get_static_pad("_DSCollection")
        srcpad.add_probe(Gst.PadProbeType.BUFFER, probe_func, args)
        g_gie_probe_list[i] = probe_func

    # Make sure add at least one probe function on last element
    if all(func is None for func in g_gie_probe_list):
        src_pad = last.get_static_pad("_DSCollection")
        probe_func = ProbeFunction(-1)
        probe_func.is_last = True
        probe_func.is_first = True
        src_pad.add_probe(Gst.PadProbeType.BUFFER, probe_func, args)

    if args.show:
        tile = Gst.ElementFactory.make("nvmultistreamtiler")
        conv = Gst.ElementFactory.make("nvvideoconvert")
        nosd = Gst.ElementFactory.make("nvdsosd")
        sink = Gst.ElementFactory.make("nveglglessink")
        if not tile or not conv or not sink:
            sys.stderr.write("error: unable to create nveglglessink\n")
            sys.exit(-1)

        if args.max_srcs == 1:
            rows, cols = 1, 2
            width, height = 1920, 960
        elif args.max_srcs == 2:
            rows, cols = 2, 2
            width, height = 1080, 1080
        else:
            rows, cols = 2, 4
            width, height = 1920, 960
        tile.set_property("rows", rows)
        tile.set_property("columns", cols)
        tile.set_property("width", width)
        tile.set_property("height", height)
        tile.set_property("gpu-id", args.gpu_id)
        tile.set_property("nvbuf-memory-type", args.mem_type)
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


def bus_call(bus, message, args):
    global g_uri_stack
    global g_eos_list
    global pipeline
    global loop

    t = message.type
    if t == Gst.MessageType.EOS:
        sys.stdout.write("End-of-stream\n")
        if not len(g_uri_stack):
            loop.quit()
        else:
            sys.stdout.write("Restart pipeline\n")
            pipeline.set_state(Gst.State.NULL)
            pipeline.set_state(Gst.State.PLAYING)

    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        sys.stderr.write("Warning: %s: %s\n" % (err, debug))

    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write("Error: %s: %s\n" % (err, debug))
        loop.quit()

    elif t == Gst.MessageType.ELEMENT:
        struct = message.get_structure()
        if struct is not None and struct.has_name("stream-eos"):
            parsed, stream_id = struct.get_uint("stream-id")
            if parsed:
                src_id = stream_id // 2
                sys.stdout.write("\rGot EOS from stream(%d)\n" % stream_id)
                g_eos_list[src_id] = True

    elif t == Gst.MessageType.STATE_CHANGED:
        old_state, new_state, _ = message.parse_state_changed()
        if message.src == pipeline:
            old_state_name = Gst.Element.state_get_name(old_state)
            new_state_name = Gst.Element.state_get_name(new_state)
            print(f"State changed {old_state_name} -> {new_state_name}")
            if new_state == Gst.State.PLAYING and args.show:
                PipelineDot.save(pipeline)

    return True


def watch_dog(args):
    global g_num_srcs
    global g_uri_stack
    global futures
    global CLEAN_NUM

    if len(futures) >= CLEAN_NUM:
        wait_until_finish()
    else:
        clear_finished()

    delete_video_sources(args)

    add_video_sources(args)

    return g_num_srcs != 0 or len(g_uri_stack) > 0


def run_loop(args):
    global pipeline
    global loop
    global g_dataset
    global executor
    global futures

    loop = GLib.MainLoop.new(None, False)
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, args)

    if args.type == "video":
        GLib.timeout_add_seconds(1, watch_dog, args)
    else:
        add_image_source(args)

    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except Exception:
        pass
    pipeline.set_state(Gst.State.NULL)


def main(args):
    global executor

    initialize(args)

    build_pipeline(args)

    run_loop(args)

    wait_until_finish()

    executor.shutdown()


if __name__ == '__main__':
    sys.exit(main(get_args()))
