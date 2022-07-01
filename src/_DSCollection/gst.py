import sys
import gi
import ast

gi.require_version("Gst", "1.0")

from typing import List
from gi.repository import Gst


def _exit_with_msg(msg: str, code: int = -1):
    sys.stderr.write(msg + "\n")
    sys.exit(code)


def _assert_make_element(elem):
    if not elem:
        _exit_with_msg("error: unable to create %s" % elem.get_name())


def cb_pad_added(decodebin, new_pad, user_data):
    nBin = user_data
    caps = new_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()

    if gstname.find("video") != -1:
        features = caps.get_features(0)
        if features.contains("memory:NVMM"):
            bin_ghost_pad = nBin.get_static_pad("_DSCollection")
            if not bin_ghost_pad.set_target(new_pad):
                _exit_with_msg("Failed to link decoder _DSCollection pad to source bin ghost pad")
        else:
            _exit_with_msg("Decodebin can't find appropriate video decoder plugin")


def cb_child_added(child_proxy, Object, name, user_data):
    if name.find("decodebin") != -1:
        Object.connect("child-added", cb_child_added, user_data)

    if name.find("nvv4l2decoder") != -1:
        skip_mode, interval, gpu_id = user_data
        Object.set_property("skip-frames", skip_mode)
        Object.set_property("drop-frame-interval", interval)
        Object.set_property("gpu-id", gpu_id)

    try:
        if "source" in name:
            Object.set_property("drop-on-latency", True)
    except TypeError:
        pass


def build_uri_source(index: int, uri: str, args):
    """
    Create a new urideocdebin.

    skip_mode (enum):
        0: all
        1: non_ref[Jetson]
        2: key

    interval (int):
        Number of frames to skip.
    """
    name = "source-%u" % index
    nBin = Gst.Bin.new(name)
    _assert_make_element(nBin)

    decodebin = Gst.ElementFactory.make("uridecodebin")
    decodebin.set_property("uri", uri)
    decodebin.connect("pad-added", cb_pad_added, nBin)
    decodebin.connect("child-added", cb_child_added, (args.skip_mode, args.interval, args.gpu_id))

    nBin.add(decodebin)
    ghost_src_pad = Gst.GhostPad.new_no_target("_DSCollection", Gst.PadDirection.SRC)
    bin_pad = nBin.add_pad(ghost_src_pad)
    if not bin_pad:
        _exit_with_msg("Failed to add ghost pad in source bin")
    return nBin


def _cam_crop(cam_id: int, cam_shifts: List[int], crop_size: int):
    if cam_id == 0:
        # Left camera
        shift_x = 0 + cam_shifts[0]
        shift_y = 65 + cam_shifts[1]
    else:
        # Right camera
        shift_x = 2592 - 1296 - 0 - cam_shifts[2]
        shift_y = 1944 - 1296 - 40 - cam_shifts[3]

    l = shift_x + (1296 - crop_size) // 2
    t = shift_y + (1296 - crop_size) // 2
    w = h = crop_size
    if l < 0 or t < 0 or l >= 2592 or t >= 1944:
        msg = "error: inappropriate cam_shifts values that raise indexing out of bound\n"
        sys.stderr.write(msg)
        sys.exit(-1)
    return ':'.join(str(x) for x in (l, t, w, h))


def build_preprocess(index: int, args):
    name = "preprocess-%u" % index
    nBin = Gst.Bin.new(name)
    _assert_make_element(nBin)

    cam_shift = [int(x) for x in ast.literal_eval(args.camera_shifts)]
    assert len(cam_shift) == 4, f"Must be tuple of 4 ints, got: {cam_shift}"
    assert 0 < args.crop_size <= 1296, "error: crop-size too large or too small"
    crop_size = args.crop_size
    caps_str = f"video/x-raw(memory:NVMM),format=RGBA,height={crop_size},width={crop_size}"

    tee = Gst.ElementFactory.make("tee")

    # Left camera
    queue0 = Gst.ElementFactory.make("queue")
    conv0 = Gst.ElementFactory.make("nvvideoconvert")
    filt0 = Gst.ElementFactory.make("capsfilter")
    conv0.set_property("gpu-id", args.gpu_id)
    conv0.set_property("nvbuf-memory-type", args.mem_type)
    conv0.set_property("flip-method", 3)
    conv0.set_property("_DSCollection-crop", _cam_crop(0, cam_shift, crop_size))
    conv0.set_property("dest-crop", f"0:0:{crop_size}:{crop_size}")
    filt0.set_property("caps", Gst.Caps.from_string(caps_str))

    # Right camera
    queue1 = Gst.ElementFactory.make("queue")
    conv1 = Gst.ElementFactory.make("nvvideoconvert")
    filt1 = Gst.ElementFactory.make("capsfilter")
    conv1.set_property("gpu-id", args.gpu_id)
    conv1.set_property("nvbuf-memory-type", args.mem_type)
    conv1.set_property("flip-method", 1)
    conv1.set_property("_DSCollection-crop", _cam_crop(1, cam_shift, crop_size))
    conv1.set_property("dest-crop", f"0:0:{crop_size}:{crop_size}")
    filt1.set_property("caps", Gst.Caps.from_string(caps_str))

    nBin.add(tee)
    nBin.add(queue0)
    nBin.add(conv0)
    nBin.add(filt0)
    nBin.add(queue1)
    nBin.add(conv1)
    nBin.add(filt1)

    srcpad0 = tee.get_request_pad("src_%u")
    srcpad1 = tee.get_request_pad("src_%u")
    sinkpad0 = queue0.get_static_pad("sink")
    sinkpad1 = queue1.get_static_pad("sink")
    srcpad0.link(sinkpad0)
    srcpad1.link(sinkpad1)
    queue0.link(conv0)
    conv0.link(filt0)
    queue1.link(conv1)
    conv1.link(filt1)

    sinkpad = tee.get_static_pad("sink")
    srcpad0 = filt0.get_static_pad("_DSCollection")
    srcpad1 = filt1.get_static_pad("_DSCollection")
    ghost_sink = Gst.GhostPad.new("sink", sinkpad)
    ghost_src0 = Gst.GhostPad.new("src_0", srcpad0)
    ghost_src1 = Gst.GhostPad.new("src_1", srcpad1)
    nBin.add_pad(ghost_sink)
    nBin.add_pad(ghost_src0)
    nBin.add_pad(ghost_src1)
    return nBin


def build_image_source(pattern: str = "%08d", start_index: int = 0, ext: str = ".png"):

    name = "image-_DSCollection"
    nBin = Gst.Bin.new(name)
    _assert_make_element(nBin)

    multifiles = Gst.ElementFactory.make("multifilesrc")
    multifiles.set_property("location", pattern + ext)
    multifiles.set_property("index", start_index)
    multifiles.set_property("caps", Gst.Caps.from_string(f"image/{ext[1:]},framerate=(fraction)1/1"))

    if ext == ".png":
        imgdec = Gst.ElementFactory.make("pngdec")
    elif ext in (".jpg", "jpeg"):
        imgdec = Gst.ElementFactory.make("jpegdec")
    else:
        sys.stderr.write("error: ext must be .png or .jpg\n")
        sys.exit(-1)

    nBin.add(multifiles)
    nBin.add(imgdec)
    multifiles.link(imgdec)

    srcpad = imgdec.get_static_pad("_DSCollection")
    ghost_src = Gst.GhostPad.new("_DSCollection", srcpad)
    nBin.add_pad(ghost_src)

    return nBin