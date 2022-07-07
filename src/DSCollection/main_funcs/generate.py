import sys

from ..utils.tasks import TaskDispatcher, TASK
from ..core.generator import Generator, InputType


def check_dependencies():
    """
    This task requires gst-python and deepstream-python.
    """
    try:
        import pyds
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst, GObject, GLib
    except ImportError:
        sys.stderr.write("error: gst-python and deepstream-python(pyds) is required!\n")
        sys.exit(-1)


@TaskDispatcher(TASK.GENERATE)
def main(args):
    check_dependencies()

    g = Generator(args.input, args.output, args.models, args.dtype, args.ext,
                  args.max_srcs, args.skip_mode, args.interval, args.gpu_id,
                  args.camera_shifts, args.crop_size, args.memory_type,
                  args.num_workers, args.contiguous)

    if g.input_type == InputType.VIDEO:
        g.build_video_pipeline(args.show)
    else:
        sys.stderr.write(f"error: Input type not supported: {g.input_type}\n")
        sys.exit(-1)

    g.run_loop()

    g.wait_until_finish()
