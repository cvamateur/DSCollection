from . import combine
from . import extract
from . import split
from . import visualize
from . import process

try:
    from . import generate
except ImportError:
    # import sys
    # sys.stderr.write("warn: unfit dependencies, task `generate` ignored.\n")
    pass

try:
    from . import augment
except ImportError:
    # import sys
    # sys.stderr.write("warn: unfit dependencies, task `augment` ignored.\n")
    pass
