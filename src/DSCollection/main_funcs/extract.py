from itertools import zip_longest

from ..utils.tasks import TaskDispatcher, TASK
from ..core.extractor import DataExtractor
from ..core.convertor import Convertor
from ..core.dataset import Dataset


@TaskDispatcher(TASK.EXTRACT)
def main(args):
    if args.classes:
        if args.remap and len(args.remap) != len(args.classes):
            msg = f"The number of remap names are not equal to the number of classes: " \
                  f"{len(args.remap)} != {len(args.classes)}"
            raise RuntimeError(msg)
        args.remap = dict(zip_longest(args.classes, args.remap, fillvalue=""))
    else:
        if args.remap:
            msg = f"Extract classes are empty, while remap to new names is ambiguous."
            raise RuntimeError(msg)
        args.remap = None

    try:
        cvt = Convertor.from_type(args.dtype, args.remap)
    except ValueError:
        msg = f"Unknown convertor type: {args.dtype}"
        raise RuntimeError(msg)
    e = DataExtractor(cvt, args.ext)

    input_datasets = []
    for input_root in args.input:
        DataKlass = Dataset.find_dataset(input_root)
        if DataKlass is None:
            msg = "Unknown dataset: %s" % input_root
            raise RuntimeError(msg)

        if args.kwargs is not None:
            # Make sure kwargs not contain split and year
            split = args.kwargs.pop("split", args.split)
            year = args.kwargs.pop("year", args.year)
        else:
            args.kwargs = {}
            split = args.split
            year = args.year
        input_datasets.append(DataKlass(input_root, split=split, year=year, **args.kwargs))

    for ds in input_datasets:
        e.extract(ds, args.output, None, args.classes, args.n_imgs, args.contiguous)
