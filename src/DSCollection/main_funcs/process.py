# @Time    : 2022/7/6 下午1:49
# @Author  : Boyang
# @Site    : 
# @File    : process.py
# @Software: PyCharm
import glob
import os

from ..utils.tasks import TASK, TaskDispatcher
from ..core.process import Process
from ..core.dataset import Dataset
from ..utils.common import check_path


@TaskDispatcher(TASK.PROCESS)
def main(args):
    p = Process(args.input, dst_dtype=args.dtype, output=args.output, force=args.force, class_map=args.class_map)
    p.run(args)
