# @Time    : 2022/7/6 下午1:49
# @Author  : Boyang
# @Site    : 
# @File    : process.py
# @Software: PyCharm
from ..utils.tasks import TASK, TaskDispatcher
from ..core.process import Process


@TaskDispatcher(TASK.PROCESS)
def main(args):
    p = Process(args.input, dst_dtype=args.dtype, output=args.output, force=args.force)
    p.run(args)
