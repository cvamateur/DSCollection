#!/usr/bin/env python3
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.DSCollection import TaskDispatcher, get_cli_parser


def main():
    args = get_cli_parser().parse_args()
    task_main_func = TaskDispatcher.get_task_handler(args.task)
    return task_main_func(args)


if __name__ == '__main__':
    sys.exit(main())
