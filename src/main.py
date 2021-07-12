import os
import argparse

from src.app.app import App
from src.utils.utils import check_mozart_spelling

TIMESTEP = 0.25
SEQUENCE_LEN = 100


def arg_parser():
    parser = argparse.ArgumentParser(
        description='Train a Bi-LSTM Attention LSTM neural network')
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    required.add_argument('-p', '--path', dest='path',
                          type=dir_path, required=True)
    required.add_argument('-c', '--composers', dest='composers',
                          nargs='+', type=str, required=True)
    args = parser.parse_args()

    composers_list = list(map(str.lower, args.composers))
    check_mozart_spelling(composers_list)

    if os.path.isabs(args.path):
        dataset_path = args.path
    else:
        dataset_path = os.path.abspath(args.path)
    return composers_list, dataset_path


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def main(path, composers_list):
    app = App(composers_list, path)
    app.run()


if __name__ == '__main__':
    composers, data_path = arg_parser()
    main(data_path, composers_list=composers)
