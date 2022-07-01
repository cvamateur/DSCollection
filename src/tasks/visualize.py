import argparse

from .._DSCollection.visualizer import Visualizer


def get_args():

    parser = argparse.ArgumentParser(description="Visualization launcher.")

    parser.add_argument("-i", "--input", required=True, nargs='*', help="Input dataset root directory.")
    parser.add_argument("-o", "--output", help="Output directory.")



def main(args):

    vis = Visualizer()

    vis.show()