from ..utils.tasks import TaskDispatcher, TASK
from ..core.visualizer import Visualizer
from ..core.dataset import Dataset


def get_colormap_info() -> str:
    return """Colormaps in Matplotlib
------------------------
#1 Sequential:
    - Perceptually Uniform Sequential:
        'viridis', 'plasma', 'inferno', 'magma', 'cividis'
    - Sequential
        'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
        'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
        'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'
    
#2 Sequential2:
    'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
    'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
    'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper'
    
#3 Diverging
    'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
    'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'
    
#4 Cyclic
    'twilight', 'twilight_shifted', 'hsv'
    
#5 Qualitative
    'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
    'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b',
    'tab20c'

#6 Miscellaneous
    'flag', 'prism', 'ocean', 'gist_earth', 'terrain',
    'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap',
    'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet',
    'turbo', 'nipy_spectral', 'gist_ncar'

Refer to https://matplotlib.org/stable/tutorials/colors/colormaps.html for more details.\n
"""


@TaskDispatcher(TASK.VISUALIZE)
def main(args) -> int:
    if isinstance(args.input, list):
        args.input = args.input[0]
    DataKlass = Dataset.find_dataset(args.input)
    if DataKlass is None:
        return -1
    else:
        kwargs = {"winName": args.win_name, "rows": args.rows, "cols": args.cols}
        vis = Visualizer(args.backend, args.cmap, **kwargs)
        ds = DataKlass(args.input)
        vis.show(ds, args.class_names, args.num_images, args.labels, args.output)
        return 0
