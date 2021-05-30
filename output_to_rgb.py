import numpy as np
from matplotlib import pyplot as plt
import sys
import matplotlib
from skimage.io import imread

if __name__ == "__main__":
    args = sys.argv
    if len(args) == 2:
        filepath = args[1]
    else:
        exit

    img = imread(filepath)

    viridis = matplotlib.cm.get_cmap('hsv', 256)
    newcolors = viridis(np.linspace(0, 1, 2048))
    whiteCm = np.array([1,1,1, 1])
    redCm = np.array([1,0,0, 1])

    newcolors[0, :] = whiteCm
    newcolors[1, :] = redCm
    newcmp = matplotlib.colors.ListedColormap(newcolors)

    plt.imsave("out_py.tiff", img, cmap=newcmp)
