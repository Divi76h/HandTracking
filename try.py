import matplotlib
from matplotlib import cm
import numpy as np
cmap = matplotlib.cm.get_cmap('gray')
volColor = 20.086798
volCmapR = np.interp(cmap(volColor)[0], [0.0, 1.0], [0, 255])
volCmapG = np.interp(cmap(volColor)[1], [0.0, 1.0], [0, 255])
volCmapB = np.interp(cmap(volColor)[2], [0.0, 1.0], [0, 255])
print(volCmapR, volCmapG, volCmapB)
