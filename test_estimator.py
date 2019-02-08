import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.image
import matplotlib.pyplot as plt
plt.set_cmap("jet")

from PIL import Image
from estimator import MonoDepthEstimator


def main():
    in_file = "data/demo/img_nyu2.png"
    out_file = "data/demo/out.png"

    estimator = MonoDepthEstimator()    
    im = Image.open(in_file)
    depth = estimator.compute_depth(im)
    matplotlib.image.imsave(out_file, depth)

if __name__ == '__main__':
    main()
