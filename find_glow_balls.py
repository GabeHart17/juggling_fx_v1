import numpy as np
from PIL import Image
from scipy.optimize import minimize
from sklearn.cluster import KMeans

"""
objective: find (y1, x1, y2, x2, y3, x3) of the balls
note: images in numpy as used here are y,x
"""

# img is pillow image
def preprocess(img):
    scale_factor = 100 / img.width
    small_img = img.convert(mode='RGB').resize((round(scale_factor * img.width), round(scale_factor * img.height)))
    img_arr = np.asarray(small_img)
    return scale_factor, img_arr

# img is np array rgb image
def find_balls(img):
    # flat_img = np.add.reduce(img, 2) / (255 * 3)  # prevent int overflow by normalizing pixels
    flat_img = np.equal(np.add.reduce(img, 2), 255 * 3)
    ys, xs = np.mgrid[:flat_img.shape[0],:flat_img.shape[1]]
    def cost(coords):
        nonlocal flat_img, ys, xs
        d1 = np.square(ys - coords[0]) + np.square(xs-coords[1])
        d2 = np.square(ys - coords[2]) + np.square(xs-coords[3])
        d3 = np.square(ys - coords[4]) + np.square(xs-coords[5])
        d_sq_min = np.minimum(d1, np.minimum(d2, d3))
        d_min = np.sqrt(d_sq_min)
        weighted = np.multiply(d_min, flat_img)
        # weighted = np.multiply(d_min, np.square(flat_img))
        # weighted = np.multiply(d_min, np.exp(flat_img))
        # weighted = np.multiply(d_sq_min, flat_img)
        # weighted = np.multiply(d_sq_min, np.square(flat_img))
        return np.add.reduce(np.add.reduce(weighted))
    start = np.linspace([0,0], flat_img.shape, num=5).flatten()[2:-2]
    # start = np.array([538.59526669, 249.50027682, 809.54870894, 380.74938588, 711.18485179, 558.53803892])
    bounds_array = zip([0] * 6, np.array([flat_img.shape]*3).flatten())
    return minimize(cost, start, method='nelder-mead',
                    options={'xatol': 1e-8, 'disp': True},
                    bounds=bounds_array)

# img is np array rgb image
def find_balls_2(img):
    thresholded = np.equal(np.add.reduce(img, 2), 255 * 3)
    ys, xs = np.mgrid[:thresholded.shape[0],:thresholded.shape[1]]
    coords = np.array([xs.flatten(), ys.flatten()]).T
    bright_coords = coords[thresholded.flatten()]
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(bright_coords)
    return kmeans.cluster_centers_
