import numpy as np
import imageio.v3 as iio
from PIL import Image, ImageDraw
import sys
from collections import deque
from find_glow_balls import find_balls_2, preprocess

trail = Image.open('shock_trail_1.png')
trail_width = 60
trail_offset = 0
min_velocity = 10

infile, outfile = sys.argv[1:3]

frames = []
balls = []

for frame in iio.imiter(infile):
    img = Image.fromarray(frame)
    scale_factor, processed = preprocess(img)
    found = np.rot90(find_balls_2(processed).reshape((3,2)).T, 3)
    balls.append(found / scale_factor)
    draw = ImageDraw.Draw(img, mode='RGBA')
    frames.append(img)

tracked_balls = [[i] for i in balls[0]]
def tot_dist(ball):
    global tracked_balls
    last = np.array([i[-1] for i in tracked_balls])
    return np.cumsum(np.square(last - ball))
for i, ball in enumerate(balls)[1:]:
    idxs = [0, 0, 0]


iio.imwrite(outfile, np.array(frames), fps=iio.immeta(infile, exclude_applied=False)['fps'])
