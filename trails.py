import numpy as np
import imageio.v3 as iio
from PIL import Image, ImageDraw
import sys
from collections import deque
from find_glow_balls import find_balls_2, preprocess

# trail = '#5e90e0'
trail = '#ffffff'
# trail = '#00f8f8'
# trail = '#99ffff'
start_opacity = 128
# start_radius = 50
start_radius = 30
radius_increment = 5
delay_frames = 0
n_frames = 10

infile, outfile = sys.argv[1:3]

queue = deque(maxlen=n_frames+delay_frames)
frames = []

for frame in iio.imiter(infile):
    img = Image.fromarray(frame)
    scale_factor, processed = preprocess(img)
    balls = find_balls_2(processed) / scale_factor
    queue.appendleft(balls)
    draw = ImageDraw.Draw(img, mode='RGBA')
    for idx,ball in enumerate(queue):
        if idx < delay_frames: continue
        i = idx - delay_frames
        for x,y in ball:
            alpha = ('0'+hex(round(start_opacity * (n_frames-i) / n_frames))[2:])[-2:]
            fill_color = trail + alpha
            radius = start_radius + i * radius_increment
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=fill_color)
    # img.show()
    # print(frame.shape)
    # print(np.asarray(img).shape)
    # break
    frames.append(np.asarray(img))

iio.imwrite(outfile, np.array(frames), fps=iio.immeta(infile, exclude_applied=False)['fps'])
