import numpy as np
import imageio.v3 as iio
from PIL import Image, ImageDraw
from find_glow_balls import find_balls_2, preprocess

frames = []

for frame in iio.imiter('glow_video.mp4'):
    img = Image.fromarray(frame)
    scale_factor, processed = preprocess(img)
    balls = find_balls_2(processed) / scale_factor
    draw = ImageDraw.Draw(img)
    for x,y in balls:
        # print(x,y)
        draw.ellipse([x-20, y-20, x+20, y+20], fill='limegreen')
    # img.show()
    # print(frame.shape)
    # print(np.asarray(img).shape)
    # break
    frames.append(np.asarray(img))

iio.imwrite("glow_processed.mp4", np.array(frames), fps=iio.immeta('glow_video.mp4', exclude_applied=False)['fps'])
