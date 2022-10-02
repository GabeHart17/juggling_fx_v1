import numpy as np
import sys
from PIL import Image
from find_glow_balls import *

img = Image.open('glow_still.png')
# img_arr = np.asarray(img.convert(mode='RGB'))
# print(img_arr.shape)
# print(np.add.reduce(img_arr,2).shape)
# scale_factor = 100 / img.width
# small_img = img.convert(mode='RGB').resize((round(scale_factor * img.width), round(scale_factor * img.height)))
# img_arr = np.asarray(small_img)
# balls = find_balls(img_arr).x
scale_factor, img_arr = preprocess(img)
balls = find_balls_2(img_arr)
print(balls / scale_factor)
