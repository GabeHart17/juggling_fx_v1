import numpy as np
import imageio.v3 as iio
from PIL import Image, ImageDraw, ImageChops
import sys
from collections import deque
import random
import math
from find_glow_balls import find_balls_2, preprocess

print('initializing')

infile, outfile = sys.argv[1:3]

# base_particle_image = Image.open('blue_shard.png')
# min_size = 10
# max_size = 70
# per_frame = 16
# min_lifetime = 10
# max_lifetime = 30
# jitter = 30
# max_velocity = 7
# min_velocity = 1
# start_alpha = 190
# end_alpha = 0
# directional = True
base_particle_image = Image.open('multi_sparkle.png')
min_size = 50
max_size = 100
per_frame = 5
min_lifetime = 5
max_lifetime = 20
jitter = 15
max_velocity = 5
min_velocity = 1
start_alpha = 190
end_alpha = 50
directional = False

alpha_decrease = Image.new('RGBA',
                           (base_particle_image.width, base_particle_image.height),
                           color=(0, 0, 0, 255 - start_alpha))
particle_image = ImageChops.subtract(base_particle_image, alpha_decrease)

class Particle:
    def __init__(self, image, x, y, width, height):
        global min_size, max_size, min_lifetime, max_lifetime, jitter, max_velocity, min_velocity, start_alpha, end_alpha
        self.frames_remaining = random.randint(min_lifetime, max_lifetime)
        self.size = random.randint(min_size, max_size)
        self.orientation = random.randint(0, 359)
        self.x = max(0, min(width - 1,x + random.randint(-jitter, jitter)))
        self.y = max(0, min(height - 1,y + random.randint(-jitter, jitter)))
        self.float_x = float(self.x)
        self.float_y = float(self.y)
        scaled = image.resize((self.size, round(self.size * (image.height / image.width))), reducing_gap=2.0)
        rotated = scaled.rotate(-self.orientation, expand=True)
        self.image = rotated
        self.alpha_increment = round((end_alpha - start_alpha) / self.frames_remaining)
        self.alpha_image = Image.new('RGBA',
                                   (self.image.width, self.image.height),
                                   color=(0, 0, 0, abs(self.alpha_increment)))
        vel = random.uniform(min_velocity, max_velocity)
        self.direction = self.orientation if directional else random.randint(0, 359)
        self.x_vel = math.cos(math.radians(self.direction)) * vel
        self.y_vel = math.sin(math.radians(self.direction)) * vel

    def update(self):
        self.frames_remaining -= 1
        self.float_x += self.x_vel
        self.float_y += self.y_vel
        self.x = round(self.float_x)
        self.y = round(self.float_y)
        if self.alpha_increment < 0:
            self.image = ImageChops.subtract(self.image, self.alpha_image)
        elif self.alpha_increment > 0:
            self.image = ImageChops.add(self.image, self.alpha_image)


particles = []
frames = []
balls = []

print('finding juggling balls')

for frame in iio.imiter(infile):
    img = Image.fromarray(frame)
    scale_factor, processed = preprocess(img)
    # found = np.rot90(find_balls_2(processed).reshape((3,2)).T, 3)
    found = find_balls_2(processed)
    balls.append(found / scale_factor)
    frames.append(img)

print('applying effects')

for frame, ball_coords in zip(frames, balls):
    # draw = ImageDraw.Draw(frame, mode='RGBA')
    for ball in ball_coords:
        for i in range(per_frame):
            particles.append(Particle(particle_image, ball[0], ball[1], frame.width, frame.height))
    for particle in particles:
        frame.paste(particle.image,
                    box=(round(particle.x - particle.image.width / 2),
                         round(particle.y - particle.image.height / 2)),
                    mask=particle.image)
        particle.update()
    particles = list(filter(lambda x: x.frames_remaining > 0, particles))

print('saving')

iio.imwrite(outfile, np.array(frames), fps=iio.immeta(infile, exclude_applied=False)['fps'])

print('done')
