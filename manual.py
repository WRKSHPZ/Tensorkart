#!/usr/bin/env python
import sys

from utils import resize_image, XboxController
from termcolor import cprint

import gym
import gym_mupen64plus
from train import create_model
import numpy as np

from skimage.io import imread

# Play
class Actor(object):

    def __init__(self):
        # Load in model from train.py and load in the trained weights
        self.model = create_model(keep_prob=1) # no dropout
        self.model.load_weights('model_weights.h5')

  
    def do_action(self, obs, fname):

        ### determine manual override

        vec = resize_image(obs)
        
        vec = np.expand_dims(vec, axis=0) # expand dimensions for predict, it wants (1,66,200,3) not (66, 200, 3)
        ## Think
        joystick = self.model.predict(vec, batch_size=1)[0]

        ## Act

        ### calibration
            
        if (len(joystick) > 1):
            output = [
                int(joystick[0] * 80),
                int(joystick[1] * 80),
                int(round(joystick[2])),
                int(round(joystick[3])),
                int(round(joystick[4])),
            ]
        else:
            output = [
                int(joystick[0] * 80)
            ]

        cprint("Joystick: " + str(joystick), 'green')


        if (abs(joystick[0]) > 0.1):
            cprint(fname, 'red')
            cprint("Joystick: " + str(joystick), 'green')
            cprint("Manual: " + str(output), 'yellow')


if __name__ == '__main__':
    actor = Actor()
    print('actor ready!')

    if (sys.argv[1].endswith('.png')):
        im = imread(sys.argv[1])
        actor.do_action(im, sys.argv[1])
    else:
        image_files = np.loadtxt(sys.argv[1] + '/data.csv', delimiter=',', dtype=str, usecols=(0,))

        for i in range(len(image_files)):
            im = imread(image_files[i])
            actor.do_action(im, image_files[i])
