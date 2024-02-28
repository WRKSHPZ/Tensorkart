import sys, os

from termcolor import cprint

import numpy as np

from PIL import Image, ImageOps

import pandas as pd 

if __name__ == '__main__':
    folder = sys.argv[1]
    new_folder = folder + "_flipped"

    try:
        os.mkdir(new_folder)
    except FileExistsError:
        cprint("Folder already exists", 'red')

    image_files = np.loadtxt(folder + '/data.csv', delimiter=',', dtype=str, usecols=(0,))

    for i in range(len(image_files)):
        im =  Image.open(image_files[i])
        hori_flippedImage = ImageOps.mirror(im)

        short_folder = image_files[i].replace(image_files[i].split('/')[-1], '').strip('/')
        flipped_folder = short_folder + '_flipped'

        new_name = image_files[i].replace(short_folder, flipped_folder)
        print(new_name)
        hori_flippedImage.save(new_name)

    df = pd.read_csv(folder + '/data.csv', header=None) 
    
    for i in df.index:
        df.at[i,0]=df.loc[i][0].replace('/img', '_flipped/img')
        df.at[i,1]=df.loc[i][1]*-1

    df.to_csv(new_folder + '/data.csv', index=False, header=None)


