import glob
import os
import numpy as np


if __name__ == '__main__':
    first_frames = glob.glob('data/pick_lemon/variation0/episodes/episode*/front_rgb/0.png')
    first_frames = sorted(first_frames, key=lambda x: int(x.split('/')[-3].replace('episode', '')))
    
    dst_dir = 'viz/pick_lemon_f1s/'
    os.makedirs(dst_dir, exist_ok=True)
    for i, first_frame in enumerate(first_frames):
        new_file_path = dst_dir + f'{i:03d}.png'
        os.system('cp {} {}'.format(first_frame, new_file_path))
