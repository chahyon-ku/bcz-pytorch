import glob
import imageio
import os
import numpy as np


if __name__ == '__main__':
    gif_paths = glob.glob('/media/rpm/Data/CSCI5890-DEEPROB/bc-z/bcz-pytorch/viz/push_block/*.gif')
    gif_paths = gif_paths[:9]
    gifs = [imageio.mimread(gif_path) for gif_path in gif_paths]
    max_len = max([len(gif) for gif in gifs])
    combined = np.zeros((max_len + 5, 128 * 3, 128 * 3, 4)).astype(np.uint8)
    for i_gif, gif in enumerate(gifs):
        l, r, t, b = 128 * (i_gif % 3), 128 * (i_gif % 3 + 1), 128 * (i_gif // 3), 128 * (i_gif // 3 + 1)
        combined[:len(gif), t:b, l:r] = gif
        combined[len(gif):, t:b, l:r] = gif[-1]
    imageio.mimwrite('viz/push_block/push_block_seen_eval.gif', combined, fps=5)