import glob
import os

if __name__ == '__main__':
    file_paths = glob.glob('/media/rpm/Data/CSCI5890-DEEPROB/bc-z/bcz-pytorch/viz/reach_target/variation0/episodes/episode0/front_rgb/*.png')
    file_paths = sorted(file_paths, key=lambda x: int(x.split('/')[-1].split('.')[0]))
    for i, file_path in enumerate(file_paths):
        new_file_path = file_path.replace('front_rgb', 'front_rgb_renamed')
        new_file_path = '/'.join(new_file_path.split('/')[:-1]) + f'/{i:03d}.png'
        os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
        print('Copying {} to {}'.format(file_path, new_file_path))
        os.system('cp {} {}'.format(file_path, new_file_path))