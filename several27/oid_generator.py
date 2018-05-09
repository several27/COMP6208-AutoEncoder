import os
from multiprocessing.pool import Pool

import numpy as np
from tqdm import tqdm

from several27.SRGAN_OID import open_resized_image, count_images

path_source = 'several27/data/oid_test_256_192/'
path_destination = 'several27/data/oid_test_256_192_1/'

size = 256, 192


def process_image(file):
    if not file.endswith('jpg'):
        return

    img = open_resized_image(path_source + file, size)
    img.save(path_destination + file)


def main():
    with tqdm(total=count_images(path_source)) as progress, Pool(processes=4) as pool:
        for file in pool.imap_unordered(process_image, os.listdir(path_source), chunksize=10):
            progress.update()


if __name__ == '__main__':
    main()
