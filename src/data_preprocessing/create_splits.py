import argparse
import glob
import os
import random
import shutil

from utils import get_module_logger


def move_files(file_lst, idx_lst, num_train, source, destination):
    # Move train
    for idx in idx_lst[:num_train]:
        _, name = os.path.split(file_lst[idx]) 
        src = os.path.join(source, name)
        dest = os.path.join(destination, "train", name)
        shutil.move(src, dest)

    # Move val
    for idx in idx_lst[num_train:]:
        _, name = os.path.split(file_lst[idx]) 
        src = os.path.join(source, name)
        dest = os.path.join(destination, "val", name)
        shutil.move(src, dest)

def split(source, destination):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
    """

    ratio = 0.8

    # Create subfolders
    for i in ["train", "val"]:
        subfolder = os.path.join(destination, i)
        if not os.path.exists(subfolder):
            os.mkdir(subfolder)

    # Get file to a list and get files with cyclists from the exploratory data analysis step
    file_lst = sorted(glob.glob(source + "/*.tfrecord"))

    cyclist_lst = [1, 15, 16, 21, 23, 24, 31, 34, 35, 37,
                   39, 42, 49, 52, 56, 61, 63, 65, 68, 72,
                   78, 79, 86, 89, 90, 94, 95]

    leftover_lst = list(set(range(len(file_lst))) - set(cyclist_lst))

    cyclist_train = int(len(cyclist_lst) * ratio)
    leftover_train = int(len(leftover_lst) * ratio)

    random.shuffle(cyclist_lst)
    random.shuffle(leftover_lst)
    
    move_files(file_lst, cyclist_lst, cyclist_train, source, destination)
    move_files(file_lst, leftover_lst, leftover_train, source, destination)

    logger.info("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--source', required=True,
                        help='source data directory')
    parser.add_argument('--destination', required=True,
                        help='destination data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.source, args.destination)