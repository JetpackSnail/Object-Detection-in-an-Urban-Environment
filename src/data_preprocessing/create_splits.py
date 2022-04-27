import argparse
import glob
import os
import random
import shutil

from utils import get_module_logger


def split(source, destination):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
    """
    # Define train, val, test split ratios (total sum to 1)
    ratio = [0.8, 0.1, 0.1]

    # Create subfolders
    for i in ["train", "val", "test"]:
        subfolder = os.path.join(destination, i)
        if not os.path.exists(subfolder):
            os.mkdir(subfolder)

    # Get file to a list and randomise the order
    file_lst = glob.glob(source + "/*.tfrecord")
    random.shuffle(file_lst)

    # Calculate number of files according to ratio
    num_train = int(len(file_lst) * ratio[0])
    num_val = int(len(file_lst) * ratio[1])

    # Move train
    for file in file_lst[:num_train]:
        shutil.move(file, os.path.join(destination, "train")) 

    # Move val
    for file in file_lst[num_train:num_val+num_train]:
        shutil.move(file, os.path.join(destination, "val")) 
    
    # Move test
    for file in file_lst[num_val+num_train:]:
        shutil.move(file, os.path.join(destination, "test"))

    logger.info(f"Number of training images: {num_train}")
    logger.info(f"Number of validation images: {num_val}")
    logger.info(f"Number of testing images: {len(file_lst) - num_train - num_val}")


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