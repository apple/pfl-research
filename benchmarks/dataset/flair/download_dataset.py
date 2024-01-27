# Copyright © 2023-2024 Apple Inc.
import argparse
import functools
import logging
import multiprocessing
import os
import subprocess
import sys
from urllib.parse import urljoin
from urllib.request import urlretrieve

logger = logging.getLogger(name=__name__)

DATA_URL = "https://docs-assets.developer.apple.com/ml-research/datasets/flair/"
NUM_IMAGE_BATCHES = 43
SMALL_IMAGE_URLS = [
    urljoin(DATA_URL, f"images/small/small_images-{str(i).zfill(2)}.tar.gz")
    for i in range(NUM_IMAGE_BATCHES)
]
RAW_IMAGE_URLS = [
    urljoin(DATA_URL, f"images/raw/images-{str(i).zfill(2)}.tar.gz")
    for i in range(NUM_IMAGE_BATCHES)
]
LABELS_AND_METADATA_URL = urljoin(DATA_URL, "labels/labels_and_metadata.json")
LABEL_RELATIONSHIP_URL = urljoin(DATA_URL, "labels/label_relationship.txt")


def extract_tar(compressed_path: str, dataset_dir: str,
                keep_archive_after_decompress: bool):
    subprocess.run(f"tar -zxf {compressed_path} -C {dataset_dir}".split(),
                   check=True)
    if not keep_archive_after_decompress:
        os.remove(compressed_path)


def decompress_images(dataset_dir: str, keep_archive_after_decompress: bool):
    compressed_paths = [
        os.path.join(dataset_dir, path) for path in os.listdir(dataset_dir)
        if path.endswith(".tar.gz")
    ]
    decompress = functools.partial(
        extract_tar,
        dataset_dir=dataset_dir,
        keep_archive_after_decompress=keep_archive_after_decompress)
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        pool.map(decompress, compressed_paths)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout,
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(
        description='Download the images and labels of FLAIR dataset.')
    parser.add_argument("--dataset_dir",
                        required=True,
                        help="Path to directory of dataset to be downloaded")
    parser.add_argument("--download_raw",
                        action="store_true",
                        help="Whether to download the raw images, "
                        "which need storage space ~1.2TB")
    parser.add_argument("--keep_archive_after_decompress",
                        action="store_true",
                        help="Whether to keep the image tarball archives")
    arguments = parser.parse_args()
    os.makedirs(arguments.dataset_dir, exist_ok=True)

    # download labels and metadata
    logger.info("Downloading labels...")
    urlretrieve(
        LABELS_AND_METADATA_URL,
        os.path.join(arguments.dataset_dir,
                     os.path.basename(LABELS_AND_METADATA_URL)))
    urlretrieve(
        LABEL_RELATIONSHIP_URL,
        os.path.join(arguments.dataset_dir,
                     os.path.basename(LABEL_RELATIONSHIP_URL)))
    # download and decompress all images
    for image_url in SMALL_IMAGE_URLS:
        logger.info(f"Downloading small image: {image_url}")
        urlretrieve(
            image_url,
            os.path.join(arguments.dataset_dir, os.path.basename(image_url)))
    if arguments.download_raw:
        for image_url in RAW_IMAGE_URLS:
            logger.info(f"Downloading raw image: {image_url}")
            urlretrieve(
                image_url,
                os.path.join(arguments.dataset_dir,
                             os.path.basename(image_url)))
    logger.info("Decompressing images...")
    decompress_images(arguments.dataset_dir,
                      arguments.keep_archive_after_decompress)
