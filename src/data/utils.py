import logging
import os
import tarfile
from urllib.error import HTTPError
from urllib.request import urlretrieve
from zipfile import ZipFile


def download_dataset(data, ds_group):
    logger = logging.getLogger(__name__)
    data_folder = os.path.join(data, ds_group.data_folder)
    if not os.path.isdir(data_folder):
        os.mkdir(data_folder)
    files = [
        (ds_group.file_names[k], ds_group.urls[k]) for k in ds_group.file_names.keys()
    ]
    for f_name, url in files:
        full_path = os.path.join(data_folder, f_name)
        if not os.path.isfile(full_path):
            logger.info(f"\tDownloading {f_name}")
            try:
                urlretrieve(url, full_path)
            except HTTPError:
                logger.warning(f"\t\tFailed to download {f_name}")


def unpack_dataset(data, ds_group):
    logger = logging.getLogger(__name__)
    data_folder = os.path.join(data, ds_group.data_folder)
    # unpacked_datafolder = os.path.join(data_folder, ds_group.unpacked_folder)  # Not to be used yet
    if not os.path.isdir(data_folder):
        os.mkdir(data_folder)
    for f_name in ds_group.file_names.values():
        full_path_src = os.path.join(data_folder, f_name)
        if f_name.endswith(".tar.gz"):
            try:
                logger.info(f"\tUnpacking {f_name}")
                full_path_dst = os.path.join(data_folder, f_name[:-7])
                tar = tarfile.open(full_path_src, "r:gz")
                tar.extractall(path=full_path_dst)
                tar.close()
            except FileNotFoundError:
                logger.warning(f"\t\tFile not found: {f_name}")
        elif f_name.endswith(".zip"):
            try:
                logger.info(f"\tUnpacking {f_name}")
                full_path_dst = os.path.join(data_folder, f_name[:-4])
                with ZipFile(full_path_src, "r") as z_object:
                    z_object.extractall(path=full_path_dst)
            except FileNotFoundError:
                logger.warning(f"\t\tFile not found: {f_name}")
