# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import hydra
from dotenv import find_dotenv, load_dotenv

from src.data.utils import *


@hydra.main(version_base=None, config_path="../../conf", config_name="data_conf.yaml")
def main(cfg):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)

    logger.info("Creating folders")

    data = cfg.data.data_folder  # Add CMPlaces or ImageNet
    if not os.path.isdir(data):
        os.mkdir(data)

    if cfg.data.download_cmplaces:
        logger.info("Downloading CMPlaces")
        download_dataset(data, cfg.data.cmplaces)

        logger.info("Unpacking CMPlaces")
        unpack_dataset(data, cfg.data.cmplaces)

        logger.info("Preparing CMPlaces")

    if cfg.data.download_imagenet:
        logger.info("Downloading ImageNet")
        download_dataset(data, cfg.data.imagenet)

        logger.info("Unpacking ImageNet")
        unpack_dataset(data, cfg.data.cmplaces)

        logger.info("Preparing ImageNet")
        logger.info("Downloading Wikipedia articles")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
