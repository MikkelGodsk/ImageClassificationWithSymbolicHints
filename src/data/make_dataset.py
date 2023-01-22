# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import hydra
from dotenv import find_dotenv, load_dotenv

import src.data.utils as utils


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
        # utils.download_dataset(data, cfg.data.cmplaces)

        logger.info("Unpacking CMPlaces")
        utils.unpack_dataset(data, cfg.data.cmplaces)

        logger.info("Preparing CMPlaces")
        utils.prepare_cmplaces(cfg)

        logger.info("Converting VGG16 from Caffe to PyTorch")
        utils.convert_vgg16(cfg)

    if cfg.data.download_imagenet:
        logger.info("Downloading ImageNet")
        utils.download_dataset(data, cfg.data.imagenet)

        logger.info("Unpacking ImageNet")
        utils.unpack_dataset(data, cfg.data.cmplaces)
        utils.unpack_imagenet(cfg)

        logger.info("Downloading Wikipedia articles")
        utils.get_wiki_descriptions(cfg)

        logger.info("Preparing ImageNet")
        utils.prepare_imagenet(cfg)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
