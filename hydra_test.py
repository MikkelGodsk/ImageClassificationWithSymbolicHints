import hydra


@hydra.main(version_base=None, config_path="conf", config_name="data_conf.yaml")
def main(cfg):
    print(cfg)
    print(cfg.data.cmplaces.file_names.descriptions)

    import os

    data = cfg.data.data_folder
    cmplaces_data = os.path.join(data, "CMPlaces")
    cmplaces_data_unpacked = os.path.join(cmplaces_data, "UnpackedDataset")
    imagenet_data = os.path.join(data, "ImageNet")
    imagenet_data_unpacked = os.path.join(imagenet_data, "UnpackedDataset")
    files = [
        (cfg.data.cmplaces.file_names[k], cfg.data.cmplaces.urls[k])
        for k in cfg.data.cmplaces.file_names.keys()
    ]
    for f_name, url in files:
        print(url, os.path.join(cmplaces_data_unpacked, f_name))


if __name__ == "__main__":
    main()
