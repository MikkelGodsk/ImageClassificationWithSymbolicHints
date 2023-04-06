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
    if not os.path.isdir(data_folder):
        os.mkdir(data_folder)
    for f_name in ds_group.file_names.values():
        full_path_src = os.path.join(data_folder, f_name)
        assert os.path.isfile(full_path_src)
        if f_name.endswith(".tar.gz") or f_name.endswith(".tar"):
            try:
                logger.info(f"\tUnpacking {f_name}")
                full_path_dst = os.path.join(
                    data_folder,
                    f_name[:-7] if f_name.endswith(".tar.gz") else f_name[:-4],
                )
                tar = tarfile.open(
                    full_path_src, "r:gz" if f_name.endswith(".tar.gz") else "r:"
                )
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


def prepare_cmplaces(cfg):
    import csv
    import json
    import os
    import shutil

    logger = logging.getLogger(__name__)

    # Source files:
    # dataset_directory_src = "/work3/s184399/CMPlaces"
    dataset_directory_src = os.path.join(
        cfg.data.data_folder, cfg.data.cmplaces.data_folder
    )
    description_folder_src = os.path.join(dataset_directory_src, "descriptions/text")
    image_folder_src = os.path.join(
        dataset_directory_src,
        cfg.data.cmplaces.file_names.imagesPlaces205_resize[:-7],
        "data/vision/torralba/deeplearning/images256",
    )

    image_train_split = os.path.join(
        dataset_directory_src,
        "trainvalsplit_places205/trainvalsplit/trainvalsplit_places205/train_places205.csv",
    )
    image_val_split = os.path.join(
        dataset_directory_src,
        "trainvalsplit_places205/trainvalsplit/trainvalsplit_places205/val_places205.csv",
    )
    text_train_split = os.path.join(
        dataset_directory_src, "cmplaces/labels/text_train.txt"
    )
    text_val_split = os.path.join(dataset_directory_src, "cmplaces/labels/text_val.txt")

    # Destination files:
    # dataset_directory_dest = "/work3/s184399/CMPlaces/UnpackedDataset"
    dataset_directory_dest = os.path.join(
        dataset_directory_src, cfg.data.cmplaces.unpacked_folder
    )
    if not os.path.isdir(dataset_directory_dest):
        os.mkdir(dataset_directory_dest)

    train_dest = os.path.join(dataset_directory_dest, "train")
    val_dest = os.path.join(dataset_directory_dest, "val")

    # Copy categories.txt
    shutil.copy(
        os.path.join(
            dataset_directory_src,
            cfg.data.cmplaces.file_names.cmplaces[:-4],
            "labels",
            "categories.txt",
        ),
        os.path.join(dataset_directory_dest, "categories.txt"),
    )

    def get_files_from_split(src):
        split_files = []
        with open(src, "r") as f:
            rows = csv.reader(f, delimiter=" ")
            for row in rows:
                split_files.append(row[0])
        return split_files

    def collect_images(src, dest, split):
        """
        Collects all images from a dataset split (specified by split_df) into a structure that can be read by
        tf.keras.utils.image_dataset_from_directory(dest, follow_links=True).

        Args:
            src : A string specifying the source folder
            dest : A string specifying the destination folder
            split_df : A list with filenames in the split
        """
        logger = logging.getLogger(__name__)
        if os.path.isdir(dest):
            # If directory already exists, abort
            logger.info("Path {:s} exists already".format(dest))
            return False

        # Create new folder
        os.mkdir(dest)

        # Loop through each subfolder and move the files
        depth_baseline = len(src.split("/"))
        for root, folders, files in os.walk(src):
            depth = len(root.split("/"))
            class_name = "_".join(
                root.split("/")[(depth_baseline + 1) :]
            )  # Add one to descend into the folder with just 1 letter, e.g. 'a'
            if len(files) > 0:
                os.mkdir(os.path.join(dest, class_name))
                logger.info(os.path.join(dest, class_name))
            for file in files:
                file_abs_path = os.path.join(root, file)
                if os.path.relpath(file_abs_path, start=src) in split:
                    # os.symlink(
                    shutil.move(
                        src=file_abs_path, dst=os.path.join(dest, class_name, file)
                    )

        return True

    def test_images(collected_images, split):
        logger = logging.getLogger(__name__)
        # Are the collected files a subset of the split?
        split = list(map(lambda s: s.split("/")[-1], split))

        for root, folders, files in os.walk(collected_images):
            for file in files:
                if not file in split:
                    logger.info(
                        "File {:s} does not belong!".format(os.path.join(root, file))
                    )
                assert file in split

        # Is the split a subset of the collected files (and is it uniquely represented)?
        for split_file in split:
            existence_counts = 0
            for root, folders, files in os.walk(collected_images):
                if split_file in files:
                    existence_counts += 1
            assert existence_counts == 1

    def collect_text(src, dest, split):
        """
        Collects text into a json file
        """
        logger = logging.getLogger(__name__)
        if os.path.isfile(dest):
            logger.info("Path {:s} already exists".format(dest))
            return False

        # Create a json file where for each category, we have a key-value pair of category x list of descriptions.
        categories = os.listdir(src)
        desc_dict = {}

        for cat in categories:
            desc_dict[cat] = []
            for file in os.listdir(
                os.path.join(src, cat)
            ):  # The file names, not rel. or abs. paths
                file_rel_path = os.path.join("data", "text", cat, file)
                if file_rel_path in split:
                    fname = os.path.join(src, cat, file)
                    with open(fname, "r") as f_obj:
                        desc_dict[cat].append(f_obj.read())
        with open(dest, "w") as f_obj:
            json.dump(desc_dict, f_obj)

        return True

    # Prepare dataset
    # collect_text()
    # collect_images()
    # check_text_image_consistency()

    # Try loading the dataset
    # ds = tf.keras.utils.image_dataset_from_directory('CMPlaces/CMPimages', follow_links=True, batch_size=1)

    # Prepare images
    train_split = get_files_from_split(image_train_split)
    collect_images(src=image_folder_src, dest=train_dest, split=train_split)
    val_split = get_files_from_split(image_val_split)
    collect_images(src=image_folder_src, dest=val_dest, split=val_split)
    # test_images(collected_images = val_dest, split=val_split)

    # Prepare text
    train_split = get_files_from_split(text_train_split)
    collect_text(
        src=description_folder_src,
        dest=os.path.join(dataset_directory_dest, "train_text.json"),
        split=train_split,
    )
    val_split = get_files_from_split(text_val_split)
    collect_text(
        src=description_folder_src,
        dest=os.path.join(dataset_directory_dest, "val_text.json"),
        split=val_split,
    )


def convert_vgg16(cfg):
    import os

    logger = logging.getLogger(__name__)

    dataset_directory_src = os.path.join(
        cfg.data.data_folder, cfg.data.cmplaces.data_folder
    )

    ir_folder = os.path.join(dataset_directory_src, "places205VGG16-IR")
    if not os.path.isdir(ir_folder):
        os.mkdir(ir_folder)  # mkdir [folder]/places205VGG16-IR
    torch_folder = os.path.join(dataset_directory_src, "places205VGG16-torch")
    if not os.path.isdir(torch_folder):
        os.mkdir(torch_folder)  # mkdir [folder]/places205VGG16-torch

    # make folder
    prototxt = os.path.join(
        dataset_directory_src, "places205vgg", "places205VGG16", "deploy_10.prototxt"
    )
    caffemodel = os.path.join(
        dataset_directory_src,
        "places205vgg",
        "places205VGG16",
        "snapshot_iter_765280.caffemodel",
    )
    os.system(
        f"mmtoir -f caffe -n {prototxt} -w {caffemodel} -o {os.path.join(ir_folder, 'caffe_vgg16_IR')}"
    )
    # mmtoir -f caffe -n [folder]/places205VGG16/deploy_10.prototxt -w [folder]/places205VGG16/snapshot_iter_765280.caffemodel -o [folder]/places205VGG16-IR/caffe_vgg16_IR

    pb_file = os.path.join(ir_folder, "caffe_vgg16_IR.pb")
    caffe_npy_file = os.path.join(ir_folder, "caffe_vgg16_IR.npy")
    py_file = os.path.join(torch_folder, "pytorch_vgg16.py")
    torch_npy_file = os.path.join(torch_folder, "pytorch_vgg16.npy")
    os.system(
        f"mmtocode -f pytorch -n {pb_file} --IRWeightPath {caffe_npy_file} --dstModelPath {py_file} -dw {torch_npy_file}"
    )
    # mmtocode -f pytorch -n [folder]/places205VGG16-IR/caffe_vgg16_IR.pb --IRWeightPath [folder]/places205VGG16-IR/caffe_vgg16_IR.npy --dstModelPath [folder]/places205VGG16-torch/pytorch_vgg16.py -dw [folder]/places205VGG16-torch/pytorch_vgg16.npy

    os.remove(py_file)
    # remove [folder]/places205VGG16-torch/pytorch_vgg16.py


def get_wiki_descriptions(cfg):
    import json
    import os
    import sys

    import wikipediaapi as wiki
    from tqdm import tqdm
    from wikidata.client import Client

    logger = logging.getLogger(__name__)

    directory = os.path.join(
        "src", "third_party_files"
    )  # Insert directory where mapping.json is located
    fname = os.path.join(directory, "mapping.json")

    with open(fname, "r") as f_obj:
        mappings = json.load(f_obj)

    for k, v in mappings.items():
        mappings[k] = v.split("/")[-1]

    client = Client()
    wiki_en = wiki.Wikipedia("en")

    articles = {}
    failures = []

    # Cache in case we would like to modify the policy and not have to make new queries.
    cache_wikidata = {}
    cache_wikipedia = {}

    use_cache = True

    i = 0
    for k, v in tqdm(mappings.items()):
        i += 1
        entity = None
        if v not in cache_wikidata.keys() or not use_cache:
            entity = client.get(v, load=True)
            cache_wikidata[v] = entity
        else:
            entity = cache_wikidata[v]
        title = None

        # First look up Wikipedia page in Wikidata. Else search Wikipedia for its title. Else count as a failure
        if "enwiki" in entity.data["sitelinks"].keys():
            title = entity.data["sitelinks"]["enwiki"]["title"]
        elif "en" in entity.data["labels"]:
            title = entity.data["labels"]["en"]["value"]
        else:
            failures.append(k)
            continue

        if title not in cache_wikipedia.keys() or not use_cache:
            article = wiki_en.page(title)
            cache_wikipedia[title] = article
        else:
            article = cache_wikipedia[title]
        if not article.exists():
            failures.append(k)
        else:
            articles[k] = article

    logger.info("\n")
    logger.info("Number of failures: {}".format(len(failures)))
    logger.info("Failures:")
    logger.info(failures)

    import nltk
    from nltk.corpus import wordnet as wn

    nltk.download("wordnet")

    mappings_reversed = {v: k for k, v in mappings.items()}

    get_synset_from_id = lambda synset_id: wn.synset_from_pos_and_offset(
        synset_id[0], int(synset_id[1:])
    )
    get_title_from_synset = lambda synset: synset.name().split(".")[0].replace("_", " ")

    logger.info("Failures:")
    for failure in failures:
        logger.info(
            "WordNet id: {} - Title: {}".format(
                failure, get_title_from_synset(get_synset_from_id(failure))
            )
        )

    logger.info("\n\nDifference in titles:")
    for wnid, article in articles.items():
        wordnet_title = get_title_from_synset(get_synset_from_id(wnid)).lower()
        article_title = article.title.lower()
        if (
            wordnet_title not in article_title
            and wordnet_title not in article.summary.lower()
        ):
            logger.info(
                "WordNet id: {} - WordNet title: {} - Wikipedia title: {}".format(
                    wnid, wordnet_title, article_title
                )
            )

    summaries = {}
    length = len(articles)
    i = 0
    for k, v in tqdm(articles.items()):
        i += 1
        summaries[k] = v.summary  # .replace('\n', ' ')

    with open(
        os.path.join(
            cfg.data.data_folder,
            cfg.data.imagenet.data_folder,
            "wiki_descriptions.json",
        ),
        "w",
    ) as f_obj:
        json.dump(summaries, f_obj)


def unpack_imagenet_train_set(cfg):
    import os
    import tarfile

    logger = logging.getLogger(__name__)
    packed_dir = os.path.join(
        cfg.data.data_folder,
        cfg.data.imagenet.data_folder,
        cfg.data.imagenet.file_names.ILSVRC2012_img_train[:-4],
    )
    unpacked_dir = os.path.join(
        cfg.data.data_folder,
        cfg.data.imagenet.data_folder,
        cfg.data.imagenet.unpacked_folder,
        "train",
    )
    if not os.path.isdir(
        os.path.join(
            cfg.data.data_folder,
            cfg.data.imagenet.data_folder,
            cfg.data.imagenet.unpacked_folder,
        )
    ):
        os.mkdir(
            os.path.join(
                cfg.data.data_folder,
                cfg.data.imagenet.data_folder,
                cfg.data.imagenet.unpacked_folder,
            )
        )
    if not os.path.isdir(unpacked_dir):
        os.mkdir(unpacked_dir)
    assert os.path.isdir(
        packed_dir
    ), f"{packed_dir} is not a directory. Either the dataset has not been unpacked or the wrong path is given"
    synsets = map(lambda x: x[:-4], os.listdir(packed_dir))
    for synset in synsets:
        full_path_src = os.path.join(packed_dir, synset) + ".tar"
        full_path_dst = os.path.join(unpacked_dir, synset)
        os.mkdir(full_path_dst)
        tar = tarfile.open(full_path_src, "r:")
        tar.extractall(path=full_path_dst)
        tar.close()


def prepare_imagenet(cfg):
    import json
    import os
    import re
    import shutil

    import nltk
    import numpy as np
    import scipy.io
    from nltk.corpus import wordnet as wn

    logger = logging.getLogger(__name__)

    # ds_dir = "/work3/s184399/ImageNet"
    # unpacked_ds_dir = os.path.join(ds_dir, "UnpackedDataset")
    ds_dir = os.path.join(cfg.data.data_folder, cfg.data.imagenet.data_folder)
    unpacked_ds_dir = os.path.join(ds_dir, cfg.data.imagenet.unpacked_folder)

    train_val_split = 0.8

    """
    This file does the following in order to prepare the dataset:
    - Properly unpacks the validation set
    - Makes a datafile of mixed Wikipedia and WordNet text (the text modality)
    - !!Important!!: Deletes the synsets from the image modality which do not have a corresponding text bit from Wikipedia.
    - Creates a file "categories.txt" which is needed for compatibility. Stores the synsets and the class-id integer.
    - Does a training and validation-split of the text modality.
    """

    # Make list of synsets
    logger.info("Creating categories.txt")
    meta = scipy.io.loadmat(
        os.path.join(
            ds_dir,
            cfg.data.imagenet.file_names.ILSVRC2012_devkit_t12[:-7],
            "ILSVRC2012_devkit_t12",
            "data",
            "meta.mat",
        )
    )
    synsets = [meta["synsets"][i][0][1][0] for i in range(0, 1000)]

    # Needs to unpack the validation set properly.
    if not os.path.isdir(os.path.join(unpacked_ds_dir, "val")):
        os.mkdir(os.path.join(unpacked_ds_dir, "val"))
        logger.info("Properly unpacking validation set")
        for synset in synsets:
            d = os.path.join(unpacked_ds_dir, "val", synset)
            if not os.path.isdir(d):
                os.mkdir(d)

        val_ground_truths = []
        with open(
            os.path.join(
                ds_dir,
                cfg.data.imagenet.file_names.ILSVRC2012_devkit_t12[:-7],
                "ILSVRC2012_devkit_t12",
                "data",
                "ILSVRC2012_validation_ground_truth.txt",
            ),
            "r",
        ) as f_obj:
            for target in f_obj:
                val_ground_truths.append(int(target))

        for i, img in enumerate(
            sorted(
                os.listdir(
                    os.path.join(
                        ds_dir, cfg.data.imagenet.file_names.ILSVRC2012_img_val[:-4]
                    )
                )
            )
        ):
            source = os.path.join(
                ds_dir, cfg.data.imagenet.file_names.ILSVRC2012_img_val[:-4], img
            )
            target = os.path.join(
                unpacked_ds_dir, "val", synsets[val_ground_truths[i] - 1], img
            )
            if not os.path.isfile(target):
                shutil.copy(source, target)  # .move(
                # os.symlink(source, target)

    # Make datafiles of merged Wikipedia-hints and WordNet-hints
    ## Wikipedia
    def prepare_sentences(desc):
        # s = desc.split('. ')
        s = re.findall("[A-Z0-9][a-z0-9]* [a-zA-Z0-9\-_():, \"']*.[ |\n]", desc)
        sentences = []
        for i, sentence in enumerate(s):
            if sentence != "":
                sentences.append(sentence.replace("\n", "").lstrip())
        return sentences

    with open(
        os.path.join(
            cfg.data.data_folder,
            cfg.data.imagenet.data_folder,
            "wiki_descriptions.json",
        ),
        "r",
    ) as f_obj:
        wiki_descriptions = json.load(f_obj)

    for k, v in wiki_descriptions.items():
        wiki_descriptions[k] = prepare_sentences(v)

    ## WordNet
    nltk.download("wordnet")
    splitting_chars = "_| |-|'"  # | as separator

    def find_synset(wnid):
        pos, offset = wnid[0], int(wnid[1:])
        return wn.synset_from_pos_and_offset(pos, offset)

    wnet_descriptions = {}
    for i, label in enumerate(synsets):
        desc = find_synset(label).definition()
        wnet_descriptions[label] = desc

    ## Combined descriptions
    combined_descriptions = {}
    for k, v in wiki_descriptions.items():
        combined_descriptions[k] = v.copy()
        combined_descriptions[k].append(
            wnet_descriptions[k]
        )  # Only add the ones that exist in both

    # Delete images without an associated Wikipedia-description.
    no_wiki_text = set(wnet_descriptions.keys()).difference(
        set(combined_descriptions.keys())
    )
    for k in no_wiki_text:
        for p in ["train", "val"]:
            directory = os.path.join(unpacked_ds_dir, p, k)
            if os.path.isdir(directory):
                shutil.rmtree(directory)
                logger.info("removed: {:s}".format(k))

    # Create a file [imagenet_folder]/UnpackedDataset/categories.txt
    with open(os.path.join(unpacked_ds_dir, "categories.txt"), "w") as f_obj:
        for i, synset in enumerate(sorted(synsets)):
            if synset not in no_wiki_text:
                f_obj.write(synset + " " + str(i) + "\n")

    # Needs to do a train/val-split.
    train_desc = {}
    val_desc = {}
    for k, v in combined_descriptions.items():
        train_ixs = np.random.choice(
            len(v), int(np.ceil(train_val_split * len(v))), replace=False
        ).tolist()
        val_ixs = [i for i in range(0, len(v)) if i not in train_ixs]
        if not len(val_ixs):
            val_ixs.append(train_ixs.pop())
        train_desc[k] = np.array(v)[train_ixs].tolist()
        val_desc[k] = np.array(v)[val_ixs].tolist()

    with open(os.path.join(unpacked_ds_dir, "train_text.json"), "w") as f_obj:
        json.dump(train_desc, f_obj)

    with open(os.path.join(unpacked_ds_dir, "val_text.json"), "w") as f_obj:
        json.dump(val_desc, f_obj)

    # Tests:
    # All keys in both:
    t = set(train_desc.keys())
    v = set(val_desc.keys())
    assert t == v

    # Hints not in both, and non-empty hints:
    for k in train_desc.keys():
        t = set(train_desc[k])
        v = set(val_desc[k])
        # assert not len(t.intersection(v))
        if len(t.intersection(v)):
            logger.info(k)
            logger.info(t.intersection(v))
        assert len(t)
        assert len(v)
