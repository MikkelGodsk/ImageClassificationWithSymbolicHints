import json
import os
import re
import shutil

import nltk
import numpy as np
import scipy.io
from nltk.corpus import wordnet as wn

ds_dir = "/work3/s184399/ImageNet"
unpacked_ds_dir = os.path.join(ds_dir, "UnpackedDataset")

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
print("Creating categories.txt")
meta = scipy.io.loadmat(
    os.path.join(ds_dir, "dev_kit", "ILSVRC2012_devkit_t12", "data", "meta.mat")
)
synsets = [meta["synsets"][i][0][1][0] for i in range(0, 1000)]


# Needs to unpack the validation set properly.
if not os.path.isdir(os.path.join(unpacked_ds_dir, "val")):
    os.mkdir(os.path.join(unpacked_ds_dir, "val"))
    print("Properly unpacking validation set")
    for synset in synsets:
        d = os.path.join(unpacked_ds_dir, "val", synset)
        if not os.path.isdir(d):
            os.mkdir(d)

    val_ground_truths = []
    with open(
        os.path.join(
            ds_dir,
            "dev_kit",
            "ILSVRC2012_devkit_t12",
            "data",
            "ILSVRC2012_validation_ground_truth.txt",
        ),
        "r",
    ) as f_obj:
        for target in f_obj:
            val_ground_truths.append(int(target))

    for i, img in enumerate(sorted(os.listdir(os.path.join(ds_dir, "val_imgs")))):
        source = os.path.join(ds_dir, "val_imgs", img)
        target = os.path.join(
            unpacked_ds_dir, "val", synsets[val_ground_truths[i] - 1], img
        )
        if not os.path.isfile(target):
            shutil.copy(source, target)  # .move(


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


with open("wiki_descriptions.json", "r") as f_obj:
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
            print("removed: {:s}".format(k))


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
if True:
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
            print(k)
            print(t.intersection(v))
        assert len(t)
        assert len(v)
