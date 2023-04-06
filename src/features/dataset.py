import json
import os
from typing import Callable, Dict, List, Tuple, Union

import torch
import torchvision

ds_dir = None
dataset = None
num_classes = None

labels_translation_dict = None


class DataHandler:
    """
        Used instead of saving the dataset info in global variables. Contains all the dataset-specific info.
    """
    def __init__(self, ds_name: str, ds_dir: str='/work3/s184399'):
        assert os.path.isdir(ds_dir)
        self.set_dataset(ds_name, ds_dir)

    def set_dataset(self, ds_name: str, ds_dir: str='/work3/s184399'):
        """
            Pick the dataset to work on
        """
        if ds_name.lower() == "imagenet":
            ds_name = 'ImageNet'
            self.num_classes = 1000
        elif ds_name.lower() == "cmplaces":
            ds_name = 'CMPlaces'
            self.num_classes = 205
        else:
            raise ValueError()
        self.ds_dir = os.path.join(ds_dir, ds_name, 'UnpackedDataset')
        self.dataset = ds_name.lower()


    ###########
    # Helpers #
    ###########
    def correct_labeling(self, x):
        """
        Since there may be a discrepancy between the label ordering of the model, and the ImageFolder-order, we need a function to translate. Expects a file "categories.txt" with label-names and label-numbers, e.g.:
        n02077923 13
        n02110063 14
        n02447366 15
        n02109047 16
        """
        global labels_translation_dict
        if labels_translation_dict is None:
            # Build if not already built
            dir_labels = sorted(
                os.listdir(os.path.join(self.ds_dir, "val"))
            )  # Has been tested to be equal to ordering of ds.class_to_idx.keys()
            dir_labels = dict(zip(dir_labels, range(0, len(dir_labels))))
            labels_translation_dict = {}
            with open(os.path.join(self.ds_dir, "categories.txt"), "r") as f_obj:
                for lines in f_obj.readlines():
                    k, v = lines.split()
                    labels_translation_dict[dir_labels[k]] = int(v)

        return labels_translation_dict[x]


    def load_text_dict(self, filename):
        with open(filename, "r") as f_obj:
            txt_dict = json.load(f_obj)
        return txt_dict


    def text_dict_to_ds_list(self, text_dict):
        classes = sorted(text_dict.keys())
        X = []
        y = []

        for i, c in enumerate(classes):
            corrected_i = self.correct_labeling(i)  # Transform to correct indexing.
            for x_sample in text_dict[c]:
                X.append(x_sample)
                y.append(corrected_i)

        return X, y


    def build_class_to_idx(self, text_dict):
        classes = sorted(text_dict.keys())
        class_to_idx = {}
        for i, c in enumerate(classes):
            class_to_idx[c] = self.correct_labeling(i)

        return class_to_idx


    def build_dataset_dict(self, text_dict, class_to_idx):
        ds_dict = {}
        for c, idx in class_to_idx.items():
            ds_dict[idx] = text_dict[c]
        return ds_dict


    #####################
    ## Dataset loaders ##
    #####################
    def load_img_ds(self, filename):
        if (
            self.dataset == "imagenet"
        ):  # Transform from example https://pytorch.org/vision/stable/models.html
            from torchvision.models import ResNet50_Weights

            transform = ResNet50_Weights.IMAGENET1K_V2.transforms()
            """return torchvision.datasets.ImageNet(
                root='/work3/s184399/imagenet/downloads/manual', 
                split=filename, 
                transform=transform
            )"""
        elif (
            self.dataset == "cmplaces"
        ):  # Transform from https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py
            mean = torch.Tensor([103.939, 116.779, 123.68])
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),  # ToTensor rescales pixels from range 0-255 to range 0-1...
                    torchvision.transforms.Resize(size=(224, 224)),
                    lambda x: x[[2, 1, 0], :, :],  # RGB to BGR.
                    lambda x: x
                    * 255,  # ToTensor scales from 0-255 to 0-1. Here I scale back.
                    torchvision.transforms.Normalize(
                        mean, torch.Tensor([1.0, 1.0, 1.0]), inplace=False
                    ),
                ]
            )
        filename = os.path.join(self.ds_dir, filename)
        return torchvision.datasets.ImageFolder(
            filename, transform=transform, target_transform=self.correct_labeling
        )


    def load_text_ds(self, filename):
        filename = os.path.join(self.ds_dir, filename)
        return self.TextModalityDS(
            self,
            root=filename,
        )


    #####################
    ## Dataset classes ##
    #####################
    ImageModalityDS = torchvision.datasets.DatasetFolder


    class TextModalityDS(torch.utils.data.Dataset):
        """
        Loads the text modality of the dataset. Has properties that generates the dataset.

        Is inside that DataHandler class, since how we process the text depends on the dataset.
        """

        def __init__(
            self,
            outer: "DataHandler",
            root: str = None,
            ds_dict: Dict[int, List[str]] = None,
            transform=None,
            target_transform=None,
        ):
            self.outer: DataHandler = outer
            self._data_root = root
            if root is not None:
                text_dict = self.outer.load_text_dict(root)
                self._X, self._y = self.outer.text_dict_to_ds_list(text_dict)
                self._class_to_idx = self.outer.build_class_to_idx(text_dict)
                self._dataset_dict = self.outer.build_dataset_dict(text_dict, self._class_to_idx)
            elif ds_dict is not None:
                self._dataset_dict = ds_dict
                self._X, self._y = self._ds_lists_from_ds_dict(ds_dict)
                self._class_to_idx = None

            self.transform = transform
            self.target_transform = target_transform

        def _ds_lists_from_ds_dict(self, ds_dict):
            X_list, y_list = [], []
            for y, Xs in ds_dict.items():
                for X in Xs:
                    X_list.append(X)
                    y_list.append(y)
            return X_list, y_list

        def __len__(self):
            return len(self._y)

        def __getitem__(self, idx):
            x, y = self._X[idx], self._y[idx]
            if self.transform:
                x = self.transform(x)
            if self.target_transform:
                y = self.target_transform(y)
            return x, y

        def get_text_for_class(self, class_id):  # Needs to live outside of the class....
            descriptions = self._dataset_dict[class_id]
            return descriptions[
                torch.randint(0, len(descriptions), (1,))
            ]  # Pick random description for the queried class

        @property
        def classes(self):
            return list(self._dataset_dict.keys())

        @property
        def class_to_idx(self):
            return self._class_to_idx

        @property
        def ds_dict(self):
            return self._dataset_dict  # Used for e.g. splitting the dataset


    class BimodalDS(torch.utils.data.Dataset):
        """
        Combines two modalities into a dataset. Example:

            text_ds = load_text_ds('val_text.json')
            img_ds = load_img_ds('val')
            bimodal_ds = BimodalDS(img_ds, text_ds)
        """

        def __init__(self, image_ds: "DataHandler.ImageModalityDS", text_ds: "DataHandler.TextModalityDS"):
            self._image_ds = image_ds
            self._text_ds = text_ds

        def __len__(self):
            return len(self._image_ds)  # Image modality is larger

        def __getitem__(self, idx):
            x_img, y = self._image_ds[idx]
            x_text = self._text_ds.get_text_for_class(y)
            return (x_img, x_text), y

        @property
        def classes(self):
            return self._image_ds.classes

        @property
        def image_ds(self) -> "DataHandler.ImageModalityDS":
            return self._image_ds

        @property
        def text_ds(self) -> "DataHandler.TextModalityDS":
            return self._text_ds


    ###############
    ## Utilities ##
    ###############
    def split_img_ds(self,
        img_ds: "DataHandler.ImageModalityDS", train_val_split: float
    ) -> Tuple["DataHandler.ImageModalityDS", "DataHandler.ImageModalityDS"]:
        t = int(len(img_ds) * train_val_split)
        train_split, val_split = torch.utils.data.random_split(
            img_ds, [t, len(img_ds) - t], generator=torch.Generator().manual_seed(42)
        )
        return train_split, val_split


    def split_text_ds(self,
        text_ds: "DataHandler.TextModalityDS", train_val_split: float
    ) -> Tuple["DataHandler.TextModalityDS", "DataHandler.TextModalityDS"]:
        train_ds_dict, val_ds_dict = {}, {}
        ds_dict = text_ds.ds_dict
        for y, Xs in ds_dict.items():
            assert len(Xs), "Class {:d} has no text-instances".format(y)
            if len(Xs) > 1:
                t = int(len(Xs) * train_val_split)
                if t >= len(Xs):
                    t = len(Xs) - 1
                split_lengths = [t, len(Xs) - t]
                train_ds_dict[y], val_ds_dict[y] = torch.utils.data.random_split(
                    Xs, split_lengths, generator=torch.Generator().manual_seed(42)
                )
            elif len(Xs) == 1:
                X = Xs[0]
                s = X.split(" ")
                train_ds_dict[y], val_ds_dict[y] = " ".join(
                    s[: int(len(s) * 0.5)]
                ), " ".join(s[int(len(s) * 0.5) :])

            assert len(train_ds_dict[y]) and len(
                val_ds_dict[y]
            ), "Class {:d} has {:d} training instances and {:d} validation instances in the text modality (it requires at least one sentence that can be split)".format(
                y, len(train_ds_dict[y]), len(val_ds_dict[y])
            )

        return (
            DataHandler.TextModalityDS(
                outer=self,
                ds_dict=train_ds_dict,
                transform=text_ds.transform,
                target_transform=text_ds.target_transform,
            ),
            DataHandler.TextModalityDS(
                outer=self,
                ds_dict=val_ds_dict,
                transform=text_ds.transform,
                target_transform=text_ds.target_transform,
            ),
        )


def img_dataloader(
    ds: DataHandler.ImageModalityDS, shuffle=True, batch_size=32, **kwargs
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        ds,
        shuffle=shuffle,
        batch_size=batch_size,
        prefetch_factor=2,
        num_workers=2,
        **kwargs
    )


def text_dataloader(
    ds: DataHandler.TextModalityDS, shuffle=True, batch_size=32, **kwargs
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        ds, shuffle=shuffle, batch_size=batch_size, num_workers=2, **kwargs
    )


def bimodal_dataloader(
    ds: DataHandler.BimodalDS, shuffle=True, batch_size=32, **kwargs
) -> torch.utils.data.DataLoader:
    return img_dataloader(ds, shuffle=shuffle, batch_size=batch_size, **kwargs)


###############
#### Tests ####
###############
def test_text():
    dhandler = DataHandler('imagenet')
    ds = dhandler.TextModalityDS("val_text.json")
    for i in range(0, len(ds.classes)):
        for _ in range(0, 100):
            found = False
            rand_text = ds.get_text_for_class(i)
            # print("Search for: {:d} - {:s}".format(i, rand_text))
            for t, c in ds:
                # print("{:d} - {:s}".format(c, t))
                if rand_text == t and i == c:
                    found = True
                    break
            assert found


def test_split_text():
    dhandler = DataHandler('imagenet')
    ds = dhandler.TextModalityDS("val_text.json")
    train, val = dhandler.split_text_ds(ds, train_val_split=0.8)
    for i in range(len(ds.classes)):
        X_t = train.ds_dict[i]
        X_v = val.ds_dict[i]
        for X in X_t:
            assert X not in X_v  # X_train[i] elements not in X_val[i]
            assert X in ds.ds_dict[i]  # X_train[i] elements belong to the correct class
        for X in X_v:
            assert X not in X_t  # X_val[i] elements not in X_train[i]
            assert X in ds.ds_dict[i]  # X_val[i] elements belong to the correct class
        assert len(X_t) + len(X_v) == len(
            ds.ds_dict[i]
        )  # All class i elements are in either X_val[i] or X_train[i]
