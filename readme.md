# Read-me:
In order to run this code, you need to have the dataset and VGG16-models setup. Follow the instructions further down.

To run the experiments, run the `main.py` file. Here, you should set `load_bert` to `False` in the `main()` function the first time you run the code as it will otherwise look for a saved model.
Hyperparameter tuning was done using the `tune_addition_based_model.py`, `tune_BERT_hparams.py`, `tune_svm.py` files.

## Software used:
Software:
- Python 3.8.2
- cuda 11.3
- cudnn v8.2.0.53-prod-cuda-11.3
- mmdnn 0.3.1 (https://github.com/microsoft/MMdnn)
- tensorboard 2.8.0

Python packages:
- torch 1.12.0+cu102
- torchvision 0.13.0+cu102
- torchmetrics 0.9.3
- optuna 2.10.0
- numpy 1.22.2
- nltk 3.6.2
- scipy 1.7.1
- wikipediaapi 0.5.4 (https://github.com/martin-majlis/Wikipedia-API)
- wikidata 0.7.0
- pytorch_lightning 1.6.3
- sklearn 1.0.1
- matplotlib 3.5.1
- transformers 4.21.0
- tqdm 4.62.2


## Prepare CMPlaces to work with the code
In the Python code, the folder `/work3/s184399/CMPlaces` will be present in multiple places. This should be replaced with whatever folder you, the user, are storing your dataset in.

**Changes to make**<br>
Let `[folder]` denote the folder you store your dataset in.
- Change `ds_dir` in `dataset.py` to `[folder]/UnpackedDataset`. 
- Change `dataset_directory_src` in `cmplaces_preparation.py` to `[folder]`.
- Change `dataset_directory_dest` in `cmplaces_preparation.py` to `[folder]/UnpackedDataset`.

### Download dataset
Run:
`wget -P [folder] [source_url]`

Files needed:
- `descriptions.zip` (from https://projects.csail.mit.edu/cmplaces/)
- `cmplaces.zip` (from https://projects.csail.mit.edu/cmplaces/)
- `imagesPlaces205_resize.tar.gz` (from http://places.csail.mit.edu/)
- `trainvalsplit_places205.tar.gz` (from http://places.csail.mit.edu/)

### Unpack
```
# Unpack text
mkdir [folder]/descriptions
unzip -q [folder]/descriptions.zip -d [folder]/descriptions

# Unpack images
mkdir /work3/s184399/CMPlaces/images
tar -xf [folder]/imagesPlaces205_resize.tar.gz -C [folder]/images

# Unpack split
mkdir /work3/s184399/CMPlaces/trainvalsplit
tar -xf [folder]/trainvalsplit_places205.tar.gz -C [folder]/trainvalsplit

# Unpack "starter kit"
mkdir /work3/s184399/CMPlaces/starterkit
unzip -q [folder]/cmplaces.zip -d [folder]/starterkit

# Copy categories
mkdir [folder]/UnpackedDataset
cp [folder]/starterkit/labels/categories.txt [folder]/UnpackedDataset/categories.txt
```

### Preparation
Time to run: approx 24h
```
python3 cmplaces_preparation.py
```

## Convert Caffe model to PyTorch
**Sources:**
* https://github.com/microsoft/MMdnn#conversion
* https://github.com/Microsoft/MMdnn/blob/master/mmdnn/conversion/pytorch/README.md
* https://github.com/Microsoft/MMdnn/blob/master/mmdnn/conversion/caffe/README.md


**Changes to make**<br>
To run the code, go to `VGG16.py` and put your weight file source (should correspond to `[folder]/places205VGG16-torch`).

### Download
Download `places205vgg.tar.gz` (from http://places.csail.mit.edu/)
```
wget -P [folder] [source_url]
tar -xvf [folder]/places205vgg.tar.gz
```

### Conversion from Caffe to PyTorch
The following will convert the Caffe model to a `.py` and `.npy` file containing the model architecture and weights:
```
pip3 install mmdnn

mkdir [folder]/places205VGG16-IR
mkdir [folder]/places205VGG16-torch

mmtoir -f caffe -n [folder]/places205VGG16/deploy_10.prototxt -w [folder]/places205VGG16/snapshot_iter_765280.caffemodel -o [folder]/places205VGG16-IR/caffe_vgg16_IR

mmtocode -f pytorch -n [folder]/places205VGG16-IR/caffe_vgg16_IR.pb --IRWeightPath [folder]/places205VGG16-IR/caffe_vgg16_IR.npy --dstModelPath [folder]/places205VGG16-torch/pytorch_vgg16.py -dw [folder]/places205VGG16-torch/pytorch_vgg16.npy
```

**Note:**<br>
There has been made some changes to the outputted `.py` file. Some `.flatten()` operations and more. Therefore the file from the repository should be used instead.

## Prepare imagenet
In the Python code, the folder `/work3/s184399/ImageNet` will be present in multiple places. This should be replaced with whatever folder you, the user, are storing your dataset in.

### Download and prepare image dataset
To start with, download `ILSVRC2012_img_train.tar`, `ILSVRC2012_img_val.tar` and `ILSVRC2012_devkit_t12.tar.gz` from the ImageNet website (https://image-net.org/ and navigate to the 2012-challenge) into a download-folder `[imagenet_folder]`. E.g. with `wget -P [imagenet_folder] [source_url]`

Next we create some files and untar:
```
mkdir [imagenet_folder]/UnpackedDataset
mkdir [imagenet_folder]/UnpackedDataset/train
mkdir [imagenet_folder]/val_imgs
mkdir [imagenet_folder]

tar -xf [imagenet_folder]/ILSVRC2012_img_train.tar -C [imagenet_folder]/UnpackedDataset/train
tar -xf [imagenet_folder]/ILSVRC2012_img_val.tar -C [imagenet_folder]/val_imgs
tar -xzf [imagenet_folder]/ILSVRC2012_devkit_t12.tar.gz -C [imagenet_folder]/dev_kit

for SYNSET in [imagenet_folder]/UnpackedDataset/train/*.tar; do
    mkdir ${SYNSET%.tar}
    tar -xf $SYNSET -C ${SYNSET%.tar}
done
```

Since the validation set has a different format, we use Python to create the appropriate folders here. See "Download WordNet and prepare dataset".


### Download Wikipedia-text
Download the `mapping.json` from ImageNet to WordNet by Philipiak et al.: https://github.com/DominikFilipiak/imagenet-to-wikidata-mapping

Also, we need to install the following libraries:
- Wikipedia-API (https://github.com/martin-majlis/Wikipedia-API)
- wikidata
```
pip3 install wikidata wikipedia-api
python3 get_wiki_descriptions.py
```

### Download WordNet and prepare dataset
Again, a path needs to be set in the code.
```
python3 imagenet_preparation.py
```