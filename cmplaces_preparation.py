import os
import json
import csv


# Source files:
dataset_directory_src = '/work3/s184399/CMPlaces'
description_folder_src = os.path.join(dataset_directory_src, 'descriptions/text')
image_folder_src = os.path.join(dataset_directory_src, 'images/data/vision/torralba/deeplearning/images256')

image_train_split = os.path.join(dataset_directory_src, 'trainvalsplit/trainvalsplit_places205/train_places205.csv')
image_val_split = os.path.join(dataset_directory_src, 'trainvalsplit/trainvalsplit_places205/val_places205.csv')
text_train_split = os.path.join(dataset_directory_src, 'starterkit/labels/text_train.txt')
text_val_split = os.path.join(dataset_directory_src, 'starterkit/labels/text_val.txt')

# Destination files:
dataset_directory_dest = '/work3/s184399/CMPlaces/UnpackedDataset'

train_dest = os.path.join(dataset_directory_dest, 'train')
val_dest = os.path.join(dataset_directory_dest, 'val')


def get_files_from_split(src):
    split_files = []
    with open(src, 'r') as f:
        rows = csv.reader(f, delimiter=' ')
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
    if os.path.isdir(dest):
        # If directory already exists, abort
        print("Path {:s} exists already".format(dest))
        return False
    
    # Create new folder
    os.mkdir(dest)
    
    # Loop through each subfolder and create symlinks to the files
    depth_baseline = len(src.split('/'))
    for root, folders, files in os.walk(src):
        depth = len(root.split('/'))
        class_name = "_".join(root.split('/')[(depth_baseline+1):])  # Add one to descend into the folder with just 1 letter, e.g. 'a'
        if len(files) > 0:
            os.mkdir(os.path.join(dest, class_name))
            print(os.path.join(dest, class_name))
        for file in files:
            file_abs_path = os.path.join(root, file)
            if os.path.relpath(file_abs_path, start=src) in split:
                os.symlink(src=file_abs_path,
                           dst=os.path.join(dest, class_name, file))

    return True
    

def test_images(collected_images, split):
    # Are the collected files a subset of the split?
    split = list(map(lambda s: s.split('/')[-1], split))
    
    for root, folders, files in os.walk(collected_images):
        for file in files:
            if not file in split:
                print("File {:s} does not belong!".format(os.path.join(root,file)))
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
    if os.path.isfile(dest):
        print("Path {:s} already exists".format(dest))
        return False
    
    # Create a json file where for each category, we have a key-value pair of category x list of descriptions.
    categories = os.listdir(src)
    desc_dict = {}
    
    for cat in categories:
        desc_dict[cat] = []
        for file in os.listdir(os.path.join(src, cat)):  # The file names, not rel. or abs. paths
            file_rel_path = os.path.join('data', 'text', cat, file)
            if file_rel_path in split:
                fname = os.path.join(src, cat, file)
                with open(fname, 'r') as f_obj:
                    desc_dict[cat].append(f_obj.read())
    with open(dest, 'w') as f_obj:
        json.dump(desc_dict, f_obj)
        
    return True
    
    

# Prepare dataset
#collect_text()
#collect_images()
#check_text_image_consistency()




# Try loading the dataset
#ds = tf.keras.utils.image_dataset_from_directory('CMPlaces/CMPimages', follow_links=True, batch_size=1)

if __name__ == '__main__':
    # Prepare images
    train_split = get_files_from_split(image_train_split)
    collect_images(src=image_folder_src, dest=train_dest, split=train_split)
    val_split = get_files_from_split(image_val_split)
    collect_images(src=image_folder_src, dest=val_dest, split=val_split)
    #test_images(collected_images = val_dest, split=val_split)
    
    # Prepare text
    train_split = get_files_from_split(text_train_split)
    collect_text(src=description_folder_src, dest=os.path.join(dataset_directory_dest, 'train_text.json'), split=train_split)
    val_split = get_files_from_split(text_val_split)
    collect_text(src=description_folder_src, dest=os.path.join(dataset_directory_dest, 'val_text.json'), split=val_split)
    