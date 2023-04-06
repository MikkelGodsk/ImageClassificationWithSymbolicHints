"""
    Time and memory requirements to run (run at Volta-100 GPU):
        - Memory: ~64GB RAM. Once ran out of memory at 40GB RAM, but LSF says it used at max ~20GB RAM.
        - Time: ~10h.

    Expected memory for the embeddings alone:
        2x  14.304.277.504 bytes (on disk)
        14.304.277.504 bytes in RAM
        
    Add some room for the models and dataset as well. Dataset is loaded incrementally, so it won't take up too much mem.
    
    
    Status:
        - Ran out of memory at 40GB RAM. Trying 64GB
"""

import os

import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow_datasets as tfds
import numpy as np

from models import get_bert, get_resnet50_no_top
import dataset


dynamic_p = 0.5 #1.0

DEBUG = False
DEBUG_NO = 10  # No. batches to log.

############
# Embedder #
############
def create_embedder(verbose=True):
    if verbose:
        print("Loading ResNet50")
    resnet_embedder = get_resnet50_no_top()
    if verbose:
        print("Loading BERT")
    bert_embedder = get_bert(final_op='avg_sequence')
    if verbose:
        print("Constructing embedder")
        
    img_input_layer = tf.keras.layers.Input(shape=(224, 224, 3), dtype=tf.float32, name='image')
    txt_input_layer = tf.keras.layers.Input(shape=(1,), dtype=tf.string, name='description')
    input_layer = {'image': img_input_layer, 
                   'description': txt_input_layer}
    
    mixed_embedder = tf.keras.Model(
        inputs=input_layer,  
        outputs=tf.keras.layers.Concatenate(axis=-1)([resnet_embedder(img_input_layer), 
                                                      bert_embedder(txt_input_layer)])
    )
    return mixed_embedder, (resnet_embedder, bert_embedder)


############
# Datasets #
############
def create_dataset(kb, batch_size, dynamic_p=0.7, train_val_split=0.8, verbose=True, seed=None):
    test_kb = None
    if kb.lower() == 'wikipedia':
        test_kb = 'wordnet'
    elif kb.lower() == 'wordnet':
        test_kb = 'wikipedia'
    elif kb.lower() == 'both':
        test_kb = None
    else:
        raise ValueError('Knowledgebase argument (kb) must be either \'wordnet\', \'wikipedia\' or \'both\'')
    
    if verbose:
        print("Creating wikipedia dataset handler")
    ds_handler = dataset.DatasetHandler(
        train_kb=kb,
        val_kb=kb,
        test_kb=test_kb,
        train_val_split=train_val_split,
        seed=seed
    ).drop_observations_without_hints(for_all_partitions=True)
    
    if verbose:
        print("Create training set")
    train_data = ds_handler.mask_words(
        dataset=ds_handler.training_set,
        p=tf.constant(dynamic_p),
        decreasing_p = False,
        seed=seed
    )
    train_data = dataset.make_batched_and_prefetched(train_data, batch_size=batch_size)
    
    if verbose:
        print("Create validation set")
    validation_data = ds_handler.mask_words(
        dataset=ds_handler.validation_set,
        p=tf.constant(dynamic_p),
        decreasing_p = False,
        seed=seed
    )
    validation_data = dataset.make_batched_and_prefetched(validation_data, batch_size=batch_size)
    
    if verbose:
        print("Create test set")
    ds_handler._test_set = ds_handler.mask_words(
        dataset = ds_handler.test_set,
        p=tf.constant(dynamic_p),
        decreasing_p = False,
        seed=seed
    )
    ds_handler._test_set = dataset.make_batched_and_prefetched(ds_handler.test_set, batch_size=batch_size)
    
    return train_data, validation_data, ds_handler

##############
# Embed data #
##############
def embed_dataset(embedder, dataset, embeddings=None, labels=None, verbose=True, dataset_size=None, assertion=True):
    """
        Embeds the dataset with the embedder. Pre-allocated embeddings can be given using the "embedding" argument.
        
        Only pass already pre-allocated data if the dataset and model output dimensions haven't changed in size, and this function preallocated earlier.
        
        Params:
            embedder - the embedder
            dataset - the tf.Dataset
            embeddings - preallocated embedding matrix (np.array)
            verbose - whether to print progress.
            dataset_size - size of dataset in order to do preallocation
            assertion - whether to check if all datapoints have been embedded. Should be True unless there is a good reason not to.
    
        
    """
    no_dimensions = embedder.output.shape[1]
    
    if embeddings is None or labels is None:
        if dataset_size is None:
            raise ValueError("Pre-allocated embedding-matrix + labels vector or dataset_size not given")
        matrix_shape = (dataset_size, no_dimensions)
        if verbose:
            print("Pre-allocating for embeddings")
        embeddings = np.empty(shape=matrix_shape, dtype=np.float32)
        labels = np.empty(shape=(dataset_size,), dtype=np.int32)   # No no no..... Originally this was set to uint8 which is insufficiently large.
        ## Issue: When instances have been removed, we need to allocate a smaller dataset.

    if verbose:
        print("Embedding dataset")
        
    f_obj = None
    is_file_stream_open = False
    if DEBUG:
        f_obj = open('embedding_dataset_debug_p_{:.1f}.txt'.format(dynamic_p), 'a')
        f_obj.write('------------- New input -------------')
        is_file_stream_open = True
        
    total_obs = 0    
    for i, sample in enumerate(dataset):
        X_batch, y_batch = sample
        if DEBUG and i<DEBUG_NO:
            f_obj.write(str(X_batch['description']) + ' ' + str(X_batch['image']) + "\n")
        if DEBUG and is_file_stream_open and i >= DEBUG_NO:
            f_obj.close()
        no_obs = np.size(y_batch)
        embeddings[no_obs*i:no_obs*(i+1), :] = embedder.predict(X_batch, steps=1)
        labels[total_obs:total_obs+no_obs] = y_batch
        total_obs += no_obs
    
    if is_file_stream_open:
        f_obj.close()
    
    if assertion:
        assert total_obs == embeddings.shape[0]
        assert total_obs == labels.shape[0]
    
    return embeddings, labels
    

def full_test():
    batch_size = 100
    no_embeddings_train = int(1015924*0.05)
    no_embeddings_test = 49550
    embedder, _ = create_embedder()
    training_data, validation_data, ds_handler = create_dataset(
        kb='wikipedia', 
        batch_size=batch_size, 
        train_val_split=0.01
    )
    #train_embeddings, labels = embed_dataset(
    #    embedder=embedder,
    #    dataset=training_data,
    #    dataset_size=no_embeddings_train,
    #    assertion=False
    #)
    test_embeddings, test_labels = embed_dataset(   # Is given WordNet hints
            embedder=embedder,
            dataset=ds_handler.test_set,
            dataset_size=no_embeddings_test
    )
    

if __name__ == '__main__':
    test = False
    if test:
        full_test()
    else:
        ##############
        # File paths #
        ##############
        dataset_path = '/work3/s184399/embedding_dataset'
        wiki_train_embed_fname = 'train_split_embeddings_wikipedia_hints_dynamic_p_{:.1f}.npy'.format(dynamic_p)
        wiki_train_label_fname = 'train_split_labels_wikipedia_hints_dynamic_p_{:.1f}.npy'.format(dynamic_p)
        wiki_val_embed_fname   = 'val_split_embeddings_wikipedia_hints_dynamic_p_{:.1f}.npy'.format(dynamic_p)
        wiki_val_label_fname   = 'val_split_labels_wikipedia_hints_dynamic_p_{:.1f}.npy'.format(dynamic_p)
        wiki_test_embed_fname  = 'val_set_embeddings_wikipedia_hints_dynamic_p_{:.1f}.npy'.format(dynamic_p)
        wiki_test_label_fname  = 'val_set_labels_wikipedia_hints_dynamic_p_{:.1f}.npy'.format(dynamic_p)
        wnet_train_embed_fname = 'train_split_embeddings_wordnet_hints_dynamic_p_{:.1f}.npy'.format(dynamic_p)
        wnet_train_label_fname = 'train_split_labels_wordnet_hints_dynamic_p_{:.1f}.npy'.format(dynamic_p)
        wnet_val_embed_fname   = 'val_split_embeddings_wordnet_hints_dynamic_p_{:.1f}.npy'.format(dynamic_p)
        wnet_val_label_fname   = 'val_split_labels_wordnet_hints_dynamic_p_{:.1f}.npy'.format(dynamic_p)
        wnet_test_embed_fname  = 'val_set_embeddings_wordnet_hints_dynamic_p_{:.1f}.npy'.format(dynamic_p)
        wnet_test_label_fname  = 'val_set_labels_wordnet_hints_dynamic_p_{:.1f}.npy'.format(dynamic_p)

        wiki_train_embed_fname = os.path.join(dataset_path, wiki_train_embed_fname)
        wiki_train_label_fname = os.path.join(dataset_path, wiki_train_label_fname)
        wiki_val_embed_fname = os.path.join(dataset_path, wiki_val_embed_fname)
        wiki_val_label_fname = os.path.join(dataset_path, wiki_val_label_fname)
        wiki_test_embed_fname = os.path.join(dataset_path, wiki_test_embed_fname)
        wiki_test_label_fname = os.path.join(dataset_path, wiki_test_label_fname)
        wnet_train_embed_fname = os.path.join(dataset_path, wnet_train_embed_fname)
        wnet_train_label_fname = os.path.join(dataset_path, wnet_train_label_fname)
        wnet_val_embed_fname = os.path.join(dataset_path, wnet_val_embed_fname)
        wnet_val_label_fname = os.path.join(dataset_path, wnet_val_label_fname)
        wnet_test_embed_fname = os.path.join(dataset_path, wnet_test_embed_fname)
        wnet_test_label_fname = os.path.join(dataset_path, wnet_test_label_fname)

        ##############
        # Parameters #
        ##############
        no_dimensions = 2816
        no_embeddings_train = 1015924  # With classes removed
        no_embeddings_val = 253987     # With classes removed
        no_embeddings_test = 49550     # With classes removed

        batch_size = 100

        ###########
        # General #
        ###########
        embedder, _ = create_embedder()

        ###################
        # Script for Wiki #
        ###################
        training_data, validation_data, ds_handler = create_dataset(
            kb='wikipedia', 
            batch_size=batch_size,
            dynamic_p=dynamic_p
        )
        train_embeddings, train_labels = embed_dataset(
            embedder=embedder,
            dataset=training_data,
            dataset_size=no_embeddings_train
        )
        val_embeddings, val_labels = embed_dataset(
            embedder=embedder,
            dataset=validation_data,
            dataset_size=no_embeddings_val
        )
        test_embeddings, test_labels = embed_dataset(   # Is given WordNet hints
            embedder=embedder,
            dataset=ds_handler.test_set,
            dataset_size=no_embeddings_test
        )
        np.save(wiki_train_embed_fname, train_embeddings)
        np.save(wiki_train_label_fname, train_labels)
        np.save(wiki_val_embed_fname, val_embeddings)
        np.save(wiki_val_label_fname, val_labels)
        np.save(wnet_test_embed_fname, test_embeddings)  # Note: WordNet and Wiki switched around to enable removal of classes with no Wikipedia hints!
        np.save(wnet_test_label_fname, test_labels)

        ######################
        # Script for WordNet #
        ######################
        training_data, validation_data, ds_handler = create_dataset(
            kb='wordnet', 
            batch_size=batch_size,
            dynamic_p=dynamic_p
        )
        train_embeddings, train_labels = embed_dataset(
            embedder=embedder,
            dataset=training_data,
            embeddings=train_embeddings,
            labels=train_labels
        )
        val_embeddings, val_labels = embed_dataset(
            embedder=embedder,
            dataset=validation_data,
            embeddings=val_embeddings, 
            labels=val_labels
        )
        test_embeddings, test_labels = embed_dataset(   # Is given WordNet hints
            embedder=embedder,
            dataset=ds_handler.test_set,
            embeddings=test_embeddings,
            labels=test_labels
        )
        np.save(wnet_train_embed_fname, train_embeddings)
        np.save(wnet_train_label_fname, train_labels)
        np.save(wnet_val_embed_fname, val_embeddings)
        np.save(wnet_val_label_fname, val_labels)
        np.save(wiki_test_embed_fname, test_embeddings)  # Note: WordNet and Wiki switched around to enable removal of classes with no Wikipedia hints!
        np.save(wiki_test_label_fname, test_labels)
        