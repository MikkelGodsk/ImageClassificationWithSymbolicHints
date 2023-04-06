import typing

import numpy as np
from memory_profiler import profile

from src.experiments.main import *


@profile
def test_model_fit_predict():
    """
        Used for TDD of XGBoost fusion. Prints stuff... what a mess

        Trains model on a very small dataset to ensure that we can actually fit to it.
    """
    dataset_name = "CMPlaces"
    ds_handler = DataHandler(dataset_name)

    text_ds: DataHandler.TextModalityDS = ds_handler.load_text_ds('val_text.json')  # For debugging only
    img_ds: DataHandler.ImageModalityDS = ds_handler.load_img_ds('val')
    
    train_val_split = 0.8
    train_text_ds, val_text_ds = ds_handler.split_text_ds(text_ds, train_val_split=train_val_split)
    train_img_ds, val_img_ds = ds_handler.split_img_ds(img_ds, train_val_split=train_val_split)
    train_split = DataHandler.BimodalDS(image_ds=train_img_ds, text_ds=train_text_ds)
    val_split = DataHandler.BimodalDS(image_ds=val_img_ds, text_ds=val_text_ds)
    print(len(val_split))

    bert_clf: text_clf_t = get_text_classifier(load_bert=True, train_text_ds=train_text_ds, dataset_name=dataset_name)
    vgg16_clf: img_clf_t = get_img_classifier(dataset_name=dataset_name)

    print("\033[1m\033[Fit model\033[0m")
    xgb_fusion = MultimodalModel.XGBFusion(vgg16_clf, bert_clf, cache_dir='/work3/s184399/cache_test')
    xgb_fusion.cuda()
    xgb_fusion.fit_top(
        bimodal_dataloader(val_split, shuffle=True, batch_size=32)
    )
    xgb_fusion.cuda()
    acc = evaluate_model(xgb_fusion, bimodal_dataloader(val_split), 'XGBoost')
    assert acc > 0.5


if __name__ == '__main__':
    test_model_fit_predict()