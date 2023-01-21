from src.experiments.main import *

val_text_ds: TextModalityDS = load_text_ds('val_text.json')

n_bins = 25

bert_params = {
    'learning_rate': 0.0006870443398072322,
    'adam_epsilon': 1e-08,
    'weight_decay': 0.0,
    'dropout_rate': 0.31547838646677684,
    'top_dense_layer_units': [],
}
bert_clf: BERT.LitBERTModel = BERT.LitBERTModel(**bert_params)
bert_clf.load_state_dict(torch.load("BERT_model_aaa"))

temperatures, nll, avg_conf, ece, mce = LitModel.manual_search_plot(bert_clf, text_dataloader(val_text_ds), model_name="BERT", calibration_lr=0.3, calibration_max_iter=50)
#for t, conf in zip(temperatures, avg_conf):
#    print("{:.3f} - {:.4f}".format(t, conf))


vgg16_clf: VGG16.LitVGG16Model = VGG16.LitVGG16Model()
val_img_ds: ImageModalityDS = load_img_ds('val')
LitModel.manual_search_plot(vgg16_clf, img_dataloader(val_img_ds), model_name="VGG16", calibration_lr=0.1)
