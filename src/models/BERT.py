import typing

import optuna
from transformers import AdamW, BertModel, BertTokenizer

from src.models.LitModel import *

#### BERT ####
max_length = 128


class LitBERTModel(LitModel):
    """
    Modified from https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/text-transformers.html
    """

    def __init__(
        self,
        num_classes: int = 205,
        bert_embedding_dim=768,
        bert_model_name: str = "bert-base-cased",
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        weight_decay: float = 0.0,
        dropout_rate: float = 0.0,
        top_dense_layer_units: typing.List[int] = [],
    ):
        super().__init__()
        self.save_hyperparameters()

        self.freeze_net = False

        # Model
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.net = BertModel.from_pretrained(bert_model_name).to(gpu)
        self._top = torch.nn.Sequential()
        if top_dense_layer_units != []:
            for n_units in top_dense_layer_units:
                self._top.append(torch.nn.LazyBatchNorm1d())
                self._top.append(torch.nn.Dropout(dropout_rate))
                self._top.append(torch.nn.LazyLinear(out_features=n_units))
                self._top.append(torch.nn.ReLU())
            self._top.append(
                torch.nn.Linear(
                    in_features=top_dense_layer_units[-1], out_features=num_classes
                )
            )
        else:
            self._top.append(torch.nn.LazyBatchNorm1d())
            self._top.append(torch.nn.Dropout(dropout_rate))
            self._top.append(
                torch.nn.Linear(
                    in_features=bert_embedding_dim, out_features=num_classes
                )
            )

        self._top = self._top.to(gpu)

    def preprocess(self, x):
        return self.tokenizer(
            x,
            padding=True,
            return_tensors="pt",
        ).to(self.net.device)

    def forward_no_top(self, x):
        """
        Takes: A batch of text
        Returns: Average word embedding
        """
        # Tokenize
        x1 = self.preprocess(x)

        # Forward pass in BERT
        x2 = self.net(**x1)

        # Return average word embedding
        return x2.last_hidden_state.mean(dim=1)

    def forward_no_softmax(self, x):
        x2 = self.forward_no_top(x)
        x3 = self._top(x2)
        return x3

    def _configure_optim_train(self):
        return torch.optim.AdamW(
            self.parameters() if not self.freeze_net else self.top.parameters(),
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
            weight_decay=self.hparams.weight_decay,
        )

    @property
    def top(self):
        return self._top


def train_and_evaluate(params, train_ds, val_ds=None, n_epochs=20, model_version=""):
    logger = pl.loggers.TensorBoardLogger(
        save_dir="~/bscproj/CMPlaces", name="lightning_logs", version=model_version
    )

    model = LitBERTModel(**params)
    model.freeze_net = True
    trainer = pl.Trainer(
        gpus=min(1, torch.cuda.device_count()),
        max_epochs=n_epochs,
        logger=logger,
        # checkpoint_callback=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    trainer.fit(model, train_ds, val_ds)
    return model, trainer
