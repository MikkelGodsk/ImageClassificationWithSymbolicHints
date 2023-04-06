import os

import click
import xgboost as xgb
import numpy as np
import pickle
from sklearn.metrics import top_k_accuracy_score


def load_pickle(f_name):
    with open(f_name, 'rb') as f_obj:
        data = pickle.load(f_obj)
    return data


@click.command()
@click.option(
    "--dataset",
    default="CMPlaces",
    help="Dataset to use. Either 'CMPlaces' or 'ImageNet'"
)
@click.option(
    "--ds_dir",
    default="/work3/s184399",
    help="The directory in which the datasets are located"
)
@click.option(
    "--n_workers",
    default=1,
    help="The number of workers used for XGBoost and SVM"
)
@click.option(
    "--xgb_subsample",
    default=1.,
    help="Rate of subsampling (1 to disable)"
)
@click.option(
    "--verbosity",
    default=0,
    help="The verbosity level. From 0 (silent) to 3 (debug)."
)
def main(*args, **kwargs):
    dataset_name = kwargs["dataset"]
    ds_dir = kwargs['ds_dir']
    n_jobs = kwargs['n_workers']
    subsample = kwargs['xgb_subsample']
    verbosity = kwargs['verbosity']

    print("Using {:d} workers".format(n_jobs))

    # Load data
    print("Loading data", flush=True)
    train = load_pickle(os.path.join(ds_dir, dataset_name, 'train.pkl'))
    val = load_pickle(os.path.join(ds_dir, dataset_name, 'val.pkl'))
    num_classes = np.max(val['y'])
    sample_weight = np.vstack((np.ones((len(train['y']),1)), np.zeros((num_classes,1))))  # Add these so it will not complain about missing classes

    # Train model
    print("Training model", flush=True)
    i = 1
    prev_clf = None
    while True:
        clf = xgb.XGBClassifier(
            n_jobs=n_jobs, 
            subsample=(None if subsample==1 else subsample), 
            verbosity=verbosity,
            objective='multi:softmax',
            num_class=num_classes,
            n_estimators=1
        )
        clf.fit(
            np.vstack((train['X'], np.zeros((num_classes, train['X'].shape[1])))), 
            np.vstack((train['y'], np.arange(num_classes).reshape((num_classes,1)))),
            sample_weight=sample_weight,
            verbose=True,
            xgb_model=prev_clf
        )

        # Predict
        print("\nEvaluating model with {:d} estimators:".format(i), flush=True)
        y_probs = clf.predict_proba(val['X'])
        top1 = top_k_accuracy_score(val['y'], y_probs, k=1)
        top5 = top_k_accuracy_score(val['y'], y_probs, k=5)
        print("Top 1 accuracy: {:f}%".format(top1*100), flush=True)
        print("Top 5 accuracy: {:f}%".format(top5*100), flush=True)

        i+=1
        prev_clf = clf.get_booster()


if __name__ == '__main__':
    main()