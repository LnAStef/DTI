# -*- coding: utf-8 -*-

import ampligraph
import openbiolink
from ampligraph import evaluation
from ampligraph.evaluation import evaluate_performance
from openbiolink.evaluation.dataLoader import DataLoader
from ampligraph.latent_features import ComplEx
import tensorflow as tf
import numpy as np
from ampligraph.latent_features import save_model, restore_model
import argparse


def train_model(save_dir, output_dir, epochs):
    # Name of the Dataset, possible values HQ_DIR, HQ_UNDIR, ALL_DIR, ALL_UNDIR. Default: HQ_DIR
    dl = DataLoader("HQ_DIR")

    cmp = []

    for el in dl.mappings['nodes']['label2id']:
      if 'compound' in el.lower():
        cmp.append(el)

    rels = set(dl.data['train_positive'][1])

    print(dl.mappings['relations']['label2id'])

    train_positive = dl.data['train_positive'].to_numpy()
    train_negative = dl.data['train_negative'].to_numpy()

    test_positive = dl.data['test_positive'].to_numpy()
    test_negative = dl.data['test_negative'].to_numpy()

    validation_positive = dl.data['valid_positive'].to_numpy()
    vakidation_negative = dl.data['valid_negative'].to_numpy()

    model = ComplEx(batches_count=250,
                    seed=0,
                    epochs=epochs,
                    k=200,
                    eta=30,
                    embedding_model_params = {'negative_corruption_entities': train_negative},
                    optimizer='adam',
                    optimizer_params={'lr':0.001},
                    loss='multiclass_nll',
                    regularizer='LP',
                    regularizer_params={'p': 3, 'lambda': 0.001},
                    verbose=True)

    tf.logging.set_verbosity(tf.logging.ERROR)
    print("---------------Training the model...---------------")
    model.fit(train_positive, early_stopping=False)

    save_model(model, save_dir)

    X_positive = np.concatenate((train_positive, test_positive, validation_positive), axis=0)

    print("---------------Evaluate the model...---------------")
    ranks = evaluate_performance(test_positive,
                                 model=model,
                                 filter_triples=X_positive,   # Corruption strategy filter defined above
                                 verbose=True)

    # save ranks
    print("---------------Saving the evaluation results...---------------")
    np.save(output_dir, np.asarray(ranks))


if __name__ == '__main__':

    # Construct the argument parser
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("-epochs", "--epochs", required=True,
                    help="Where to save the evaluation results.")

    ap.add_argument("-save-dir", "--save-dir", required=True,
                    help="Directory to save the trained model.")

    ap.add_argument("-output", "--output-dir", required=True,
                     help="Where to save the evaluation results.")

    args = ap.parse_args()
    print(args)
    save_dir = args.save_dir
    output_dir = args.output_dir
    epochs = args.epochs
    train_model(save_dir, output_dir, epochs)
