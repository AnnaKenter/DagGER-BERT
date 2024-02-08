import argparse
import os
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings('ignore')

import torch
import numpy as np
from .data_helpers import *
from .evaluation import *
import random
from torch.utils.data import DataLoader


def train(train_loader, val_loader, mode, lexicon, basemode):

    print('Training model...')

    labels_counter = Counter()

    for batch in train_loader:

        sents, masks, segs, idxes_mask, labels = batch

        if mode == 'pfx' or mode == 'sfx':

            labels_counter.update(labels.tolist())

        elif mode == 'both':

            labels_counter.update([tuple(l) for l in labels.tolist()])

    preds = list(labels_counter.keys())

    mrr_micro, mrr_macro_dict = test(val_loader, mode, preds)

    mrr_macro = np.mean(list(mrr_macro_dict.values()))

    print('MRR@10 (micro):\t{:.4f}'.format(mrr_micro))
    print('MRR@10 (mcro):\t{:.4f}'.format(mrr_macro))
    print('Best:\t', sorted(mrr_macro_dict.items(), key=lambda x: x[1], reverse=True)[:5])
    print('Worst:\t', sorted(mrr_macro_dict.items(), key=lambda x: x[1])[:5])

    with open('/netscratch/kenter/results_final/results_random_{}_{}_{}_micro.txt'.format(lexicon, mode, basemode), 'a+') as f:
        f.write('{:.3f} & '.format(mrr_micro))

    with open('/netscratch/kenter/results_final/results_random_{}_{}_{}_macro.txt'.format(lexicon, mode, basemode), 'a+') as f:
        f.write('{:.3f} & '.format(mrr_macro))


def test(test_loader, mode, preds):

    print('Evaluating model...')

    y_true = list()
    y_pred = list()

    for batch in test_loader:

        sents, masks, segs, idxes_mask, labels = batch

        # Store labels and predictions
        if mode == 'pfx' or mode == 'sfx':
            y_true.extend(labels.tolist())
        elif mode == 'both':
            y_true.extend([tuple(l) for l in labels.tolist()])

        for _ in range(len(labels)):
            y_pred.append(random.sample(preds, len(preds)))

    return mrr_micro(y_true, y_pred, 10), mrr_macro(y_true, y_pred, 10)


def main():

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default=None, type=str, required=True, help='Affix type.')
    parser.add_argument('--basemode', default=None, type=str, required=True, help='Base type.')
    parser.add_argument('--batch_size', default=None, type=int, required=True, help='Batch size.')
    parser.add_argument('--lexicon', default=None, type=str, required=True, help='Lexicon setting')

    args = parser.parse_args()

    for count in [1, 2, 3, 4, 5, 6, 7, 8]:

        print('Mode: {}'.format(args.mode))
        print('Count: {}'.format(count))
        print('Lexicon setting: {}'.format(args.lexicon))
        print('Batch size: {}'.format(args.batch_size))

        # Define poath to data
        inpath = '/netscratch/kenter/train_data'

        if args.lexicon == "split":
            if args.basemode == "baseplus":
                train_path = f"{args.mode}_{count}_{args.lexicon}_base_train.xlsx"
                val_path = f"{args.mode}_{count}_{args.lexicon}_base_test.xlsx"
            else:
                train_path = f"{args.mode}_{count}_{args.lexicon}_{args.basemode}_train.xlsx"
                val_path = f"{args.mode}_{count}_{args.lexicon}_{args.basemode}_test.xlsx"
        else:
            train_path = f"{args.mode}_{count}_{args.lexicon}_train.xlsx"
            val_path = f"{args.mode}_{count}_{args.lexicon}_test.xlsx"
        train_path = os.path.join(inpath, train_path)
        val_path = os.path.join(inpath, val_path)

        # Initialize train loader
        print('Load training data...')
        try:
            train_data = AffixDataset(train_path, args.mode, args.basemode)
        except FileNotFoundError:
            print('Bin not found.')
            continue

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_sents)

        # Initialize val loader
        print('Load validation data...')
        val_data = AffixDataset(val_path, args.mode, args.basemode)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, collate_fn=collate_sents)

        train(train_loader, val_loader, args.mode, args.lexicon, basemode=args.basemode)


if __name__ == '__main__':
    main()
