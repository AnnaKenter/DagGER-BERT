import argparse
import os
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import torch
import numpy as np
from .data_helpers import *
from .evaluation import *
import random
from .model import AffixPredictor
from torch import nn, optim, softmax
from torch.utils.data import DataLoader
from transformers import BertTokenizer


def train(train_loader, val_loader, model, mode, lr, n_epochs, cuda, count, lexicon, freeze, basemode):

    # Define device
    if torch.cuda.is_available():
        print("Available GPU found.")
    else:
        print("Fallback to CPU")
    #device = torch.device('cuda:{}'.format(cuda) if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define optimizer and training objective
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Move model to CUDA
    model = model.to(device)

    if mode == 'pfx' or mode == 'sfx':
        mrr_micro, mrr_macro_dict = test_single(val_loader, model, mode, cuda)
    elif mode == 'both':
        mrr_micro, mrr_macro_dict = test_both(val_loader, model, cuda)

    mrr_macro = np.mean(list(mrr_macro_dict.values()))

    print('MRR@10 (micro) before training:\t{:.4f}'.format(mrr_micro))
    print('MRR@10 (macro) before training:\t{:.4f}'.format(mrr_macro))
    print('Best:\t', sorted(mrr_macro_dict.items(), key=lambda x: x[1], reverse=True)[:5])
    print('Worst:\t', sorted(mrr_macro_dict.items(), key=lambda x: x[1])[:5])

    print('Training model...')

    best_mrr = None

    for epoch in range(1, n_epochs + 1):

        print('Starting epoch {}...'.format(epoch))

        model.train()

        for i, batch in enumerate(train_loader):

            if batch[0].size(-1) > 512:
                print('Skipping batch {}...'.format(i))
                continue

            if i % 100 == 0:
                print('Processed {} batches...'.format(i))

            optimizer.zero_grad()

            sents, masks, segs, idxes_mask, labels = batch

            sents, masks, segs, idxes_mask = sents.to(device), masks.to(device), segs.to(device), idxes_mask.to(device)

            # Forward pass
            try:
                output = model(sents, masks, segs, idxes_mask)
            except RuntimeError:
                print('Skipping batch {} in forward pass...'.format(i))
                continue

            labels = labels.to(device)

            # Compute loss
            if mode == 'pfx' or mode == 'sfx':
                loss = criterion(output, labels)
            elif mode == 'both':
                loss = criterion(output.reshape(-1, output.size(-1)), labels.reshape(-1))

            # Backpropagate loss
            try:
                loss.backward()
            except RuntimeError:
                print('Skipping batch {} in backpropagation...'.format(i))
                continue

            # Update weights
            optimizer.step()

            # Delete tensors to free memory
            del sents, masks, segs, idxes_mask, labels, output

        if mode == 'pfx' or mode == 'sfx':
            mrr_micro, mrr_macro_dict = test_single(val_loader, model, mode, cuda)
        elif mode == 'both':
            mrr_micro, mrr_macro_dict = test_both(val_loader, model, cuda)

        mrr_macro = np.mean(list(mrr_macro_dict.values()))

        print('MRR@10 (micro) after epoch {}:\t{:.4f}'.format(epoch, mrr_micro))
        print('MRR@10 (macro) after epoch {}:\t{:.4f}'.format(epoch, mrr_macro))
        print('Best:\t', sorted(mrr_macro_dict.items(), key=lambda x: x[1], reverse=True)[:5])
        print('Worst:\t', sorted(mrr_macro_dict.items(), key=lambda x: x[1])[:5])

        if freeze:
            with open('/netscratch/kenter/results/results_bert_freeze_{}_{}_{}_{:02d}_micro.txt'.format(lexicon, mode, basemode, count), 'a+') as f:
                f.write('{:.0e}\t{}\n'.format(lr, mrr_micro))
            with open('/netscratch/kenter/results/results_bert_freeze_{}_{}_{}_{:02d}_macro.txt'.format(lexicon, mode, basemode, count), 'a+') as f:
                f.write('{:.0e}\t{}\n'.format(lr, mrr_macro))
        else:
            with open('/netscratch/kenter/results/results_bert_{}_{}_{}_{:02d}_micro.txt'.format(lexicon, mode, basemode, count), 'a+') as f:
                f.write('{:.0e}\t{}\n'.format(lr, mrr_micro))
            with open('/netscratch/kenter/results/results_bert_{}_{}_{}_{:02d}_macro.txt'.format(lexicon, mode, basemode, count), 'a+') as f:
                f.write('{:.0e}\t{}\n'.format(lr, mrr_macro))

        if best_mrr is None or mrr_macro > best_mrr:

            best_mrr = mrr_macro

            if freeze:
                torch.save(model.state_dict(), '/netscratch/kenter/trained_models/model_bert_freeze_{}_{}_{}_{:02d}_{:.0e}.torch'.format(lexicon, mode, basemode, count, lr))
            else:
                torch.save(model.state_dict(), '/netscratch/kenter/trained_models/model_bert_{}_{}_{}_{:02d}_{:.0e}.torch'.format(lexicon, mode, basemode, count, lr))

    if freeze:
        with open('/netscratch/kenter/results/results_bert_freeze_{}_{}_{}_{:02d}_micro.txt'.format(lexicon, mode, basemode, count), 'a+') as f:
            f.write('\n')
        with open('/netscratch/kenter/results/results_bert_freeze_{}_{}_{}_{:02d}_macro.txt'.format(lexicon, mode, basemode, count), 'a+') as f:
            f.write('\n')
    else:
        with open('/netscratch/kenter/results/results_bert_{}_{}_{}_{:02d}_micro.txt'.format(lexicon, mode, basemode, count), 'a+') as f:
            f.write('\n')
        with open('/netscratch/kenter/results/results_bert_{}_{}_{}_{:02d}_macro.txt'.format(lexicon, mode, basemode, count), 'a+') as f:
            f.write('\n')


def test_single(test_loader, model, mode, cuda, save_preds=False, save_preds_path=""):

    tok = BertTokenizer.from_pretrained('dbmdz/bert-base-german-uncased')

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mode2afx = {'pfx': 'prefixes', 'sfx': 'suffixes'}

    # Initialize affix list
    afxes = []
    with open('/netscratch/kenter/lingu_data/{}.txt'.format(mode2afx[mode]), encoding="latin-1") as f:
        for l in f:
            if l.strip() == '' or l.strip() == 'abil':
                continue
            afxes.append(l.strip().lower())

    if mode == 'pfx':
        idxes_afx, _ = torch.sort(torch.tensor(tok.convert_tokens_to_ids(afxes)).to(device))

    elif mode == 'sfx':
        idxes_afx, _ = torch.sort(torch.tensor(tok.convert_tokens_to_ids(['##' + a for a in afxes])).to(device))

    print('Evaluating model...')

    y_true = list()
    y_pred = list()

    model.eval()

    with torch.no_grad():

        for batch in test_loader:

            sents, masks, segs, idxes_mask, labels = batch
            sents, masks, segs, idxes_mask = sents.to(device), masks.to(device), segs.to(device), idxes_mask.to(device)

            # Forward pass
            output = model(sents, masks, segs, idxes_mask)

            # Filter affixes
            output_afx = torch.index_select(output, -1, idxes_afx)

            # Rank predictions
            vals_afx, preds_afx = torch.topk(output_afx, k=output_afx.size(-1), dim=-1)

            labels = labels.to(device)

            # Store labels and predictions
            y_true.extend([l.item() for l in labels])
            y_pred.extend([[idxes_afx[p].item() for p in list(l)] for l in preds_afx])

            # Delete tensors to free memory
            del sents, masks, segs, idxes_mask, labels, output

    if save_preds:
        with open(save_preds_path, "a+") as f:
            f.write(f"y_true,y_pred\n")
            for p, b in zip(y_true, y_pred):
                f.write(f"{p},{b[0]}\n")

    return mrr_micro(y_true, y_pred, 10), mrr_macro(y_true, y_pred, 10)


def test_both(test_loader, model, cuda):

    tok = BertTokenizer.from_pretrained("dbmdz/bert-base-german-uncased")

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize prefix list
    pfxes = []
    with open('/netscratch/kenter/lingu_data/prefixes.txt', encoding="latin-1") as f:
        for l in f:
            if l.strip() == '':
                continue
            pfxes.append(l.strip().lower())

    # Initialize suffix list
    sfxes = []
    with open('/netscratch/kenter/lingu_data/suffixes.txt', encoding="latin-1") as f:
        for l in f:
            if l.strip() == '' or l.strip() == 'abil':
                continue
            sfxes.append(l.strip().lower())

    idxes_pfx, _ = torch.sort(torch.tensor(tok.convert_tokens_to_ids(pfxes)).to(device))
    idxes_sfx, _ = torch.sort(torch.tensor(tok.convert_tokens_to_ids(['##' + s for s in sfxes])).to(device))

    print('Evaluating model...')

    y_true = list()
    y_pred = list()

    model.eval()

    with torch.no_grad():

        for batch in test_loader:

            sents, masks, segs, idxes_mask, labels = batch

            sents, masks, segs, idxes_mask = sents.to(device), masks.to(device), segs.to(device), idxes_mask.to(device)

            # Forward pass
            output = model(sents, masks, segs, idxes_mask)

            # Filter affixes and convert to log probabilities
            output_pfx = softmax(torch.index_select(output[:, 0, :], -1, idxes_pfx), dim=-1).log()
            output_sfx = softmax(torch.index_select(output[:, 1, :], -1, idxes_sfx), dim=-1).log()

            # Rank predictions
            vals_pfx, preds_pfx = torch.topk(output_pfx, k=output_pfx.size(-1), dim=-1)
            vals_sfx, preds_sfx = torch.topk(output_sfx, k=output_sfx.size(-1), dim=-1)

            pfx_scores = [[(idxes_pfx[p].item(), v.item()) for p, v in zip(list(preds), list(vals))] for preds, vals in zip(preds_pfx, vals_pfx)]
            sfx_scores = [[(idxes_sfx[p].item(), v.item()) for p, v in zip(list(preds), list(vals))] for preds, vals in zip(preds_sfx, vals_sfx)]

            labels = labels.to(device)

            # Store labels and predictions
            y_true.extend([tuple(v.item() for v in r) for r in labels])

            for pfx_s, sfx_s in zip(pfx_scores, sfx_scores):

                bundle_scores = list()

                for pfx, s_i in pfx_s:
                    for sfx, s_j in sfx_s:
                        bundle_scores.append(((pfx, sfx), s_i + s_j))

                # Extract bundle predictions
                y_pred.append([t[0] for t in sorted(bundle_scores, key=lambda x: x[1], reverse=True)])

            # Delete tensors to free memory
            del sents, masks, segs, idxes_mask, labels, output

    return mrr_micro(y_true, y_pred, 10), mrr_macro(y_true, y_pred, 10)


def main():

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    parser = argparse.ArgumentParser()

    parser.add_argument('--count', default=None, type=int, required=True, help='Frequency bin.')
    parser.add_argument('--lexicon', default=None, type=str, required=True, help='Lexicon setting. shared/split')
    parser.add_argument('--mode', default=None, type=str, required=True, help='Affix type. suffix/prefix/both')
    parser.add_argument('--basemode', default=None, type=str, required=True, help='Base mode type. stem/base')
    parser.add_argument('--batch_size', default=None, type=int, required=True, help='Batch size.')
    parser.add_argument('--lr', default=None, type=float, required=True, help='Learning rate.')
    parser.add_argument('--n_epochs', default=None, type=int, required=True, help='Number of epochs.')
    parser.add_argument('--cuda', default=None, type=int, required=True, help='Selected CUDA.')
    parser.add_argument('--freeze', default=False, action='store_true', help='Freeze BERT parameters.')

    args = parser.parse_args()

    print('Mode: {}'.format(args.mode))
    print('Basemode: {}'.format(args.basemode))
    print('Lexicon setting: {}'.format(args.lexicon))
    print('Count: {}'.format(args.count))
    print('Batch size: {}'.format(args.batch_size))
    print('Learning rate: {}'.format(args.lr))
    print('Number of epochs: {}'.format(args.n_epochs))
    print('Freeze: {}'.format(args.freeze))

    # Define poath to data
    inpath = '/netscratch/kenter/train_data'

    if args.lexicon == "split":
        if args.basemode == "baseplus":
            train_path = f"{args.mode}_{args.count}_{args.lexicon}_base_train.xlsx"
            val_path = f"{args.mode}_{args.count}_{args.lexicon}_base_test.xlsx"
        else:
            train_path = f"{args.mode}_{args.count}_{args.lexicon}_{args.basemode}_train.xlsx"
            val_path = f"{args.mode}_{args.count}_{args.lexicon}_{args.basemode}_test.xlsx"
    else:
        train_path = f"{args.mode}_{args.count}_{args.lexicon}_train.xlsx"
        val_path = f"{args.mode}_{args.count}_{args.lexicon}_test.xlsx"
    train_path = os.path.join(inpath, train_path)
    val_path = os.path.join(inpath, val_path)

    # Initialize train loader
    print('Load training data...')
    train_data = AffixDataset(train_path, args.mode, args.basemode)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_sents)

    # Initialize val loader
    print('Load validation data...')
    val_data = AffixDataset(val_path, args.mode, args.basemode)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, collate_fn=collate_sents)

    print(train_data[101])
    # Initialize model
    print("Initializing the AffixPredictor")
    affix_predictor = AffixPredictor(args.mode, freeze=args.freeze)

    print("Starting training!!")
    train(train_loader, val_loader, affix_predictor, args.mode, args.lr, args.n_epochs, args.cuda, args.count, args.lexicon, args.freeze, args.basemode)


if __name__ == '__main__':
    main()
