import argparse
import os
import pickle
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import torch
import numpy as np
from .data_helpers import AffixDataset, collate_sents
from .evaluation import *
import random
from .model import AffixPredictor
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from finetuning import test_single, test_both


def main():

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', default=None, type=int, required=True, help='Batch size.')

    args = parser.parse_args()

    best_models = [
        ('pfx', 1, 'shared', '1e-04', "stem"), #DONE
        ('pfx', 2, 'shared', '3e-03', "stem"),
        ('pfx', 3, 'shared', '3e-03', "stem"),
        ('pfx', 4, 'shared', '3e-03', "stem"),
        ('pfx', 5, 'shared', '3e-03', "stem"),
        ('pfx', 6, 'shared', '3e-03', "stem"),
        ('pfx', 7, 'shared', '3e-03', "stem"),
        ('pfx', 8, 'shared', '1e-03', "stem"),

        ('pfx', 1, 'split', '1e-04', "stem"), #DONE
        ('pfx', 2, 'split', '1e-04', "stem"),
        ('pfx', 3, 'split', '1e-04', "stem"),
        ('pfx', 4, 'split', '1e-04', "stem"),
        ('pfx', 5, 'split', '3e-03', "stem"),
        ('pfx', 6, 'split', '3e-04', "stem"),
        ('pfx', 7, 'split', '1e-04', "stem"),
        ('pfx', 8, 'split', '1e-04', "stem"),

        ('sfx', 1, 'shared', '3e-04', "stem"), #DONE
        ('sfx', 2, 'shared', '3e-04', "stem"),
        ('sfx', 3, 'shared', '1e-03', "stem"),
        ('sfx', 4, 'shared', '3e-03', "stem"),
        ('sfx', 5, 'shared', '3e-03', "stem"),
        ('sfx', 6, 'shared', '1e-03', "stem"),
        ('sfx', 7, 'shared', '1e-03', "stem"),
        ('sfx', 8, 'shared', '1e-03', "stem"),

        ('sfx', 1, 'split', '1e-04', "stem"), #DONE
        ('sfx', 2, 'split', '1e-04', "stem"),
        ('sfx', 3, 'split', '1e-04', "stem"),
        ('sfx', 4, 'split', '1e-04', "stem"),
        ('sfx', 5, 'split', '1e-04', "stem"),
        ('sfx', 6, 'split', '1e-04', "stem"),
        ('sfx', 7, 'split', '1e-04', "stem"),
        ('sfx', 8, 'split', '1e-04', "stem"),

        ('both', 1, 'shared', '3e-04', "stem"), #DONE
        ('both', 2, 'shared', '3e-04', "stem"),
        ('both', 3, 'shared', '3e-03', "stem"),
        ('both', 4, 'shared', '3e-03', "stem"),
        ('both', 5, 'shared', '3e-03', "stem"),
        ('both', 6, 'shared', '3e-03', "stem"),
        ('both', 7, 'shared', '3e-03', "stem"),
        ('both', 8, 'shared', '3e-03', "stem"),

        ('both', 1, 'split', '1e-04', "stem"), #DONE
        ('both', 2, 'split', '1e-04', "stem"),
        ('both', 3, 'split', '1e-04', "stem"),
        ('both', 4, 'split', '1e-04', "stem"),
        ('both', 5, 'split', '1e-04', "stem"),
        ('both', 6, 'split', '1e-04', "stem"),
        ('both', 7, 'split', '1e-04', "stem"),
        ('both', 8, 'split', '3e-03', "stem"),

        ('pfx', 1, 'shared', '1e-04', "baseplus"), #DONE
        ('pfx', 2, 'shared', '1e-04', "baseplus"),
        ('pfx', 3, 'shared', '1e-03', "baseplus"),
        ('pfx', 4, 'shared', '3e-03', "baseplus"),
        ('pfx', 5, 'shared', '3e-03', "baseplus"),
        ('pfx', 6, 'shared', '3e-03', "baseplus"),
        ('pfx', 7, 'shared', '3e-03', "baseplus"),
        ('pfx', 8, 'shared', '1e-03', "baseplus"),

        ('pfx', 1, 'split', '1e-04', "baseplus"), #DONE
        ('pfx', 2, 'split', '1e-04', "baseplus"),
        ('pfx', 3, 'split', '1e-04', "baseplus"),
        ('pfx', 4, 'split', '1e-04', "baseplus"),
        ('pfx', 5, 'split', '1e-04', "baseplus"),
        ('pfx', 6, 'split', '1e-04', "baseplus"),
        ('pfx', 7, 'split', '1e-04', "baseplus"),
        ('pfx', 8, 'split', '1e-04', "baseplus"),

        ('sfx', 1, 'shared', '1e-03', "base"), #DONE
        ('sfx', 2, 'shared', '1e-03', "base"),
        ('sfx', 3, 'shared', '1e-03', "base"),
        ('sfx', 4, 'shared', '1e-03', "base"),
        ('sfx', 5, 'shared', '3e-03', "base"),
        ('sfx', 6, 'shared', '3e-03', "base"),
        ('sfx', 7, 'shared', '1e-03', "base"),
        ('sfx', 8, 'shared', '3e-03', "base"),

        ('sfx', 1, 'split', '1e-04', "base"), #DONE
        ('sfx', 2, 'split', '1e-04', "base"),
        ('sfx', 3, 'split', '1e-04', "base"),
        ('sfx', 4, 'split', '1e-04', "base"),
        ('sfx', 5, 'split', '1e-04', "base"),
        ('sfx', 6, 'split', '1e-04', "base"),
        ('sfx', 7, 'split', '1e-04', "base"),
        ('sfx', 8, 'split', '1e-04', "base"),

        ('both', 1, 'shared', '1e-04', "base"), #DONE
        ('both', 2, 'shared', '1e-03', "base"),
        ('both', 3, 'shared', '3e-03', "base"),
        ('both', 4, 'shared', '3e-03', "base"),
        ('both', 5, 'shared', '3e-03', "base"),
        ('both', 6, 'shared', '3e-03', "base"),
        ('both', 7, 'shared', '3e-03', "base"),
        ('both', 8, 'shared', '1e-03', "base"),

        ('both', 1, 'split', '3e-04', "base"), #DONE
        ('both', 2, 'split', '1e-04', "base"),
        ('both', 3, 'split', '1e-04', "base"),
        ('both', 4, 'split', '1e-04', "base"),
        ('both', 5, 'split', '1e-04', "base"),
        ('both', 6, 'split', '1e-04', "base"),
        ('both', 7, 'split', '1e-04', "base"),
        ('both', 8, 'split', '1e-04', "base")
    ]

    for bm in best_models:

        print('Mode: {}'.format(bm[0]))
        print('Count: {}'.format(bm[1]))
        print('Lexicon setting: {}'.format(bm[2]))
        print('Learning rate: {}'.format(bm[3]))

        print('Batch size: {}'.format(args.batch_size))

        # Define path to data
        inpath = '/netscratch/kenter/train_data'
        count_mode = bm[1]
        lexicon = bm[2]
        basemode = bm[4]
        mode = bm[0]
        learning_r = bm[3]

        if lexicon == 'shared':
            test_path = f"{mode}_{count_mode}_{lexicon}_test.xlsx"
            dev_path = f"{mode}_{count_mode}_{lexicon}_dev.xlsx"
        elif lexicon == 'split':
            if basemode == "baseplus":
                test_path = f"{mode}_{count_mode}_{lexicon}_base_test.xlsx"
                dev_path = f"{mode}_{count_mode}_{lexicon}_base_dev.xlsx"
            else:
                test_path = f"{mode}_{count_mode}_{lexicon}_{basemode}_test.xlsx"
                dev_path = f"{mode}_{count_mode}_{lexicon}_{basemode}_dev.xlsx"
        else:
            raise ValueError(f"Wrong shared/split input format, try (shared/split) instead of {lexicon}")
        test_path = os.path.join(inpath, test_path)
        dev_path = os.path.join(inpath, dev_path)


        # Initialize val loader
        print('Load validation data...')
        try:
            test_data = AffixDataset(test_path, mode, basemode)
            dev_data = AffixDataset(dev_path, mode, basemode)
        except FileNotFoundError:
            print(f'Bin not found: {mode}, {basemode}, {lexicon}: {test_path} or {dev_path}')
            continue

        test_loader = DataLoader(test_data, batch_size=args.batch_size, collate_fn=collate_sents)
        dev_loader = DataLoader(dev_data, batch_size=args.batch_size, collate_fn=collate_sents)

        #tok = BertTokenizer.from_pretrained('dbmdz/bert-base-german-uncased')

        # Define device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize model
        affix_predictor = AffixPredictor(mode, freeze=True)

        model_path = '/netscratch/kenter/trained_models/model_bert_freeze_{}_{}_{}_{:02d}_{}.torch'.format(lexicon, mode, basemode, count_mode, learning_r)
        # Load finetuned model weights
        print(f'Loading finetuned model weights from {model_path}')
        affix_predictor.load_state_dict(torch.load(model_path, map_location=device))

        # Move model to CUDA
        affix_predictor = affix_predictor.to(device)

        if bm[0] == 'pfx' or bm[0] == 'sfx':
            mrr_micro_t, mrr_macro_dict_t = test_single(test_loader, affix_predictor, mode, 0)
            mrr_micro_d, mrr_macro_dict_d = test_single(dev_loader, affix_predictor, mode, 0)
        elif bm[0] == 'both':
            mrr_micro_t, mrr_macro_dict_t = test_both(test_loader, affix_predictor, 0)
            mrr_micro_d, mrr_macro_dict_d = test_both(dev_loader, affix_predictor, 0)
        else:
            raise ValueError("Wrong mode choosen.")


        mean_mmr_macro_t = np.mean(list(mrr_macro_dict_t.values()))
        mean_mmr_macro_d = np.mean(list(mrr_macro_dict_d.values()))
        overall_mrr_macro = mean_mmr_macro_d*0.25 + mean_mmr_macro_t*0.75
        print(f"Mean MRR macro test: {mean_mmr_macro_t}, dev: {mean_mmr_macro_d}, mean: {overall_mrr_macro}.")
        print(f"MRR micro test: {mrr_micro_t}, dev: {mrr_micro_d}.")
        with open('/netscratch/kenter/results_final/results_bert_freeze_{}_{}_{}_finetuned_macro.txt'.format(lexicon, mode, basemode), 'a+') as f:
            f.write('{:.3f} & '.format(overall_mrr_macro))
        with open('/netscratch/kenter/results_final/results_bert_freeze_{}_{}_{}_finetuned_micro.txt'.format(lexicon, mode, basemode), 'a+') as f:
            f.write('{:.3f} & '.format(mrr_micro_d))


if __name__ == '__main__':
    main()
