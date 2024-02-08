from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class AffixDataset(Dataset):

    def __init__(self, filename, mode, base_mode):

        # Define which affix type to extract
        self.mode = mode
        self.base_mode = base_mode

        # Initialize tokenizer
        self.tok = BertTokenizer.from_pretrained('dbmdz/bert-base-german-uncased')

        self.sents = list()
        self.idxes_mask = list()
        self.labels = list()

        df = pd.read_excel(filename, engine="openpyxl")
        if base_mode == "stem":
            df = df[df["stem"].notna()]
        elif base_mode == "baseplus":
            df = df[df["base"].notna()]
        elif base_mode == "base":
            if mode == "prefix":
                df = df[df["stem"].notna()]
            else:
                df = df[df["base"].notna()]
        else:
            raise ValueError(f"Basemode {base_mode} not valid.")

        for index, row in df.iterrows():

            derivate = row["token1"].lower()
            affix = row["affix"]
            context = row["context"]
            #context_sents = row["context"][2:-2].split("']['")

            #for context in context_sents:
            if type(context) != str:
                print(f"This is wrong context: {context}, skipping.")
                continue
            context = context.lower()
            split_context = context.split(derivate)  # divide context into pre-derivate and post-derivate

            if len(split_context) > 2:  # derivate appears more than once in context sentence
                #print(f"The derivate {derivate} was more than once in context: {context}.")
                continue
            elif len(split_context) < 2 and not context.startswith(derivate):
                print(f"The derivate {derivate} is not in the context: {context}")
                continue

            if context.startswith(derivate):
                s_1 = ""
                s_2 = split_context[0]
            else:
                s_1 = split_context[0]
                s_2 = split_context[1]

            # Prefix (with trick)
            if self.mode == 'pfx' and row["mode"] == "prefix":

                if self.base_mode == "stem":
                    if not row["stem"] or pd.isna(row["stem"]):
                        continue
                    toki = self.tok.tokenize(row["stem"])
                elif self.base_mode == "base":
                    if not row["stem"] or pd.isna(row["stem"]):
                        continue
                    toki = self.tok.tokenize(row["stem"])
                elif self.base_mode == "baseplus":
                    if not row["base"] or pd.isna(row["base"]):
                        continue
                    toki = self.tok.tokenize(row["base"])
                else:
                    raise ValueError(f"Base mode has to be either base, stem or baseplus (not {self.base_mode}).")

                # Tokenize sentence and add mask token
                s = ['[CLS]'] + self.tok.tokenize(s_1) + ['[MASK]', '-'] + toki + self.tok.tokenize(s_2) + ['[SEP]']

                # Store index of mask token
                self.idxes_mask.append(s.index('[MASK]'))

                # Encode sentence
                s = self.tok.convert_tokens_to_ids(s)

                # Store tokenized sentence and label
                self.sents.append(s)
                self.labels.append(self.tok.convert_tokens_to_ids(affix))

            # Suffix
            elif self.mode == 'sfx' and row["mode"] == "suffix":

                if self.base_mode == "stem":
                    if not row["stem"] or pd.isna(row["stem"]):
                        continue
                    toki = self.tok.tokenize(row["stem"])
                elif self.base_mode == "base":
                    if not row["base"] or pd.isna(row["base"]):
                        continue
                    toki = self.tok.tokenize(row["base"])
                elif self.base_mode == "baseplus":
                    if not row["base"] or pd.isna(row["base"]):
                        continue
                    toki = self.tok.tokenize(row["base"])
                else:
                    raise ValueError(f"Base mode has to be either base, stem or baseplus (not {self.base_mode}).")

                # Tokenize sentence and add mask token
                s = ['[CLS]'] + self.tok.tokenize(s_1) + toki + ['[MASK]'] + self.tok.tokenize(s_2) + ['[SEP]']
                # Store index of mask token
                self.idxes_mask.append(s.index('[MASK]'))

                # Encode sentence
                s = self.tok.convert_tokens_to_ids(s)

                # Store tokenized sentence and label
                self.sents.append(s)
                self.labels.append(self.tok.convert_tokens_to_ids('##' + affix))

            # Both
            elif self.mode == 'both' and row["mode"] == "both":

                if self.base_mode == "stem":
                    if not row["stem"] or pd.isna(row["stem"]):
                        continue
                    toki = self.tok.tokenize(row["stem"])
                elif self.base_mode == "base":
                    if not row["base"] or pd.isna(row["base"]):
                        continue
                    toki = self.tok.tokenize(row["base"])
                elif self.base_mode == "baseplus":
                    if not row["base"] or pd.isna(row["base"]):
                        continue
                    toki = self.tok.tokenize(row["base"])
                else:
                    raise ValueError(f"Base mode has to be either base, stem or baseplus (not {self.base_mode}).")

                # Tokenize sentence and add mask tokens
                s = ['[CLS]'] + self.tok.tokenize(s_1) + ['[MASK]', '-'] + toki + ['[MASK]'] + self.tok.tokenize(s_2) + ['[SEP]']

                # Store index of mask token
                self.idxes_mask.append([i for i in range(len(s)) if s[i] == '[MASK]'])

                # Encode sentence
                s = self.tok.convert_tokens_to_ids(s)

                # Store tokenized sentence and label
                self.sents.append(s)
                affixs = affix.split(",")
                prefix = self.tok.convert_tokens_to_ids(affixs[0][2:-1].strip())
                suffix = self.tok.convert_tokens_to_ids('##' + affixs[1][2:-2].strip())

                self.labels.append([prefix, suffix])

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):

        # Select sentence, index if mask token, and label
        s = self.sents[idx]
        idx_mask = self.idxes_mask[idx]
        l = self.labels[idx]

        return s, idx_mask, l


def collate_sents(batch):
    batch_size = len(batch)

    sents = [s for s, idx_mask, l in batch]
    idxes_mask = [idx_mask for s, idx_mask, l in batch]
    labels = [l for s, idx_mask, l in batch]

    # Get maximum sentence length in batch
    max_len = max(len(s) for s in sents)

    sents_pad = np.zeros((batch_size, max_len))
    masks_pad = np.zeros((batch_size, max_len))
    segs_pad = np.zeros((batch_size, max_len))

    for i, s in enumerate(sents):
        sents_pad[i, :len(s)] = s
        masks_pad[i, :len(s)] = 1

    return torch.tensor(sents_pad).long(), torch.tensor(masks_pad).long(), torch.tensor(segs_pad).long(), torch.tensor(idxes_mask).long(), torch.tensor(labels).long()
