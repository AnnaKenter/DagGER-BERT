# DagGER-BERT

Code for finetuning DagGER-BERT, a BERT variant trained to predict derivationally complex morphology

This code and model are modified to the German language and our own experiments from [DagoBERT](https://github.com/valentinhofmann/dagobert)
and the publication [DagoBERT: Generating Derivational Morphology
with a Pretrained Language Model](https://aclanthology.org/2020.emnlp-main.316.pdf).

## Setup

The code requires `Python>=3.6`, `numpy>=1.18`, `torch>=1.2`, and `transformers>=2.5`.

The scripts expect the full dataset in `data/final/`, use [this library](https://github.com/AnnaKenter/DerivativeCorpusGerman)
to extract the dataset and formatting of the data.

## Start finetunig
Execute the script 'finetuning.py'