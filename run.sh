#!/usr/bin/bash

rm *.zip.*
mkdir -p lexica
mkdir -p analysis
mkdir -p data

# Download and unpack the Billion Words for pre-training
wget http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz
tar -zxvf 1-billion-word-language-modeling-benchmark-r13output.tar.gz
python code/sort_billion.py
cat data/bwb* > data/bwb_small.txt

# Download and unpack the CoLA files.
wget https://nyu-mll.github.io/CoLA/cola_public_1.1.zip
unzip cola_public_1.1.zip

# Pre-train the model on BWB
ccglink train --corpus data/bwb_small.txt --format text -s 50 \
  --lexicon lexica/bwb_small.lex

# Fine-tune on CoLA.
ccglink train --corpus cola_public/tokenized/in_domain_train.tsv -s 50 \
  --initial lexica/bwb_small.lex --format cola
  --lexicon lexica/in_domain_train.lex

# Create training scores from CoLA.
ccglink evaluate --corpus cola_public/tokenized/in_domain_train.tsv -s 50 \
  --format cola --lexicon lexica/in_domain_train.lex \
  > analysis/in_domain_scores.tsv

# Validate on CoLA.
ccglink evaluate --corpus cola_public/tokenized/out_of_domain_dev.tsv -s 50 \
  -f cola --lexicon lexica/in_domain_train.lex > analysis/out_of_domain_scores.tsv

# Evaluate the regression
python code/regression_test.py
