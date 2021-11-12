# Event-Driven Query Expansion

This repository provides the data and implementation of the paper:
>Event-Driven Query Expansion<br>
>Guy D. Rosin, Ido Guy, and Kira Radinsky<br>
>WSDM 2021<br>
>Preprint: https://arxiv.org/abs/2012.12065

## Prerequisites

- Install [Terrier](http://terrier.org/download/).
- Obtain a [TREC dataset](https://trec.nist.gov/data.html) (e.g., [Robust04](https://trec.nist.gov/data/t13_robust.html)) and index it using Terrier.
- Download the [Wikipedia2vec model](http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_100d.pkl.bz2) (see the full list of models [here](https://wikipedia2vec.github.io/wikipedia2vec/pretrained/)).
- Obtain a collection of temporal word2vec models (e.g., from the New York Times).

## Usage

- Run `event_projection.py` to enrich the temporal embeddings with events from Wikipedia.
- Run `trec_search.py` to perform retrieval with or without query expansion and evaluate, after setting the relevant parameters (model paths, dataset, QE method, etc.).


## Pretrained Word Embeddings
The word embeddings used in the paper can be downloaded from [here](https://karpef.cs.technion.ac.il/index.php/s/YA3G3iqDZyagnFH/download).

The embeddings were created using [Gensim](https://github.com/RaRe-Technologies/gensim).
Each model is based on data from a single year of the New York Times, and is then enriched with events from Wikipedia.
There are 38 models in total, for the years of 1981-2018.

## Dependencies

- Python 3.7
- trectools (custom version: https://github.com/guyrosin/trectools)
- Terrier 5.1 (http://terrier.org)
- numpy
- scipy
- scikit-learn
- nltk
- gensim 3.8
- pandas
- tqdm
- sqlitedict