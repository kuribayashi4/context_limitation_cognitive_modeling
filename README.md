## Paper infoamtion
Context Limitations Make Neural Language Models More Human-Like (EMNLP2022)  
arXiv: https://arxiv.org/abs/2205.11463

## Replication
requirements: 
- python 3.8.5  
- R 4.1.1
- mecab 0.996
- unidic dictionary [unidic-mecab-211](https://clrd.ninjal.ac.jp/unidic/back_number.html)

Two choices to replicate our results:

1. from scratch (except for LM training); run `run.sh` with
- corpus data (`data/`)
- model files (`models/`; 50GB)  

    In this case, one should have a right to access the Dundee corpus and BCCWJ-EyeTrack corpus.

2. from pre-computed surprisals; run `experiments/visualize.ipynb` with
- surprisal files (`surprisals/`; 25GB)

In either case, please contact `kuribayashi.research [at] gmail.com`  to get the necessary data to replicate.  
We would provide the links to download them (due to server load considerations, the link is not publickly available).

From [here](https://github.com/kuribayashi4/context_limitation_cognitive_modeling/blob/main/experiments/visualize.ipynb), you can also just overview the results.
