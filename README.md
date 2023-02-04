## Paper infoamtion
Context Limitations Make Neural Language Models More Human-Like (EMNLP2022)  
arXiv: https://arxiv.org/abs/2205.11463

## Replication
### View results
From [here](https://github.com/kuribayashi4/context_limitation_cognitive_modeling/blob/main/experiments/visualize.ipynb), you can overview the results.

### From scratch
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

## Notes on the implementation details
In our experiments using HuggingFace (HF) and vanilla LMs, we input the  $\mathrm{ < s > }$ token only when the context is empty (i.e. when $i = 0$) as per Eq. (1) in our paper.

$$
I_\mathrm{lossy}(w_i,c_{ < i})= -\log p_{\theta}(w_{i}| \mathrm{ < s > } \circ f([w_0, \cdots, w_{i-1}])) \space \space \space (1)
$$

This is because in the HF and vanilla settings, the  $\mathrm{ < s > }$  token has a special meaning as the beginning of a sentence/document. Concatenating  $\mathrm{ < s > }$  with a context starting from the middle of a sentence (e.g., $[w_5, w_6]$ ) can cause confusion for the model as if a sentence begins with $w_5$.  
It's important to note this implementation choice to replicate our results, as we have received feedback on this. Thank you!
