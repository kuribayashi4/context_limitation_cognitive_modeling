## Paper information
Context Limitations Make Neural Language Models More Human-Like (EMNLP2022)  
paper: https://aclanthology.org/2022.emnlp-main.712/
```
@inproceedings{kuribayashi-etal-2022-context,
    title = "Context Limitations Make Neural Language Models More Human-Like",
    author = "Kuribayashi, Tatsuki  and
      Oseki, Yohei  and
      Brassard, Ana  and
      Inui, Kentaro",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.712",
    pages = "10421--10436",
    abstract = "Language models (LMs) have been used in cognitive modeling as well as engineering studies{---}they compute information-theoretic complexity metrics that simulate humans{'} cognitive load during reading.This study highlights a limitation of modern neural LMs as the model of choice for this purpose: there is a discrepancy between their context access capacities and that of humans.Our results showed that constraining the LMs{'} context access improved their simulation of human reading behavior.We also showed that LM-human gaps in context access were associated with specific syntactic constructions; incorporating syntactic biases into LMs{'} context access might enhance their cognitive plausibility.",
}
```

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
We would provide the links to download them (due to server load considerations, the link is not publicly available).

## Notes on the implementation details
In our experiments using HuggingFace (HF) and vanilla LMs, we input the  $\mathrm{ < s > }$ token only when the context is empty (i.e., when $i = 0$) as per Eq. (1) in our paper.

$$
I_\mathrm{lossy}(w_i,c_{ < i})= -\log p_{\theta}(w_{i}| \mathrm{ < s > } \circ f([w_0, \cdots, w_{i-1}])) \space \space \space (1)
$$

This is because in the HF and vanilla settings, the  $\mathrm{ < s > }$  token has a special meaning as the beginning of a sentence/document. Concatenating  $\mathrm{ < s > }$  with a context starting from the middle of a sentence (e.g., $[w_5, w_6]$ ) can cause confusion for the model as if a sentence begins with $w_5$.  
It's important to note this implementation choice to replicate our results, as we have received feedback on this. Thank you!
