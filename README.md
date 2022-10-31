## Replication
requirements: 
- python 3.8.5  
- R 4.1.1

WIP: data, surprisal files, model files

`pip install requirements.txt`  
`python preprocess/BE/get_sents.py > data/BE/sents.txt`  
`cat data/BE/sents.txt | mecab -d /opt/local/lib/mecab/dic/unidic > data/BE/morph.tsv`  
`python preprocess/BE/add_annotation.py`  
`python preprocess/BE/filter.py`  
`python preprocess/BE/data_points4modeling.py`  
`bash preprocess/BE/modify_context.sh`

`python preprocess/DC/add_annotation.py`  
`python preprocess/DC/filter.py`  
`python preprocess/DC/data_points4modeling.py`  
`bash preprocess/DC/modify_context.sh`

`bash experiments/ja_surprisal.sh`  
`python experiments/convert_scores.py --dir surprisals/BE --corpus BCCWJ`  
`Rscript experiments/bccwj.r surprisals/BE/`  
`python experiments/aggregate.py --dir surprisals/BE --file likelihood.txt > surprisals/BE/aggregated.txt`

`bash experiments/ja_surprisal_vanilla.sh`  
`python experiments/convert_scores.py --dir surprisals/BE-vanilla --corpus BCCWJ`  
`Rscript experiments/bccwj.r surprisals/BE-vanilla/`  
`python experiments/aggregate.py --dir surprisals/BE-vanilla --file likelihood.txt > surprisals/BE-vanilla/aggregated.txt`

`bash experiments/en_surprisal.sh`  
`python experiments/convert_scores.py --dir surprisals/DC --corpus dundee`  
`Rscript experiments/dundee.r surprisals/DC/`  
`python experiments/aggregate.py --dir surprisals/DC --file likelihood.txt > surprisals/DC/aggregated.txt`

`bash experiments/en_surprisal.sh`  
`python experiments/convert_scores.py --dir surprisals/DC-vanilla --corpus dundee`  
`Rscript experiments/dundee.r surprisals/DC-vanilla/`  
`python experiments/aggregate.py --dir surprisals/DC-vanilla --file likelihood.txt > surprisals/DC-vanilla/aggregated.txt`

`bash experiments/en_surprisal_hf.sh`  
`python experiments/convert_scores.py --dir surprisals/DC-hf --corpus dundee`  
`Rscript experiments/dundee.r surprisals/DC-hf/`  
`python experiments/aggregate.py --dir surprisals/DC-hf --file likelihood.txt > surprisals/DC-hf/aggregated.txt`

Run `experiments/visualize.ipynb`