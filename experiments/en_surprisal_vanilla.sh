batch=50
for ngram in "20" "10" "7" "5" "3" "2" "1000"
do
    for func in "delete" "lossy-0.5" "lossy-0.25" "lossy-0.125" "lossy-0.0625"
    do
        for arc in "lstm" "gpt" "transformer"
        do
            do
            for seed in "seed-1" "seed-39" "seed-46"
            do
                echo ${func}
                echo ${arc}
                echo ${ngram}
                echo ${splitb}
                CUDA_VISIBLE_DEVICES=1 python experiments/calc_surprisal.py -m models/en_lm_vanilla/${arc}/${seed}/checkpoint_last.pt -o surprisals/DC-vanilla/arch_${arc}-ngram_${ngram}-contextfunc_${func}/${seed} -a ${arc} --batchsize ${batch} --corpus dundee -d data/DC/ngram_${ngram}-contextfunc_${func}.json
            done
        done
    done
done
