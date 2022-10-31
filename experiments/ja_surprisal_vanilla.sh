batch=50
for ngram in  "20" "10" "7" "5" "3" "2" "1000"
do
    for func in "delete" "lossy-0.5" "lossy-0.25" "lossy-0.125" "lossy-0.0625"
    do
        for arc in "lstm" "transformer" "gpt"
        do
            for seed in "seed-1" "seed-39" "seed-46"
            do                        
                echo ${func}
                echo ${arc}
                echo ${ngram}
                CUDA_VISIBLE_DEVICES=0 python experiments/calc_surprisal.py -m models/ja_lm_vanilla/${arc}/${seed}/checkpoint.pt -o surprisals/BE-vanilla/arch_${arc}-ngram_${ngram}-contextfunc_${func}/${seed} -a ${arc} --batchsize ${batch} -d data/BE/ngram_${ngram}-contextfunc_${func}.json
            done
        done
    done
done



