batch=10
for ngram in "20" "10" "7" "5" "3" "2" "1000"
do
    for func in "delete" "lossy-0.5" "lossy-0.25" "lossy-0.125" "lossy-0.0625"
    do
        for arc in "gpt2" "gpt2_medium" "gpt2_large" "gpt2_xl"
        do
            echo ${func}
            echo ${arc}
            echo ${ngram}
            CUDA_VISIBLE_DEVICES=1 python experiments/calc_surprisal_hf.py -m ${arc} -o surprisals/DC-hf/arch_${arc}-ngram_${ngram}-contextfunc_${func} --batchsize ${batch} -d data/DC/ngram_${ngram}-contextfunc_${func}.json
        done
    done
done
