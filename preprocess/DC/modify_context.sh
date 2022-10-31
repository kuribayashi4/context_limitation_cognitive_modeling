for n in "20" "10" "7" "5" "3" "2"
do
    python preprocess/DC/modify_context.py -n ${n} --context-func delete
    for slope in "0.5" "0.25" "0.125" "0.0625"
    do
        python preprocess/DC/modify_context.py -n ${n} --context-func lossy --lossy-slope ${slope}
    done
done
python preprocess/DC/modify_context.py -n 1000 --context-func delete