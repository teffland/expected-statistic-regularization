# Subsample the UD-13 and UD-test test treebanks to simulate limited supervision


# UD-13 and UD-test eval treebanks
declare -a fs=(
    "UD_German-HDT"
    "UD_Indonesian-GSD"
    "UD_Maltese-MUDT"
    "UD_Persian-PerDT"
    "UD_Vietnamese-VTB"
)

declare -a sizes=( 10 50 100 500 1000 )
declare -a seeds=( 0 1 2 )

for tb in "${fs[@]}"
do
    echo "Treebank ${tb}"
    files=(/home/ubuntu//data/ud-treebanks-v2.8/$tb/*train.conllu)
    train_f="${files[0]}"
    pred_f=${train_f%.conllu}_udify-13_pred.conllu
    dev_f=${train_f%train.conllu}dev.conllu

    for size in "${sizes[@]}"
    do
        for seed in "${seeds[@]}"
        do
            echo "Sampling train: Size:${size} , Seed:${seed}"
            python -m ml.cmd.subsample_dataset $train_f -s $size -r $seed
            python -m ml.cmd.subsample_dataset $pred_f -s $size -r $seed

            # Dev size = min(train size, 100) examples
            if [ "$size" -le 100 ]
            then
                echo "Sampling dev: Size:${size} , Seed:${seed}"
                python -m ml.cmd.subsample_dataset $dev_f -s $size -r $seed
            fi
        done
    done
done
echo "ALL DONE"