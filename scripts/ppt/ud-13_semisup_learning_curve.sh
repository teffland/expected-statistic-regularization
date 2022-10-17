export wandb_tag="UD-13 Semisup Real Learning Curve"
export wandb_note="Train semi-supervised. Use the real stats."
export treebank=$1
export log_name=$2
export config_path=config/ppt-unsupervised.jsonnet

train_fs=(data/ud-treebanks-v2.8/$treebank/*train.conllu)
train_f="${train_fs[0]}"
export train_data_path=$train_f
export test_data_path=${train_f%train.conllu}test.conllu
export vocab_path=data/vocab/udify-13_vocabulary

source ~//scripts/experiment_setup.sh
set_base_hps
export train_data_limit=10000
export learning_rate=0.000002

declare -a sizes=( 50 100 500 1000 )
declare -a seeds=( 0 1 2 )
for size in "${sizes[@]}"
do
  for seed in "${seeds[@]}"
  do
    if [ "$size" -lt 100 ]
    then
        export validation_data_path=${train_f%train.conllu}dev_S${size}_R${seed}.conllu
    else
        export validation_data_path=${train_f%train.conllu}dev_S100_R${seed}.conllu
    fi
    export weights_file_path=logs/21-12-12_ud-13_learning_curve/${treebank}_ud-13_sup_S${size}_R${seed}/best.th
    
    export wandb_name="${treebank}_ud-13_ft-ppt_S${size}_R${seed}"
    run_train
    eval_dev
    eval_test
    sync_results
    sync_model
    cleanup_weights
done
done
echo "DONE MIXED"