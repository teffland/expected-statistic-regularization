export wandb_tag="UD-13 PPT Hp search"
export wandb_note="PPT hp search for learning rate"
export treebank=$1
export log_name=$2
export config_path=config/ppt-unsupervised.jsonnet

train_fs=(data/ud-treebanks-v2.8/$treebank/*train.conllu)
train_f="${train_fs[0]}"
export train_data_path=$train_f
export validation_data_path=${train_f%train.conllu}dev.conllu
export vocab_path=data/vocab/udify-13_vocabulary
export weights_file_path=data/weights/udify-13-model_weights.th
export stats_csv_path=${train_f%.conllu}_gold_udify-13_stats.csv

source ~//scripts/experiment_setup.sh
set_base_hps
export train_data_limit=10000

declare -a lrs=( 0.00007 0.00002 0.000007 0.000002 0.0000007 )
for lr in "${lrs[@]}"
do
    export learning_rate=$lr
    export wandb_name="${treebank}_ud-13_ppt_lr-${lr}"
    run_train
    eval_dev
    sync_results
    cleanup_weights
done
echo "DONE"

# Result: best is 0.000002