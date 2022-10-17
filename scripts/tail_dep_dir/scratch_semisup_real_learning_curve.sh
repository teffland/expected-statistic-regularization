export wandb_tag="Scratch Semisup Real Learning Curve"
export wandb_note="Train semi-supervised from scratch. Use the real stats."
export treebank=$1
export log_name=$2

train_fs=(data/ud-treebanks-v2.8/$treebank/*train.conllu)
train_f="${train_fs[0]}"
export train_data_path=$train_f
export test_data_path=${train_f%train.conllu}test.conllu
export vocab_path=data/vocab/udify-13_vocabulary
export config_path=config/semi-supervised_scratch.jsonnet

source ~//scripts/experiment_setup.sh
set_base_hps
export tail_dep_dir_weight=1.0
export num_epochs=40
export scheduler_num_epochs=$num_epochs
export learning_rate=0.00002
export unsup_loss_weight=0.01
export num_unsup_per_sup=4
export weights_file_path=data/weights/udify-13-model_weights.th
export unsupervised_train_data_path=${train_f}

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
    export supervised_train_data_path=${train_f%.conllu}_S${size}_R${seed}.conllu
    export stats_csv_path=${train_f%.conllu}_S${size}_R${seed}_udify-13_stats.csv
    
    export wandb_name="${treebank}_scratch_real_stat-trd_S${size}_R${seed}"
    run_train
    eval_dev
    eval_test
    sync_results
    sync_model
    cleanup_weights
done
done
echo "DONE MIXED"