export wandb_tag="UD-13 Supervsed Learning curve"
export wandb_note="Varying amount of supervised data"
export treebank=$1
export log_name=$2
export config_path=config/supervised.jsonnet

train_fs=(data/ud-treebanks-v2.8/$treebank/*train.conllu)
train_f="${train_fs[0]}"
export test_data_path=${train_f%train.conllu}test.conllu
export vocab_path=data/vocab/udify-13_vocabulary
export weights_file_path=data/weights/udify-13-model_weights.th


source ~//scripts/experiment_setup.sh
set_base_hps
export num_batches_per_epoch=200
export num_epochs=100
export scheduler_num_epochs=$num_epochs
export patience=10
export learning_rate=0.00002
declare -a sizes=( 50 100 500 1000 )
declare -a seeds=( 0 1 2 )
for size in "${sizes[@]}"
do
  for seed in "${seeds[@]}"
  do

    export train_data_path=${train_f%.conllu}_S${size}_R${seed}.conllu
    # Dev size = min(train size, 100) examples
    if [ "$size" -lt 100 ]
    then
        export validation_data_path=${train_f%train.conllu}dev_S${size}_R${seed}.conllu
    else
        export validation_data_path=${train_f%train.conllu}dev_S100_R${seed}.conllu
    fi
    
    export wandb_name="${treebank}_ud-13_sup_S${size}_R${seed}"
    run_train
    eval_dev
    make_archive
    eval_test
    sync_results
    cleanup_weights
done
done
echo "DONE"