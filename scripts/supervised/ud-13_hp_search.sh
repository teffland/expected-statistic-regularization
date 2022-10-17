export wandb_tag="UD-13 Supervsed HP search"
export wandb_note="Small LR grid search"
export treebank=$1
export log_name=$2
export config_path=config/supervised.jsonnet
train_fs=(data/ud-treebanks-v2.8/$treebank/*train.conllu)
train_f="${train_fs[0]}"
export train_data_path=${train_f}
export validation_data_path=${train_f%train.conllu}dev.conllu
export vocab_path=data/vocab/udify-13_vocabulary
export weights_file_path=data/weights/udify-13-model_weights.th

# Run a small HP grid search on the full size
source ~//scripts/experiment_setup.sh
set_base_hps
export num_batches_per_epoch=200
export num_epochs=100
export scheduler_num_epochs=$num_epochs
export patience=10
declare -a lrs=( 0.00007 )
#0.00002 0.000007 0.000002 0.0000007 )
for lr in "${lrs[@]}"
do
  export wandb_name="${treebank}_ud-13_sup_lr-${lr}_ne-100"
  export learning_rate=$lr
  run_train
  eval_dev
  sync_results
  sync_model
  cleanup_weights
done
echo "DONE"