export wandb_tag="Oracle Stat smoothl1 UD-13"
export wandb_note="Train model unsupervised with oracle stats"
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
export learning_rate=0.000002
export train_data_limit=10000
export wandb_name="${treebank}_ud-13_ppt"
# NO NEED, the best HP search run corresponds to this experiment

# run_train
# eval_dev
# sync_results
# cleanup_weights
done
echo "DONE"