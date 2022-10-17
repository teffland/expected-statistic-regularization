export wandb_tag="Oracle Stat UD-13 Loss Ablation"
export wandb_note="Train model unsupervised with oracle stats on UD 13 with different loss types"
export treebank=$1
export log_name=$2
export config_path=config/unsupervised.jsonnet

train_fs=(data/ud-treebanks-v2.8/$treebank/*train.conllu)
train_f="${train_fs[0]}"
export train_data_path=$train_f
export validation_data_path=${train_f%train.conllu}dev.conllu
export vocab_path=data/vocab/udify-13_vocabulary
export weights_file_path=data/weights/udify-13-model_weights.th
export stats_csv_path=${train_f%.conllu}_gold_udify-13_stats.csv

source ~//scripts/experiment_setup.sh
set_base_hps
export head_tail_dep_dir_weight=1.0

# L1 hard-margin
export loss_type="l1-margin"
export wandb_name="${treebank}_ud-13_oracle_crd_l1-margin"
run_train
eval_dev
eval_test
# sync_results
cleanup_weights


# L1
export loss_type="l1"
export loss_margin=0
export wandb_name="${treebank}_ud-13_oracle_crd_l1"
run_train
eval_dev
eval_test
# sync_results
cleanup_weights


# L2
export loss_type="l2"
export loss_margin=0
export wandb_name="${treebank}_ud-13_oracle_crd_l2"
run_train
eval_dev
eval_test
# sync_results
cleanup_weights


done
echo "DONE"