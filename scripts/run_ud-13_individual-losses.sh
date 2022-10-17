# Run an hp search with the ud-13 model and data
cd ~/
source ./scripts/set_base_hps.sh
tb=$1

# Set metadata and various log/output paths
export wandb_tag="Oracle Stat smoothl1 UD-13"
export wandb_note="Test out a specific stat function on a dataset"
log_dir=21-11-14_ud-13_unsup_indiv_smoothl1_mean/${tb}
log_path=logs/$log_dir
run_log_dir=logs/run_logs/${log_dir}
mkdir -p $log_path $run_log_dir

# Set the config hps
train_fs=(data/ud-treebanks-v2.8/$tb/*train.conllu)
train_f="${train_fs[0]}"
export train_data_path=$train_f
export validation_data_path=${train_f%train.conllu}dev.conllu
export vocab_path=data/vocab/udify-13_vocabulary
export weights_file_path=data/weights/udify-13-model_weights.th
export stats_csv_path=${train_f%.conllu}_gold_udify-13_stats.csv

setup() {
    set_base_hps
    export loss_target="mean"
    export loss_margin="stddev"
    export validation_metric="+.run/.sum"
}

run() {
    export config_path=config/unsupervised.jsonnet
    export run_log_path=${run_log_dir}/${wandb_name}.out
    export out_dir=${log_path}/${wandb_name}

    echo
    echo "Oracle unsupervised hp run for ${tb}"
    echo "  has name: ${wandb_name}"
    echo "  using config: ${config_path}"
    echo "  train output at ${out_dir}"
    echo "  train log at ${run_log_path}"

    /home/ubuntu/anaconda3/envs/env2/bin/allennlp train ${config_path}\
    -f -s "${out_dir}"\
    --include ml --file-friendly-logging\
    &> ${run_log_path}

    echo "Finished run at ${run_log_path}"
    echo "Evaluating ${out_dir}/model.tar.gz on dev data for ${treebank_name}"
    /home/ubuntu/anaconda3/envs/env2/bin/python predict.py\
    ${out_dir}/model.tar.gz\
    $validation_data_path\
    ${out_dir}/dev_preds.conllu\
    --eval_file ${out_dir}/dev_results.json
    echo DONE
}


# Initial udify baseline. Improvements are measure relative to this
setup
export wandb_name="${tb}_ud-13_baseline"
export num_epochs=1
export num_batches_per_epoch=1
export learning_rate=0.0
export weight_decay=0.0
run


base_name="${tb}_oracle"

setup
export pos_weight=1.0
export wandb_name="${base_name}_stat=pos"
run

setup
export dep_dir_weight=1.0
export wandb_name="${base_name}_stat=dep_dir"
run

setup
export head_dir_weight=1.0
export wandb_name="${base_name}_stat=head_dir"
run

setup
export tail_dir_weight=1.0
export wandb_name="${base_name}_stat=tail_dir"
run

setup
export pos_pair_weight=1.0
export wandb_name="${base_name}_stat=pos_pair"
run

setup
export pos_pair_given_pos_weight=1.0
export wandb_name="${base_name}_stat=pos_pair_given_pos"
run

setup
export dep_dir_dist_weight=1.0
export wandb_name="${base_name}_stat=dep_dir_dist"
run

setup
export dep_dir_dist_given_dep_weight=1.0
export wandb_name="${base_name}_stat=dep_dir_dist_given_dep"
run

setup
export pos_valency_weight=1.0
export wandb_name="${base_name}_stat=pos_valency"
run

setup
export pos_valency_given_pos_weight=1.0
export wandb_name="${base_name}_stat=pos_valency_given_pos"
run

setup
export head_tail_dir_weight=1.0
export wandb_name="${base_name}_stat=head_tail_dir"
run

setup
export head_tail_dir_given_head_weight=1.0
export wandb_name="${base_name}_stat=head_tail_dir_given_head"
run

setup
export head_tail_dir_given_tail_weight=1.0
export wandb_name="${base_name}_stat=head_tail_dir_given_tail"
run

setup
export head_dep_dir_weight=1.0
export wandb_name="${base_name}_stat=head_dep_dir"
run

setup
export head_dep_dir_given_dep_weight=1.0
export wandb_name="${base_name}_stat=head_dep_dir_given_dep"
run

setup
export head_dep_dir_given_head_weight=1.0
export wandb_name="${base_name}_stat=head_dep_dir_given_head"
run

setup
export tail_dep_dir_weight=1.0
export wandb_name="${base_name}_stat=tail_dep_dir"
run

setup
export tail_dep_dir_given_dep_weight=1.0
export wandb_name="${base_name}_stat=tail_dep_dir_given_dep"
run

setup
export tail_dep_dir_given_tail_weight=1.0
export wandb_name="${base_name}_stat=tail_dep_dir_given_tail"
run

setup
export head_tail_dep_dir_weight=1.0
export wandb_name="${base_name}_stat=head_tail_dep_dir"
run

setup
export head_tail_dep_dir_given_head_weight=1.0
export wandb_name="${base_name}_stat=head_tail_dep_dir_given_head"
run

setup
export head_tail_dep_dir_given_tail_weight=1.0
export wandb_name="${base_name}_stat=head_tail_dep_dir_given_tail"
run

setup
export head_tail_dep_dir_given_dep_weight=1.0
export wandb_name="${base_name}_stat=head_tail_dep_dir_given_dep"
run

These need to be redone with corrected stats
export stats_csv_path=${train_f%.conllu}_gold_udify-13_stats_fixed-sibs.csv
setup
export head_tail1_tail2_dir1_dir2_weight=1.0
export wandb_name="${base_name}_stat=head_tail1_tail2_dir1_dir2"
run

setup
export head_tail1_tail2_dir1_dir2_given_head_weight=1.0
export wandb_name="${base_name}_stat=head_tail1_tail2_dir1_dir2_given_head"
run

setup
export head_dep1_dep2_dir1_dir2_weight=1.0
export wandb_name="${base_name}_stat=head_dep1_dep2_dir1_dir2"
run

setup
export head_dep1_dep2_dir1_dir2_given_head_weight=1.0
export wandb_name="${base_name}_stat=head_dep1_dep2_dir1_dir2_given_head"
run

export stats_csv_path=${train_f%.conllu}_gold_udify-13_stats.csv
setup
export head_tail_gtail_dir_gdir_weight=1.0
export wandb_name="${base_name}_stat=head_tail_gtail_dir_gdir"
run

setup
export head_tail_gtail_dir_gdir_given_tail_weight=1.0
export wandb_name="${base_name}_stat=head_tail_gtail_dir_gdir_given_tail"
run

setup
export tail_dep_gdep_dir_gdir_weight=1.0
export wandb_name="${base_name}_stat=tail_dep_gdep_dir_gdir"
run

setup
export tail_dep_gdep_dir_gdir_given_tail_weight=1.0
export wandb_name="${base_name}_stat=tail_dep_gdep_dir_gdir_given_tail"
run

echo ALL DONE
