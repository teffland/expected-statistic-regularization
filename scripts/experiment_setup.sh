# Base setup for all experiments
cd ~/syntax/

log_path=logs/${log_name}
run_log_dir=logs/run_logs/${log_name}
mkdir -p $log_path $run_log_dir

source scripts/set_base_hps.sh

run_train() {
    run_log_path=${run_log_dir}/${wandb_name}.out
    out_dir=${log_path}/${wandb_name}

    echo
    echo "Run training setup:"
    echo " - Treebank: ${treebank}"
    echo " - Run name: ${wandb_name}"
    echo " - Run tag: ${wandb_tag}"
    echo " - Config: ${config_path}"
    echo " - Train Data: ${train_data_path}"
    echo " - Dev Data: ${validation_data_path}"
    echo " - Archive: ${out_dir}"
    echo " - Log: ${run_log_path}"

     /home/ubuntu/.pyenv/versions/esr/bin/allennlp train ${config_path}\
        -f -s "${out_dir}"\
        --include ml --file-friendly-logging\
        &> ${run_log_path}
    echo "Done training"
}

eval_dev() {
    run_log_path=${run_log_dir}/${wandb_name}_eval_dev.out
    out_dir=${log_path}/${wandb_name}

    echo
    echo "Evaluating dev data:"
    echo " - Treebank: ${treebank}"
    echo " - Run name: ${wandb_name}"
    echo " - Run tag: ${wandb_tag}"
    echo " - Config: ${config_path}"
    echo " - Train Data: ${train_data_path}"
    echo " - Dev Data: ${validation_data_path}"
    echo " - Archive: ${out_dir}"
    echo " - Log: ${run_log_path}"

    /home/ubuntu/.pyenv/versions/esr/bin/python predict.py\
        "${out_dir}/model.tar.gz"\
        "$validation_data_path"\
        "${out_dir}/dev_preds.conllu"\
        --eval_file "${out_dir}/dev_results.json" \
        &> ${run_log_path}
    echo "Done evaluating on dev. Printing results:"
    tail -n50 ${run_log_path}
}

eval_test() {
    run_log_path=${run_log_dir}/${wandb_name}_eval_test.out
    out_dir=${log_path}/${wandb_name}

    echo
    echo "Evaluating test data:"
    echo " - Treebank: ${treebank}"
    echo " - Run name: ${wandb_name}"
    echo " - Run tag: ${wandb_tag}"
    echo " - Config: ${config_path}"
    echo " - Train Data: ${train_data_path}"
    echo " - Dev Data: ${validation_data_path}"
    echo " - Test Data: ${test_data_path}"
    echo " - Archive: ${out_dir}"
    echo " - Log: ${run_log_path}"

    /home/ubuntu/.pyenv/versions/esr/bin/python predict.py\
        "${out_dir}/model.tar.gz"\
        "$test_data_path"\
        "${out_dir}/test_preds.conllu"\
        --eval_file "${out_dir}/test_results.json" \
        &> ${run_log_path}
    echo "Done evaluating on test. Printing results:"
    tail -n50 ${run_log_path}
    
}

make_archive() {
    out_dir=${log_path}/${wandb_name}
    echo
    echo "Creating archive for evaluating with"
    echo "  - Weights file: ${out_dir}/best.th"
    /home/ubuntu/.pyenv/versions/esr/bin/python make_archive.py ${out_dir}

}

sync_results() {
    out_dir=${log_path}/${wandb_name}
    MAIN_NODE_IP=172.31.37.7  # ohio
    # MAIN_NODE_IP=172.31.10.214  # n.virg
    echo "Syncing results to ${MAIN_NODE_IP}"
    ssh-keygen -R ${MAIN_NODE_IP} &> /dev/null
    ssh-keyscan -H ${MAIN_NODE_IP} >> ~/.ssh/known_hosts    
    # rsync -avzP -e "ssh -i /home/ubuntu/aws-ec2-mcollins.pem" --exclude="*th" --exclude="*model.tar.gz"  /home/ubuntu//data/ ubuntu@${MAIN_NODE_IP}:/home/ubuntu//data
    ssh -i /home/ubuntu/aws-ec2-mcollins.pem ubuntu@${MAIN_NODE_IP} mkdir -p /home/ubuntu//${out_dir}
    rsync -avzP -e "ssh -i /home/ubuntu/aws-ec2-mcollins.pem" --exclude="*th" --exclude="*model.tar.gz"  /home/ubuntu//$out_dir/ ubuntu@${MAIN_NODE_IP}:/home/ubuntu//${out_dir}    
}

sync_model() {
    out_dir=${log_path}/${wandb_name}
    MAIN_NODE_IP=172.31.37.7  # ohio
    # MAIN_NODE_IP=172.31.10.214  # n.virg
    echo "Syncing model from ${out_dir} to ${MAIN_NODE_IP}"
    ssh-keygen -R ${MAIN_NODE_IP} &> /dev/null
    ssh-keyscan -H ${MAIN_NODE_IP} >> ~/.ssh/known_hosts    
    rsync -avzP -e "ssh -i /home/ubuntu/aws-ec2-mcollins.pem" --exclude="*state*.th" --exclude="weights.th" --exclude="model.tar.gz" --include="best.th" /home/ubuntu//${out_dir}/ ubuntu@${MAIN_NODE_IP}:/home/ubuntu//${out_dir}    
}

cleanup_weights() {
    out_dir=${log_path}/${wandb_name}
    echo "Cleaning up weights from ${out_dir}"
    rm -f $out_dir/*state*th
    rm -f $out_dir/*weights.th
    rm -f $out_dir/*model.tar.gz
}