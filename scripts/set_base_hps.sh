set_base_hps() {
    echo "Setting base HPs"
    # Standard hyperparams
    export validation_metric="+.run/.sum"
    export loss_type="smoothl1"
    export loss_target="mean" 
    export loss_margin="stddev"
    export random_seed=0
    export train_data_limit=0
    export batch_size=8
    export num_batches_per_epoch=1000
    export num_epochs=25
    export scheduler_num_epochs=$num_epochs
    export patience=5
    export learning_rate=0.0000002  # 2e-7
    export weight_decay=0.0001  # 1e-4
    export topk=0
    export topk_min=10
    export topk_max=500
    export worstk=0
    export worstk_min=10
    export worstk_max=500
    export zeros_only=false
    export preindex=false
    export mean_after_loss=false

    # Set the loss weights all to zero. Individual methods can then just set the ones they want
    export pos_weight=0.0
    export pos_sent_weight=0.0
    export dep_dir_weight=0.0
    export head_dir_weight=0.0
    export tail_dir_weight=0.0
    export pos_pair_weight=0.0
    export pos_pair_given_pos_weight=0.0
    export dep_dir_dist_weight=0.0
    export dep_dir_dist_given_dep_weight=0.0
    export pos_valency_weight=0.0
    export pos_valency_given_pos_weight=0.0
    export head_tail_dir_weight=0.0
    export head_tail_dir_given_head_weight=0.0
    export head_tail_dir_given_tail_weight=0.0
    export head_dep_dir_weight=0.0
    export head_dep_dir_given_dep_weight=0.0
    export head_dep_dir_given_head_weight=0.0
    export tail_dep_dir_weight=0.0
    export tail_dep_dir_given_dep_weight=0.0
    export tail_dep_dir_given_tail_weight=0.0
    export head_tail_dep_weight=0.0
    export head_tail_dep_dir_weight=0.0
    export head_tail_dep_dir_sent_weight=0.0
    export head_tail_dep_dir_given_head_weight=0.0
    export head_tail_dep_dir_given_tail_weight=0.0
    export head_tail_dep_dir_given_dep_weight=0.0
    export head_tail_dep_dir_sent_given_head_sent_weight=0.0
    export head_tail_dep_dir_sent_given_tail_sent_weight=0.0
    export head_tail_dep_dir_sent_given_dep_sent_weight=0.0
    export head_tail1_tail2_dir1_dir2_weight=0.0
    export head_tail1_tail2_dir2_dir2_given_head_weight=0.0
    export head_dep1_dep2_dir1_dir2_weight=0.0
    export head_dep1_dep2_dir1_dir2_given_head_weight=0.0
    export head_tail_gtail_dir_gdir_weight=0.0
    export head_tail_gtail_dir_gdir_given_tail_weight=0.0
    export tail_dep_gdep_dir_gdir_weight=0.0
    export tail_dep_gdep_dir_gdir_given_tail_weight=0.0
    
    export pos_entropy_weight=0.1
    export dep_entropy_weight=0.1
    export tree_entropy_weight=0.1
}