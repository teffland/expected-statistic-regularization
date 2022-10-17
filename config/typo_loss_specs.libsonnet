# Location for the stats comparison file used for defining losses
local stats_csv_path = std.extVar('stats_csv_path');
local loss_type = std.extVar('loss_type');
local loss_target = std.extVar('loss_target');
local loss_margin = std.extVar('loss_margin');
local topk = std.parseJson(std.extVar('topk'));
local topk_min = std.parseJson(std.extVar('topk_min'));
local topk_max = std.parseJson(std.extVar('topk_max'));
local worstk = std.parseJson(std.extVar('worstk'));
local worstk_min = std.parseJson(std.extVar('worstk_min'));
local worstk_max = std.parseJson(std.extVar('worstk_max'));
local zeros_only = std.parseJson(std.extVar('zeros_only'));
local preindex = std.parseJson(std.extVar('preindex'));
local mean_after_loss = std.parseJson(std.extVar('mean_after_loss'));

# Weights for each of the different losses we try in the paper, spaced by group
local pos_weight = std.parseJson(std.extVar('pos_weight'));
local pos_sent_weight =  std.parseJson(std.extVar('pos_sent_weight'));

local dep_dir_weight = std.parseJson(std.extVar('dep_dir_weight'));
local head_dir_weight = std.parseJson(std.extVar('head_dir_weight'));
local tail_dir_weight = std.parseJson(std.extVar('tail_dir_weight'));

local pos_pair_weight = std.parseJson(std.extVar('pos_pair_weight'));
local pos_pair_given_pos_weight = std.parseJson(std.extVar('pos_pair_given_pos_weight'));

local dep_dir_dist_weight = std.parseJson(std.extVar('dep_dir_dist_weight'));
local dep_dir_dist_given_dep_weight = std.parseJson(std.extVar('dep_dir_dist_given_dep_weight'));

local pos_valency_weight = std.parseJson(std.extVar('pos_valency_weight'));
local pos_valency_given_pos_weight = std.parseJson(std.extVar('pos_valency_given_pos_weight'));

local head_tail_dir_weight = std.parseJson(std.extVar('head_tail_dir_weight'));
local head_tail_dir_given_head_weight = std.parseJson(std.extVar('head_tail_dir_given_head_weight'));
local head_tail_dir_given_tail_weight = std.parseJson(std.extVar('head_tail_dir_given_tail_weight'));

local head_dep_dir_weight = std.parseJson(std.extVar('head_dep_dir_weight'));
local head_dep_dir_given_dep_weight = std.parseJson(std.extVar('head_dep_dir_given_dep_weight'));
local head_dep_dir_given_head_weight = std.parseJson(std.extVar('head_dep_dir_given_head_weight'));

local tail_dep_dir_weight = std.parseJson(std.extVar('tail_dep_dir_weight'));
local tail_dep_dir_given_dep_weight = std.parseJson(std.extVar('tail_dep_dir_given_dep_weight'));
local tail_dep_dir_given_tail_weight = std.parseJson(std.extVar('tail_dep_dir_given_tail_weight'));

local head_tail_dep_weight = std.parseJson(std.extVar('head_tail_dep_weight'));
local head_tail_dep_dir_weight = std.parseJson(std.extVar('head_tail_dep_dir_weight'));
local head_tail_dep_dir_given_head_weight = std.parseJson(std.extVar('head_tail_dep_dir_given_head_weight'));
local head_tail_dep_dir_given_tail_weight = std.parseJson(std.extVar('head_tail_dep_dir_given_tail_weight'));
local head_tail_dep_dir_given_dep_weight = std.parseJson(std.extVar('head_tail_dep_dir_given_dep_weight'));

local head_tail_dep_dir_sent_weight = std.parseJson(std.extVar('head_tail_dep_dir_sent_weight'));
local head_tail_dep_dir_sent_given_head_sent_weight = std.parseJson(std.extVar('head_tail_dep_dir_sent_given_head_sent_weight'));
local head_tail_dep_dir_sent_given_tail_sent_weight = std.parseJson(std.extVar('head_tail_dep_dir_sent_given_tail_sent_weight'));
local head_tail_dep_dir_sent_given_dep_sent_weight = std.parseJson(std.extVar('head_tail_dep_dir_sent_given_dep_sent_weight'));

local head_tail1_tail2_dir1_dir2_weight = std.parseJson(std.extVar('head_tail1_tail2_dir1_dir2_weight'));
local head_tail1_tail2_dir2_dir2_given_head_weight = std.parseJson(std.extVar('head_tail1_tail2_dir2_dir2_given_head_weight'));

local head_dep1_dep2_dir1_dir2_weight = std.parseJson(std.extVar('head_dep1_dep2_dir1_dir2_weight'));
local head_dep1_dep2_dir1_dir2_given_head_weight = std.parseJson(std.extVar('head_dep1_dep2_dir1_dir2_given_head_weight'));

local head_tail_gtail_dir_gdir_weight = std.parseJson(std.extVar('head_tail_gtail_dir_gdir_weight'));
local head_tail_gtail_dir_gdir_given_tail_weight = std.parseJson(std.extVar('head_tail_gtail_dir_gdir_given_tail_weight'));

local tail_dep_gdep_dir_gdir_weight = std.parseJson(std.extVar('tail_dep_gdep_dir_gdir_weight'));
local tail_dep_gdep_dir_gdir_given_tail_weight = std.parseJson(std.extVar('tail_dep_gdep_dir_gdir_given_tail_weight'));

local pos_entropy_weight = std.parseJson(std.extVar('pos_entropy_weight'));
local dep_entropy_weight = std.parseJson(std.extVar('dep_entropy_weight'));
local tree_entropy_weight = std.parseJson(std.extVar('tree_entropy_weight'));


{
    stats_csv_path: stats_csv_path,
    local spec = {
        type: loss_type,
        target: loss_target,
        margin: loss_margin,
        preindex: preindex,
        mean_after_loss: mean_after_loss,
        topk: topk,
        topk_min: topk_min,
        topk_max: topk_max,
        worstk: worstk,
        worstk_min: worstk_min,
        worstk_max: worstk_max,
        zeros_only: zeros_only,
    },
    losses: {
        "pos_freq": spec+{ weight: pos_weight },
        "pos_sent_freq": spec+{ weight: pos_sent_weight },
        "dep_dir_freq": spec+{ weight: dep_dir_weight },
        "head_dir_freq": spec+{ weight: head_dir_weight },
        "tail_dir_freq": spec+{ weight: tail_dir_weight },

        "pos_pair_freq": spec+{ weight: pos_pair_weight },
        "pos_pair_freq|pos_freq": spec+{ weight: pos_pair_given_pos_weight },

        "dep_dir_dist_freq": spec+{ weight: dep_dir_dist_weight },
        "dep_dir_dist_freq|dep_freq": spec+{ weight: dep_dir_dist_given_dep_weight },

        "pos_valency_freq": spec+{ weight: pos_valency_weight },
        "pos_valency_freq|pos_freq": spec+{ weight: pos_valency_given_pos_weight },

        "head_tail_dir_freq": spec+{ weight: head_tail_dir_weight },
        "head_tail_dir_freq|head_freq": spec+{ weight: head_tail_dir_given_head_weight },
        "head_tail_dir_freq|tail_freq": spec+{ weight: head_tail_dir_given_tail_weight },

        "head_dep_dir_freq": spec+{ weight: head_dep_dir_weight },
        "head_dep_dir_freq|dep_freq": spec+{ weight: head_dep_dir_given_dep_weight },
        "head_dep_dir_freq|head_freq": spec+{ weight: head_dep_dir_given_head_weight },

        "tail_dep_dir_freq": spec+{ weight: tail_dep_dir_weight },
        "tail_dep_dir_freq|dep_freq": spec+{ weight: tail_dep_dir_given_dep_weight },
        "tail_dep_dir_freq|tail_freq": spec+{ weight: tail_dep_dir_given_tail_weight },

        "head_tail_dep_freq": spec+{ weight: head_tail_dep_weight },
        "head_tail_dep_dir_freq": spec+{ weight: head_tail_dep_dir_weight },
        "head_tail_dep_dir_freq|head_freq": spec+{ weight: head_tail_dep_dir_given_head_weight },
        "head_tail_dep_dir_freq|tail_freq": spec+{ weight: head_tail_dep_dir_given_tail_weight },
        "head_tail_dep_dir_freq|dep_freq": spec+{ weight: head_tail_dep_dir_given_dep_weight },

        "head_tail_dep_dir_sent_freq": spec+{ weight: head_tail_dep_dir_sent_weight },
        "head_tail_dep_dir_sent_freq|head_sent_freq": spec+{ weight: head_tail_dep_dir_sent_given_head_sent_weight },
        "head_tail_dep_dir_sent_freq|tail_sent_freq": spec+{ weight: head_tail_dep_dir_sent_given_tail_sent_weight },
        "head_tail_dep_dir_sent_freq|dep_sent_freq": spec+{ weight: head_tail_dep_dir_sent_given_dep_sent_weight },

        "head_tail1_tail2_dir1_dir2_freq": spec+{ weight: head_tail1_tail2_dir1_dir2_weight },
        "head_tail1_tail2_dir1_dir2_freq|head_freq": spec+{ weight: head_tail1_tail2_dir2_dir2_given_head_weight },

        "head_dep1_dep2_dir1_dir2_freq": spec+{ weight: head_dep1_dep2_dir1_dir2_weight },
        "head_dep1_dep2_dir1_dir2_freq|head_freq": spec+{ weight: head_dep1_dep2_dir1_dir2_given_head_weight },

        "head_tail_gtail_dir_gdir_freq": spec+{ weight: head_tail_gtail_dir_gdir_weight },
        "head_tail_gtail_dir_gdir_freq|tail_freq": spec+{ weight: head_tail_gtail_dir_gdir_given_tail_weight },
        
        "tail_dep_gdep_dir_gdir_freq": spec+{ weight: tail_dep_gdep_dir_gdir_weight },
        "tail_dep_gdep_dir_gdir_freq|tail_freq": spec+{ weight: tail_dep_gdep_dir_gdir_given_tail_weight },

        pos_avg_entropy: {
            weight: pos_entropy_weight,
            type: 'l1-margin',
            target: 0.0,
            margin: 1.0,
        },
        dep_avg_entropy: {
            weight: dep_entropy_weight,
            type: 'l1-margin',
            target: 0.0,
            margin: 2.0,
        },
        tree_entropy: {
            weight: tree_entropy_weight,
            type: 'l1-margin',
            target: 0.0,
            margin: 10.0,
        },
    },
}