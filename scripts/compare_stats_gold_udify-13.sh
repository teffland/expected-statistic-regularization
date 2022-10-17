cd ~/

declare -a fs=(
    "UD_Persian-PerDT"
    "UD_Indonesian-GSD"
    "UD_Maltese-MUDT"
    "UD_Vietnamese-VTB"
    "UD_German-HDT"
)

declare -a stat_funcs=(
    "pos_freq" "dep_dir_freq" "head_dir_freq" "tail_dir_freq"
    "pos_pair_freq" "pos_pair_freq|pos_freq"
    "dep_dir_dist_freq" "dep_dir_dist_freq|dep_freq"
    "pos_valency_freq" "pos_valency_freq|pos_freq"
    "head_tail_dir_freq" "head_tail_dir_freq|head_freq" "head_tail_dir_freq|tail_freq"
    "head_dep_dir_freq" "head_dep_dir_freq|dep_freq" "head_dep_dir_freq|head_freq"
    "tail_dep_dir_freq" "tail_dep_dir_freq|dep_freq" "tail_dep_dir_freq|tail_freq"
    "head_tail_dep_freq"
    "head_tail_dep_dir_freq" "head_tail_dep_dir_freq|head_freq" "head_tail_dep_dir_freq|tail_freq" "head_tail_dep_dir_freq|dep_freq"
    "head_tail1_tail2_dir1_dir2_freq" "head_tail1_tail2_dir1_dir2_freq|head_freq"
    "head_dep1_dep2_dir1_dir2_freq" "head_dep1_dep2_dir1_dir2_freq|head_freq"
    "head_tail_gtail_dir_gdir_freq" "head_tail_gtail_dir_gdir_freq|tail_freq"
    "tail_dep_gdep_dir_gdir_freq" "tail_dep_gdep_dir_gdir_freq|tail_freq"
)


for tb in "${fs[@]}"
do

    train_fs=(/home/ubuntu//data/ud-treebanks-v2.8/$tb/*train.conllu)
    train_f="${train_fs[0]}"
    pred_f=${train_f%.conllu}_udify-13_pred.conllu
    stats_csv_path=${train_f%.conllu}_gold_udify-13_stats.csv
    jsd_csv_path=${train_f%.conllu}_gold_udify-13_jsd.csv

    cp $stats_csv_path ${stats_csv_path}_COPY

    echo $tb
    echo ==============================
    python -m ml.cmd.compare_datasets\
    --gold-paths $train_f\
    --pred-paths $pred_f\
    --out-csv-path $stats_csv_path\
    --out-jsd-csv-path $jsd_csv_path\
    --n-bootstrap-samples 1000\
    --batch-size 8\
    --stat-funcs ${stat_funcs[@]}
done
echo ALL DONE