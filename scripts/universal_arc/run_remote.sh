source ~//scripts/run_remote_experiment.sh

declare -A tb_public_ips=(
    ["UD_German-HDT"]="3.22.187.206"
    ["UD_Indonesian-GSD"]="3.21.100.177"
    ["UD_Maltese-MUDT"]="18.224.184.138"
    ["UD_Persian-PerDT"]="3.21.90.172"
    ["UD_Vietnamese-VTB"]="3.145.74.32"
)

declare -A tb_private_ips=(
    ["UD_German-HDT"]="172.31.31.59"
    ["UD_Indonesian-GSD"]="172.31.11.62"
    ["UD_Maltese-MUDT"]="172.31.38.22"
    ["UD_Persian-PerDT"]="172.31.37.7"
    ["UD_Vietnamese-VTB"]="172.31.1.245"  
)


tb_script=~//scripts/universal_arc/scratch_semisup_real_learning_curve.sh
log_name=21-12-27_scratch_universal_semisup_learning_curve
run_remote_experiment


tb_script=~//scripts/universal_arc/ud-13_semisup_real_learning_curve.sh
log_name=21-12-28_ud-13_universal_semisup_learning_curve
run_remote_experiment

