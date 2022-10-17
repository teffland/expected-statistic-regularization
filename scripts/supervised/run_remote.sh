source ~//scripts/run_remote_experiment.sh
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

declare -A tb_public_ips=(
    ["UD_German-HDT"]="3.145.79.168"
    ["UD_Indonesian-GSD"]="3.145.79.168"
    ["UD_Maltese-MUDT"]="3.145.79.168"
    ["UD_Persian-PerDT"]="3.145.79.168"
    ["UD_Vietnamese-VTB"]="3.145.79.168"
)

declare -A tb_private_ips=(
    ["UD_German-HDT"]="172.31.13.128"
    ["UD_Indonesian-GSD"]="172.31.13.128"
    ["UD_Maltese-MUDT"]="172.31.13.128"
    ["UD_Persian-PerDT"]="172.31.13.128"
    ["UD_Vietnamese-VTB"]="172.31.13.128"  
)

name="sup"

tb_script=${parent_path}/ud-13_hp_search.sh
log_name=21-12-11_ud-13_${name}_hp_others
run_remote_experiment

tb_script=${parent_path}/scratch_hp_search.sh
log_name=21-12-11_scratch_${name}_hp
run_remote_experiment

tb_script=${parent_path}/ud-13_learning_curve.sh
log_name=21-12-12_ud-13_learning_curve
run_remote_experiment

tb_script=${parent_path}/scratch_learning_curve.sh
log_name=21-12-12_scratch_learning_curve
run_remote_experiment
