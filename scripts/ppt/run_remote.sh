source ~//scripts/run_remote_experiment.sh
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

declare -A tb_public_ips=(
    ["UD_German-HDT"]="18.219.216.46"
    ["UD_Indonesian-GSD"]="3.145.36.41"
    ["UD_Maltese-MUDT"]="3.144.143.216"
    ["UD_Persian-PerDT"]="18.118.0.14"
    ["UD_Vietnamese-VTB"]="3.15.231.31"
)

declare -A tb_private_ips=(
    ["UD_German-HDT"]="172.31.31.59"
    ["UD_Indonesian-GSD"]="172.31.11.62"
    ["UD_Maltese-MUDT"]="172.31.38.22"
    ["UD_Persian-PerDT"]="172.31.13.128"
    ["UD_Vietnamese-VTB"]="172.31.1.245"  
)

name="ppt"

tb_script=${parent_path}/hp_search.sh
log_name=21-12-11_ud-13_${name}_unsup_hp
run_remote_experiment

tb_script=${parent_path}/oracle_ud-13.sh
log_name=21-12-11_oracle_ud-13_${name}
run_remote_experiment

tb_script=${parent_path}/ud-13_semisup_learning_curve.sh
log_name=21-12-17_ud-13_${name}_semisup_learning_curve
run_remote_experiment

tb_script=${parent_path}/scratch_semisup_learning_curve.sh
log_name=21-12-18_scratch_${name}_semisup_learning_curve
run_remote_experiment
