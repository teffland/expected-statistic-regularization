source ~//scripts/run_remote_experiment.sh
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

declare -A tb_public_ips=(
    ["UD_German-HDT"]="18.223.97.84"
    ["UD_Indonesian-GSD"]="52.14.125.132"
    ["UD_Maltese-MUDT"]="3.21.128.127"
    ["UD_Persian-PerDT"]="3.21.90.172"
    ["UD_Vietnamese-VTB"]="3.15.231.31"
)

declare -A tb_private_ips=(
    ["UD_German-HDT"]="172.31.31.59"
    ["UD_Indonesian-GSD"]="172.31.11.62"
    ["UD_Maltese-MUDT"]="172.31.38.22"
    ["UD_Persian-PerDT"]="172.31.37.7"
    ["UD_Vietnamese-VTB"]="172.31.1.245"  
)


name="trd"


tb_script=${parent_path}/ud-13_semisup_real_learning_curve.sh
log_name=21-12-14_ud-13_${name}_semisup_real_learning_curve
run_remote_experiment

tb_script=${parent_path}/ud-13_semisup_oracle_learning_curve.sh
log_name=21-12-14_ud-13_${name}_semisup_oracle_learning_curve
run_remote_experiment

tb_script=${parent_path}/scratch_semisup_real_learning_curve.sh
log_name=21-12-19_scratch_${name}_semisup_real_learning_curve
run_remote_experiment

