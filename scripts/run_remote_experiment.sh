run_remote_experiment(){
    # Requires the following variables to exist in local context to run properly:
    #  - $tb_script - the actual script to run the experiment with the chosen tb
    #  - $log_name - name of outermost log file in the run_logs. Actual log will go to logs/run_logs/run_${logname}_${tb}.out
    #  - $tb_public_ips - a hashmap of treebanks to public ip addresses of remote nodes
    #  - $tb_private_ips - a hashmap of treebanks to private ip addresses of remote nodes
    
    # Iterate over treebanks
    for tb in "${!tb_public_ips[@]}"
    do

        cmd=$(cat << EOF
source ~/.bashrc
cd /home/ubuntu/
bash ${tb_script} ${tb} ${log_name}
EOF
)

        # Run the agent command on the remote ip
        public_ip=${tb_public_ips[$tb]}
        private_ip=${tb_private_ips[$tb]}
        logpath=logs/run_logs/run_${log_name}_${tb}.out
        echo
        echo $tb at $public_ip running cmd:
        echo $cmd
        echo Remote run output for $tb at: $logpath
        # nohup bash scripts/run_remote_experiment_worker.sh $public_ip $private_ip "${cmd}" &> $logpath &
        bash scripts/run_remote_experiment_worker.sh $public_ip $private_ip "${cmd}" &> $logpath
        sleep 1s

    done
    echo ALL DONE LAUNCHES
}