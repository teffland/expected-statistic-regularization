# Log into a remote node at the specified ip, sync the code, and run the scropt
while getopts "ld" flag; do
case "$flag" in
    d) LOCAL=$flag;;  # -d just prints command
    l) LOCAL=$flag;;  # -l runs it locally
esac
done
shift $((OPTIND-1)) 
PUBLIC_IP=$1
PRIVATE_IP=$2
CMD=$3


# Print command if -d, run locally if -l, run remotely otherwise
if [[ $LOCAL == "d" ]]
then
    echo Command we would run is:
    printf "%s\n" "${CMD}"
elif [[ $LOCAL == "l" ]]
then
    echo "Running the following command locally:"
    printf "%s\n" "${CMD}"
    bash -c "${CMD}"
else
    # Setup experiment and make sure we have a clean remote connection
    ssh-keygen -R $PRIVATE_IP &> /dev/null
    ssh-keyscan -H $PRIVATE_IP >> ~/.ssh/known_hosts
    sleep 1s
    echo

    MAIN_NODE_IP=172.31.37.7  # ohio
    # MAIN_NODE_IP=172.31.10.214  # n.virg

    # Update the bashrc first
    echo SYNCING BASHRC
    scp -i /home/ubuntu/aws-ec2-mcollins.pem ~/.bashrc ubuntu@$PRIVATE_IP:~/.bashrc
    scp -i /home/ubuntu/aws-ec2-mcollins.pem /home/ubuntu//credential_helper.sh ubuntu@$PRIVATE_IP:/home/ubuntu//credential_helper.sh

    echo RUNNING THE FOLLOWING SCRIPT ON ${PUBLIC_IP}:
    echo --------------------------------------------------------
    printf "%s\n" "${CMD}"
    echo --------------------------------------------------------
    echo check on server at: http://${PUBLIC_IP}:8888
    echo
    echo
    echo  REMOTE OUTPUT
    echo ========================================================
    # Login to the remote node, sync the code, run jupyter notebook if needed, run the command, finally sync all logs back
    ssh -q -i /home/ubuntu/aws-ec2-mcollins.pem ubuntu@$PRIVATE_IP << EOF
source ~/.bashrc
cd /home/ubuntu/
/home/ubuntu/anaconda3/envs/env2/bin/pip install einops
echo "SETTING UP ENVIRONMENT"
echo ========================================================
ssh-keygen -R ${MAIN_NODE_IP} &> /dev/null
ssh-keyscan -H ${MAIN_NODE_IP} >> ~/.ssh/known_hosts
git config credential.helper "/bin/bash /home/ubuntu//credential_helper.sh"
git fetch --all
git reset --hard origin/main
pgrep jupyter >/dev/null && echo "Jupyter already on" || bash -c 'nohup /home/ubuntu/anaconda3/envs/env2/bin/jupyter notebook --ip 0.0.0.0 &'
echo
echo "SYNCING DATA TO REMOTE NODE"
echo ========================================================
rsync -avzP -e "ssh -i /home/ubuntu/aws-ec2-mcollins.pem"  ubuntu@${MAIN_NODE_IP}:/home/ubuntu//data/ud-treebanks-v2.8/ /home/ubuntu//data/ud-treebanks-v2.8/
rsync -avzP -e "ssh -i /home/ubuntu/aws-ec2-mcollins.pem"  ubuntu@${MAIN_NODE_IP}:/home/ubuntu//data/ud-treebanks-v2.2/ /home/ubuntu//data/ud-treebanks-v2.2/
rsync -avzP -e "ssh -i /home/ubuntu/aws-ec2-mcollins.pem"  ubuntu@${MAIN_NODE_IP}:/home/ubuntu//data/weights/ /home/ubuntu//data/weights/
rsync -avzP -e "ssh -i /home/ubuntu/aws-ec2-mcollins.pem"  ubuntu@${MAIN_NODE_IP}:/home/ubuntu//data/vocab/ /home/ubuntu//data/vocab/
# rsync -avzP -e "ssh -i /home/ubuntu/aws-ec2-mcollins.pem"  ubuntu@${MAIN_NODE_IP}:/home/ubuntu/udify/udify-13-model/ /home/ubuntu/udify/udify-en-model/
# rsync -avzP -e "ssh -i /home/ubuntu/aws-ec2-mcollins.pem"  ubuntu@${MAIN_NODE_IP}:/home/ubuntu/udify/udify-full-model/ /home/ubuntu/udify/udify-full-model/

echo
echo "RUNNING INPUT COMMAND:"
printf "%s\n" "${CMD}"
echo "REMINDER: jupyter server can be accessed at http://${PUBLIC_IP}:8888"
echo ========================================================
bash -c '${CMD}'
echo
echo "SYNCING DATA AND LOGS BACK TO MAIN NODE"
echo ========================================================
# rsync -avzP -e "ssh -i /home/ubuntu/aws-ec2-mcollins.pem" --exclude="*th" --exclude="*model.tar.gz"  /home/ubuntu//data/ ubuntu@${MAIN_NODE_IP}:/home/ubuntu//data
# rsync -avzP -e "ssh -i /home/ubuntu/aws-ec2-mcollins.pem" --exclude="*th" --exclude="*model.tar.gz"  /home/ubuntu//logs/ ubuntu@${MAIN_NODE_IP}:/home/ubuntu//logs
echo "ALL DONE. EXITING"
exit
EOF

fi


