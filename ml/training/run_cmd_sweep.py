""" A helper script for interfacing arbitrary terminal commands with wandb sweep.

The sweep config should pass two positional arguments. The first is the command to run. The second is the params
as json. The json params will be set to environment variables available to the command

All of the keys/values from params will be set to environment variables. Then the `cmd` will be run as a subprocess.
This 
"""
import os
import json
import subprocess
from argparse import ArgumentParser
import logging


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    parser = ArgumentParser()
    parser.add_argument("cmd", type=str)
    parser.add_argument("params", type=str)
    args = parser.parse_args()
    
    logger.info(f'CMD: {args.cmd}')
    params = { k:str(v) for k,v in json.loads(args.params).items() }
    logger.info(f'PARAMS: {args.params}')
    env = os.environ.copy()
    env.update(params)
    try:
        subprocess.check_call(args.cmd, shell=True, env=env)  # stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except Exception as e:
        logger.error(e, exc_info=True)
        raise e
    logger.info("ALL DONE")
