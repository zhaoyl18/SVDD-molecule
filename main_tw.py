import torch
import argparse
import time
from parsers.parser import Parser
from parsers.config import get_config
from trainer import Trainer
from sampler import Sampler, Sampler_mol
from decode import decode_Sampler_mol
import wandb
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def main(work_type_args):
    ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())
    args = Parser().parse()
    config = get_config(args.config, args.seed)

    # -------- Train --------
    if work_type_args.type == 'train':
        trainer = Trainer(config) 
        ckpt = trainer.train(ts)
        if 'sample' in config.keys():
            config.ckpt = ckpt
            sampler = Sampler(config) 
            sampler.sample()

    # -------- Generation --------
    elif work_type_args.type == 'sample':
        if config.data.data in ['QM9', 'ZINC250k']:
            sampler = Sampler_mol(config)
        else:
            sampler = Sampler(config) 
        sampler.sample()
    elif work_type_args.type == 'decode_valueF':
        wandb.init()
        if config.data.data in ['QM9', 'ZINC250k']:
            sampler = decode_Sampler_mol(config, args.reward_name)
        else:
            sampler = Sampler(config)

        reward_model_preds, selected_baseline_preds, baseline_preds = sampler.controlled_tw_decode(sample_M=args.sample_M, largerbetter=True)
        hepg2_values_ours = reward_model_preds.cpu().numpy()
        hepg2_values_selected = selected_baseline_preds.cpu().numpy()
        hepg2_values_baseline = baseline_preds.cpu().numpy()
        # Create a DataFrame for seaborn

        np.savez( "./log/%s-%s-%s-%s-tw" %(config.data.data, args.sample_M, args.reward_name, args.version), decoding = hepg2_values_ours, baseline = hepg2_values_baseline)
        wandb.finish()
    elif work_type_args.type == 'decode_valueF_train':
        wandb.init()
        if config.data.data in ['QM9', 'ZINC250k']:
            sampler = decode_Sampler_mol(config, args.reward_name)
        else:
            raise ValueError('Wrong data, not mol')
        sampler.train(wandb)
        wandb.finish()
    else:
        raise ValueError(f'Wrong type : {work_type_args.type}')

if __name__ == '__main__':

    work_type_parser = argparse.ArgumentParser()
    work_type_parser.add_argument('--type', type=str, required=True)

    main(work_type_parser.parse_known_args()[0])
