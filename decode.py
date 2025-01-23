import os
import time
import pickle
import math
from tqdm import trange
import torch
import wandb
import numpy as np
import pandas as pd
from torch.cuda.amp import GradScaler
from utils.loader import load_sde
from utils.logger import Logger, set_log, start_log, train_log, sample_log, check_log
from utils.loader import load_ckpt, load_data, load_seed, load_device, load_model_from_ckpt, \
                         load_ema_from_ckpt, load_sampling_fn, load_sampling_fn_refine, load_eval_settings
from utils.graph_utils import adjs_to_graphs, init_flags, quantize, quantize_mol
from utils.plot import save_graph_list, plot_graphs_list
from evaluation.stats import eval_graph_list
from utils.mol_utils import gen_mol, mols_to_smiles, load_smiles, canonicalize_smiles, mols_to_nx
from moses.metrics.metrics import get_all_metrics
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Contrib.SA_Score import sascorer
from GINs import BaseModel
from MOOD_scorer.scorer import get_scores
import gc
import multiprocessing as mp

from utils.graph_utils import mask_adjs, mask_x, gen_noise


# -------- Sampler for molecule generation tasks --------
class decode_Sampler_mol(object):
    def __init__(self, config, rewardf):
        self.config = config
        self.device = load_device()
        # -------- Load checkpoint --------
        self.ckpt_dict = load_ckpt(self.config, self.device)
        self.configt = self.ckpt_dict['config']

        # load_seed(self.config.seed)

        self.log_folder_name, self.log_dir, _ = set_log(self.configt, is_train=False)
        self.log_name = f"{self.config.ckpt}-sample"
        logger = Logger(str(os.path.join(self.log_dir, f'{self.log_name}.log')), mode='a')
        self.logger = logger

        if not check_log(self.log_folder_name, self.log_name):
            start_log(logger, self.configt)
            train_log(logger, self.configt)
            
        sample_log(logger, self.config)
        self.configt.data.dir = self.config.data.dir
        # -------- Load models --------
        self.model_x = load_model_from_ckpt(self.ckpt_dict['params_x'], self.ckpt_dict['x_state_dict'], self.device)
        self.model_adj = load_model_from_ckpt(self.ckpt_dict['params_adj'], self.ckpt_dict['adj_state_dict'], self.device)

        self.train_graph_list, _ = load_data(self.configt, get_graph_list=True)  # for init_flags
        # with open(f'data/{self.configt.data.data.lower()}_test_nx.pkl', 'rb') as f:
        #     self.test_graph_list = pickle.load(f)                                   # for NSPDK MMD
        self.logger.log(f"Gen batch size: {self.config.data.batch_size}")
        self.logger.log(f"Reward: {rewardf}")
        if rewardf == 'SA':
            self.rewardf = self.SA_reward
        elif 'vina' in rewardf:
            self.vina_type = int(rewardf[-1])
            self.rewardf = self.vina_reward
        elif rewardf == 'QED':
            self.rewardf = self.qed_reward
        elif rewardf == 'logbarrier':
            self.rewardf = self.logbarrier_reward

    def sample(self, strat='controlled', sample_M=20, alpha = 0.3, largerbetter=True, ckpt=False):
        self.sampling_fn = load_sampling_fn(self.configt, 
                                            self.config.sampler, 
                                            self.config.sample, 
                                            self.device, 
                                            self.config.data.batch_size, 
                                            strat=strat
                                        )

        self.init_flags = init_flags(self.train_graph_list, self.configt, self.config.data.batch_size).to(self.device[0])
        if strat=='controlled' or strat=='controlled_tw':
            if ckpt:
                x, adj, _ = self.sampling_fn(self.model_x, self.model_adj, self.init_flags, self.model.pred_out, sample_M=sample_M, larger_better=largerbetter)
            else:
                x, adj, _ = self.sampling_fn(self.model_x, self.model_adj, self.init_flags, self.rewardf, sample_M=sample_M, larger_better=largerbetter)
            return x, adj
        elif strat=='tds':
            x, adj, _ = self.sampling_fn(self.model_x, self.model_adj, self.init_flags, self.rewardf, alpha = alpha, larger_better=largerbetter)
            return x, adj
        else:
            x, adj, x_mid, adj_mid, _ = self.sampling_fn(self.model_x, self.model_adj, self.init_flags)
            return x.detach(), adj.detach(), x_mid, adj_mid

    def sample_refinement(self, K, x_s, adj_s, flags, strat='controlled', sample_M=20, alpha = 0.3, largerbetter=True, ckpt=False):
        sampling_fn = load_sampling_fn_refine(K,
                                            self.configt, 
                                            self.config.sampler, 
                                            self.config.sample, 
                                            self.device, 
                                            self.config.data.batch_size, 
                                            strat=strat
                                        )

        # self.init_flags = init_flags(self.train_graph_list, self.configt, self.config.data.batch_size).to(self.device[0])
        if strat=='controlled' or strat=='controlled_tw':
            if ckpt:
                x, adj, _ = sampling_fn(x_s, adj_s, flags, self.model_x, self.model_adj, self.init_flags, self.model.pred_out, sample_M=sample_M, larger_better=largerbetter)
            else:
                x, adj, _ = sampling_fn(x_s, adj_s, flags, self.model_x, self.model_adj, self.init_flags, self.rewardf, sample_M=sample_M, larger_better=largerbetter)
            return x, adj
        else:
            pass
        # elif strat=='tds':
        #     x, adj, _ = self.sampling_fn(self.model_x, self.model_adj, self.init_flags, self.rewardf, alpha = alpha, larger_better=largerbetter)
        #     return x, adj
        # else:
        #     x, adj, x_mid, adj_mid, _ = self.sampling_fn(self.model_x, self.model_adj, self.init_flags)
        #     return x.detach(), adj.detach(), x_mid, adj_mid
    

    @torch.no_grad()
    def controlled_decode(self, gen_batch_num=1, sample_M=20, largerbetter=True):
        # samples = []
        # value_func_preds = []
        reward_model_preds = []
        for i in range(gen_batch_num):
            x, adj = self.sample(strat='controlled', sample_M=sample_M, largerbetter=largerbetter)
            qeds = self.rewardf(x, adj)
            reward_model_preds.extend(qeds)

        self.logger.log("Value-weighted sampling done.")
        # baseline_samples = []
        baseline_preds = []
        all_preds = []
        for i in range(gen_batch_num * sample_M):
            x, adj, x_mid, adj_mid = self.sample(strat='ori')
            qeds = self.rewardf(x, adj)
            if i < gen_batch_num:
                baseline_preds.extend(qeds)
            all_preds.extend(qeds)
        self.logger.log("Baseline sampling done.")

        all_values = torch.cat(all_preds)
        # Compute the number of top elements to select
        k = int(len(all_values) / sample_M)
        # Get the top k values
        if largerbetter:
            top_k_values, _ = torch.topk(all_values, k)  # larger better, else: largest=False
        else:
            top_k_values, _ = torch.topk(all_values, k, largest=False)
        return torch.cat(reward_model_preds), top_k_values, torch.cat(baseline_preds)

    @torch.no_grad()
    def simple_decode(self, gen_batch_num=1, sample_M=1, largerbetter=True):
        
        # samples = []
        
        # baseline_samples = []
        baseline_preds = []
        all_preds = []
        for i in range(gen_batch_num * sample_M):
            x, adj, x_mid, adj_mid = self.sample(strat='ori')
            qeds = self.rewardf(x, adj)
            if i < gen_batch_num:
                baseline_preds.extend(qeds)
            all_preds.extend(qeds)
        self.logger.log("Baseline sampling done.")

        all_values = torch.cat(all_preds)
        # Compute the number of top elements to select
        k = int(len(all_values) / sample_M)
        # Get the top k values
        if largerbetter:
            top_k_values, _ = torch.topk(all_values, k)  # larger better, else: largest=False
        else:
            top_k_values, _ = torch.topk(all_values, k, largest=False)
        return top_k_values, torch.cat(baseline_preds)
    
    @torch.no_grad()
    def controlled_valueF_decode(self, gen_batch_num=1, sample_M=20, largerbetter=True):
        self.logger.log(f"largerbetter: {largerbetter}")
        x_v, adj_v, x_mid_v, adj_mid_v = self.sample(strat='ori')
        val_targets = self.rewardf(x_v, adj_v).repeat(1000, 1)
        self.model = BaseModel(self.config, x_mid_v, adj_mid_v, val_targets)
        self.logger.log(f"loading stored model: {self.config.load_checkpoint_path}")
        checkpoint = torch.load(self.config.load_checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.logger.log(f'total params: {sum(p.numel() for p in self.model.parameters())}')
        self.model.cuda()
        self.model.eval()
        # samples = []
        value_func_preds = []
        reward_model_preds = []
        for i in range(gen_batch_num):
            x, adj = self.sample(strat='controlled', sample_M=sample_M, largerbetter=largerbetter, ckpt=True)
            qeds = self.rewardf(x, adj)
            reward_model_preds.extend(qeds)
            # samples.append(batch_samples)
            value_func_preds.extend(self.model.pred_out(x.cuda(), adj.cuda()).detach())

        self.logger.log("Value-weighted sampling done.")
        # baseline_samples = []
        baseline_preds = []
        all_preds = []
        for i in range(gen_batch_num * sample_M):
            x, adj, x_mid, adj_mid = self.sample(strat='ori')
            qeds = self.rewardf(x, adj)
            if i < gen_batch_num:
                baseline_preds.extend(qeds)
            all_preds.extend(qeds)
        self.logger.log("Baseline sampling done.")

        all_values = torch.cat(all_preds)
        # Compute the number of top elements to select
        k = int(len(all_values) / sample_M)
        # Get the top k values
        if largerbetter:
            top_k_values, _ = torch.topk(all_values, k)  # larger better, else: largest=False
        else:
            top_k_values, _ = torch.topk(all_values, k, largest=False)
        return torch.cat(value_func_preds), torch.cat(reward_model_preds), top_k_values, torch.cat(baseline_preds)

    @torch.no_grad()
    def controlled_tw_decode(self, gen_batch_num=1, sample_M=10, largerbetter=True):
        reward_model_preds = []
        for i in range(gen_batch_num):
            x, adj = self.sample(strat='controlled_tw', sample_M=sample_M, largerbetter=largerbetter)

            qeds = self.rewardf(x, adj)
            reward_model_preds.extend(qeds)
            # samples.append(batch_samples)
            # value_func_preds.extend(self.model.pred_out(x.cuda(), adj.cuda(), qeds.cuda()).detach())
            # if self.task == "rna_saluki":
            #     pred = self.reward_model(self.transform_samples_saluki(batch_samples).float()).detach().squeeze(2)
            # else:
            #     pred = self.reward_model(onehot_samples.float().transpose(1, 2)).detach()[:, 0]

        self.logger.log("Value-weighted sampling done.")
        # baseline_samples = []
        baseline_preds = []
        all_preds = []
        for i in range(gen_batch_num * sample_M):
            x, adj, x_mid, adj_mid = self.sample(strat='ori')
            qeds = self.rewardf(x, adj)
            if i < gen_batch_num:
                baseline_preds.extend(qeds)
            all_preds.extend(qeds)
        self.logger.log("Baseline sampling done.")

        all_values = torch.cat(all_preds)
        # Compute the number of top elements to select
        k = int(len(all_values) / sample_M)
        # Get the top k values
        if largerbetter:
            top_k_values, _ = torch.topk(all_values, k)  # larger better, else: largest=False
        else:
            top_k_values, _ = torch.topk(all_values, k, largest=False)
        return torch.cat(reward_model_preds), top_k_values, torch.cat(baseline_preds)

    @torch.no_grad()
    def controlled_tw_decode_sequential_refinement(self, gen_batch_num=1, sample_M=10, K=50, S=5, largerbetter=True):
        reward_model_preds = []
        for i in range(gen_batch_num):
            
            # Initial design
            x, adj = self.sample(strat='controlled_tw', sample_M=sample_M, largerbetter=largerbetter)
            flags = self.init_flags
            
            x_s = x
            adj_s = adj
            
            for s in trange(0, (S), desc = '[Seq-refine iter]', position = 1, leave=False):
                # Phase 1: noising
                sde_x = load_sde(self.configt.sde.x) #sde.x : {'type': 'VP', 'beta_min': 0.1, 'beta_max': 1.0, 'num_scales': 1000}
                sde_adj = load_sde(self.configt.sde.adj) # sde.adj : {'type': 'VE', 'beta_min': 0.2, 'beta_max': 1.0, 'num_scales': 1000}
                
                eps = self.config.sample.eps
                timesteps = torch.linspace(sde_adj.T, eps, sde_x.N, device=x_s.device)
                t = torch.ones(x.shape[0], device=x_s.device) * timesteps[sde_x.N-1-K]

                z_x = gen_noise(x_s, flags, sym=False)
                mean_x, std_x = sde_x.marginal_prob(x_s, t)
                noised_x = mean_x + std_x[:, None, None] * z_x
                noised_x = mask_x(noised_x, flags)

                z_adj = gen_noise(adj_s, flags, sym=True) 
                mean_adj, std_adj = sde_adj.marginal_prob(adj_s, t)
                noised_adj = mean_adj + std_adj[:, None, None] * z_adj
                noised_adj = mask_adjs(noised_adj, flags)

                
                # Phase 2: reward optimization
                x_s, adj_s = self.sample_refinement(K, 
                                     noised_x,
                                     noised_adj,
                                     flags,
                                     strat='controlled_tw', 
                                     sample_M=sample_M, 
                                     largerbetter=largerbetter
                                )
                

            qeds = self.rewardf(x_s, adj_s)
            reward_model_preds.extend(qeds)

        self.logger.log("Sequantial refinement sampling done.")

        baseline_preds = []
        all_preds = []
        for i in range(gen_batch_num * sample_M):
            x, adj, x_mid, adj_mid = self.sample(strat='ori')
            qeds = self.rewardf(x, adj)
            if i < gen_batch_num:
                baseline_preds.extend(qeds)
            all_preds.extend(qeds)
        self.logger.log("Best of N (baseline) sampling done.")

        all_values = torch.cat(all_preds)
        
        # Compute the number of top elements to select
        k = int(len(all_values) / sample_M)
        # Get the top k values
        if largerbetter:
            top_k_values, _ = torch.topk(all_values, k)  # larger better, else: largest=False
        else:
            top_k_values, _ = torch.topk(all_values, k, largest=False)
        return torch.cat(reward_model_preds), top_k_values, torch.cat(baseline_preds)


    @torch.no_grad()
    def controlled_tds_decode(self, gen_batch_num=1, alpha = 0.3, sample_M =1 ,largerbetter=True):
        reward_model_preds = []
        for i in range(gen_batch_num):
            x, adj = self.sample(strat='tds', alpha = alpha, largerbetter=largerbetter)

            qeds = self.rewardf(x, adj)
            reward_model_preds.extend(qeds)

        self.logger.log("Value-weighted sampling done.")
        # baseline_samples = []
        baseline_preds = []
        all_preds = []
        for i in range(gen_batch_num * sample_M):
            x, adj, x_mid, adj_mid = self.sample(strat='ori')
            qeds = self.rewardf(x, adj)
            if i < gen_batch_num:
                baseline_preds.extend(qeds)
            all_preds.extend(qeds)
        self.logger.log("Baseline sampling done.")

        all_values = torch.cat(all_preds)
        # Compute the number of top elements to select
        k = int(len(all_values) / sample_M)
        # Get the top k values
        if largerbetter:
            top_k_values, _ = torch.topk(all_values, k)  # larger better, else: largest=False
        else:
            top_k_values, _ = torch.topk(all_values, k, largest=False)
        return torch.cat(reward_model_preds), top_k_values, torch.cat(baseline_preds)
    
    def Save_Mol(self, x, adj, app=None):
        samples_int = quantize_mol(adj)

        samples_int = samples_int - 1
        samples_int[samples_int == -1] = 3      # 0, 1, 2, 3 (no, S, D, T) -> 3, 0, 1, 2

        adj = torch.nn.functional.one_hot(torch.tensor(samples_int), num_classes=4).permute(0, 3, 1, 2)
        x = torch.where(x > 0.5, 1, 0)
        x = torch.concat([x, 1 - x.sum(dim=-1, keepdim=True)], dim=-1)      # 32, 9, 4 -> 32, 9, 5

        gen_mols, num_mols_wo_correction = gen_mol(x, adj, self.configt.data.data)
        # num_mols = len(gen_mols)

        gen_smiles = mols_to_smiles(gen_mols)
        gen_smiles = [smi for smi in gen_smiles if len(smi)]
        
        # -------- Save generated molecules --------
        if app is not None:
            with open(os.path.join(self.log_dir, f'{self.log_name}_{app}.txt'), 'a') as f:
                for smiles in gen_smiles:
                    f.write(f'{smiles}\n')
        else:
            with open(os.path.join(self.log_dir, f'{self.log_name}.txt'), 'a') as f:
                 for smiles in gen_smiles:
                     f.write(f'{smiles}\n')

    def vina_reward(self, x, adj, protein='parp1'):
        samples_int = quantize_mol(adj)

        samples_int = samples_int - 1
        samples_int[samples_int == -1] = 3  # 0, 1, 2, 3 (no, S, D, T) -> 3, 0, 1, 2

        adj = torch.nn.functional.one_hot(torch.tensor(samples_int), num_classes=4).permute(0, 3, 1, 2)
        x = torch.where(x > 0.5, 1, 0)
        x = torch.concat([x, 1 - x.sum(dim=-1, keepdim=True)], dim=-1)  # 32, 9, 4 -> 32, 9, 5

        gen_mols, _ = gen_mol(x, adj, self.configt.data.data)
        num_mols = len(gen_mols)
        if self.vina_type == 1:
            vina1 = get_scores('parp1', gen_mols)
            print(vina1)
            return torch.FloatTensor(vina1).unsqueeze(1)
        elif self.vina_type == 2:
            vina2 = get_scores('fa7', gen_mols)
            print(vina2)
            return torch.FloatTensor(vina2).unsqueeze(1)
        elif self.vina_type == 3:
            vina3 = get_scores('5ht1b', gen_mols)
            print(vina3)
            return torch.FloatTensor(vina3).unsqueeze(1)
        elif self.vina_type == 4:
            vina4 = get_scores('jak2', gen_mols)
            print(vina4)
            return torch.FloatTensor(vina4).unsqueeze(1)
        elif self.vina_type == 5:
            vina5 = get_scores('braf', gen_mols)
            print(vina5)
            return torch.FloatTensor(vina5).unsqueeze(1)
        else:
            raise ValueError('Unknown target protein')

    def SA_reward(self, x, adj):
        samples_int = quantize_mol(adj)

        samples_int = samples_int - 1
        samples_int[samples_int == -1] = 3      # 0, 1, 2, 3 (no, S, D, T) -> 3, 0, 1, 2

        adj = torch.nn.functional.one_hot(torch.tensor(samples_int), num_classes=4).permute(0, 3, 1, 2)
        x = torch.where(x > 0.5, 1, 0)
        x = torch.concat([x, 1 - x.sum(dim=-1, keepdim=True)], dim=-1)      # 32, 9, 4 -> 32, 9, 5

        gen_mols, _ = gen_mol(x, adj, self.configt.data.data)
        num_mols = len(gen_mols)

        scores = [ (10.0 - sascorer.calculateScore(mol))/9.0 if mol!=None else 0.0 for mol in gen_mols]
        return torch.FloatTensor(scores).unsqueeze(1)

    def qed_reward(self, x, adj):
        samples_int = quantize_mol(adj)

        samples_int = samples_int - 1
        samples_int[samples_int == -1] = 3      # 0, 1, 2, 3 (no, S, D, T) -> 3, 0, 1, 2

        adj = torch.nn.functional.one_hot(torch.tensor(samples_int), num_classes=4).permute(0, 3, 1, 2)
        x = torch.where(x > 0.5, 1, 0)
        x = torch.concat([x, 1 - x.sum(dim=-1, keepdim=True)], dim=-1)      # 32, 9, 4 -> 32, 9, 5

        gen_mols, _n = gen_mol(x, adj, self.configt.data.data)

        qed_scores = [QED.qed(mol) if mol!=None else 0.0 for mol in gen_mols]

        return torch.FloatTensor(qed_scores).unsqueeze(1)

    def logbarrier_reward(self, x, adj, c1=1.0, c2=0.01, c=0.70):
        qed_scores = self.qed_reward(x, adj)
        sa_scores = self.SA_reward(x, adj)
        
        penalized_scores = qed_scores + c2 * torch.log(torch.max(sa_scores - c, torch.tensor(c1, device=sa_scores.device, dtype=sa_scores.dtype)))

        
        return penalized_scores
        
        

    def save_checkpoint(self, epoch, model, best_loss, optimizer, tokens, scaler, save_path):
        raw_model = model.module if hasattr(model, "module") else model
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': raw_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),  # Include scaler state
            'tokens': tokens,
            'best_loss': best_loss,
        }
        if self.config.dist:
            if self.device == 0:
                torch.save(checkpoint, save_path)
        else:
            torch.save(checkpoint, save_path)
        self.logger.log(f"Checkpoint saved to {save_path}")

    def load_checkpoint(self, load_path, optimizer, scaler):
        checkpoint = torch.load(load_path, map_location='cuda')
        # self.model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.tokens = checkpoint['tokens']
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        return checkpoint['epoch'], checkpoint['best_loss']

    def train(self, wandb, train_strat='rewd'):
        self.logger.log(f'train strat: {train_strat}')
        x_v, adj_v, x_mid_v, adj_mid_v = self.sample(strat='ori' if train_strat!='PM' else 'PM')
        if train_strat == 'ori':
            val_targets = self.rewardf(x_v, adj_v).repeat(1000, 1)
        else:
            val_targets = self.rewardf(torch.cat(x_mid_v, dim=0), torch.cat(adj_mid_v, dim=0))
        self.model = BaseModel(self.config, x_mid_v, adj_mid_v, val_targets)
        self.model = torch.nn.DataParallel(self.model).to(f'cuda:{self.device[0]}')
        self.tokens = 0
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        # optimizer = torch.optim.AdamW(raw_model.parameters(), lr=config.learning_rate, betas=config.betas)
        optimizer = raw_model.configure_optimizers(config, self.logger)
        scaler = GradScaler()
        if config.load_checkpoint_path is not None:
            self.logger.log(f'resuming training from {config.load_checkpoint_path}...')
            start_epoch, best_loss = self.load_checkpoint(config.load_checkpoint_path, optimizer, scaler)
            # model = self.model
        else:
            start_epoch = -1
            best_loss = float('inf')
            self.tokens = 0  # counter used for learning rate decay

        # wandb.define_metric("val_MSE", step_metric="val_step")
        wandb.define_metric("test_MSE", step_metric="test_step")
        # wandb.define_metric("test_pearsonR", step_metric="test_step")
        wandb.define_metric("train_MSE", step_metric="train_step")

        def run_epoch(split, epoch):
            is_train = (split == 'train')
            model.train(is_train)

            losses = []
            for it in range(config.max_iter):
                x, adj, x_mid, adj_mid = self.sample(strat='ori' if train_strat!='PM' else 'PM')
                if train_strat == 'ori':
                    targets = self.rewardf(x, adj).repeat(1000, 1)
                else:
                    targets = self.rewardf(torch.cat(x_mid, dim=0), torch.cat(adj_mid, dim=0))
                # forward the model
                with torch.cuda.amp.autocast():
                    with torch.set_grad_enabled(is_train):

                        loss = model(x_mid, adj_mid, targets)
                        loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                        losses.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    scaler.step(optimizer)
                    scaler.update()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += 32 * 128 * 200 * 4
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(
                                max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    if (it + epoch * config.max_iter) % 500 == 0:
                        self.logger.log(
                            f"step_train_loss: {loss} train_step: {it + epoch * config.max_iter}, learning_rate: {lr}")

                    wandb.log({'train_MSE': loss, 'train_step': int(it + epoch * config.max_iter)})
                    if (it + epoch * config.max_iter) % 30 == 0:
                        test_loss = eval_seq_step('test', epoch, ((it + epoch * config.max_iter) / 30))
                        self.logger.log(
                            f"step: {it + epoch * config.max_iter}, test_loss: {test_loss}")
                        ckpt_path = f'storage/{self.config.run_name}_it{it + epoch * config.max_iter}.pt'
                        self.logger.log(f'Saving at latest epoch: {ckpt_path}')
                        self.save_checkpoint(epoch, model, best_loss, optimizer, self.tokens, scaler, ckpt_path)
                        model.train(is_train)

            if is_train:
                return float(np.mean(losses)), test_loss

            if not is_train:
                test_loss = float(np.mean(losses))
                print("eval loss: %f", test_loss)
                return test_loss

        def eval_seq_step(split, epoch, num_run):
            is_train = (split == 'train')
            model.train(is_train)
            # forward the model
            with torch.cuda.amp.autocast():
                with torch.set_grad_enabled(is_train):
                    losses = model.module.evaluate_seq_step(self.config.data.batch_size)  # Enformer  , pearsons

            for it, loss in enumerate(losses):
                wandb.log({'test_MSE': loss, 'test_step': int(it + num_run * 1000)})
            self.logger.log(f"last eval step: {999 + num_run * 1000}")

            test_loss = np.mean(losses)
            print("eval loss: %f", test_loss)
            return test_loss

        for epoch in range(start_epoch + 1, config.max_epochs):

            train_loss, test_loss = run_epoch('train', epoch)
            self.logger.log(f"epoch_train_loss: {train_loss}, epoch: {epoch + 1}")

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = test_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss
                self.logger.log(f'Saving at epoch {epoch + 1}: {self.config.ckpt_path}')
                # self.save_checkpoint()
                self.save_checkpoint(epoch, model, best_loss, optimizer, self.tokens, scaler, self.config.ckpt_path)

            if ((epoch + 1) >= self.config.save_start_epoch and (
                    epoch + 1) % self.config.save_interval_epoch == 0) or epoch == config.max_epochs - 1:
                # last_model = self.model.module if hasattr(self.model, "module") else self.model
                ckpt_path = f'../cond_gpt/weights/{self.config.run_name}_ep{epoch + 1}.pt'
                self.logger.log(f'Saving at latest epoch {epoch + 1}: {ckpt_path}')
                if self.config.dist:
                    if self.device == 0:
                        self.save_checkpoint(epoch, model, best_loss, optimizer, self.tokens, scaler, ckpt_path)
                else:
                    self.save_checkpoint(epoch, model, best_loss, optimizer, self.tokens, scaler, ckpt_path)

        return None


