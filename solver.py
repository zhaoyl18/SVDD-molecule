import torch
import numpy as np
import abc
from tqdm import trange

from losses import get_score_fn
from utils.graph_utils import mask_adjs, mask_x, gen_noise
from sde import VPSDE, subVPSDE


class Predictor(abc.ABC):
  """The abstract class for a predictor algorithm."""
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__()
    self.sde = sde
    # Compute the reverse SDE/ODE
    self.rsde = sde.reverse(score_fn, probability_flow)
    self.score_fn = score_fn

  @abc.abstractmethod
  def update_fn(self, x, t, flags):
    pass


class Corrector(abc.ABC):
  """The abstract class for a corrector algorithm."""
  def __init__(self, sde, score_fn, snr, scale_eps, n_steps):
    super().__init__()
    self.sde = sde
    self.score_fn = score_fn
    self.snr = snr
    self.scale_eps = scale_eps
    self.n_steps = n_steps

  @abc.abstractmethod
  def update_fn(self, x, t, flags):
    pass


class EulerMaruyamaPredictor(Predictor):
  def __init__(self, obj, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)
    self.obj = obj

  def update_fn(self, x, adj, flags, t):
    dt = -1. / self.rsde.N

    if self.obj=='x':
      z = gen_noise(x, flags, sym=False)
      drift, diffusion = self.rsde.sde(x, adj, flags, t, is_adj=False)
      x_mean = x + drift * dt
      x = x_mean + diffusion[:, None, None] * np.sqrt(-dt) * z
      return x, x_mean

    elif self.obj=='adj':
      z = gen_noise(adj, flags)
      drift, diffusion = self.rsde.sde(x, adj, flags, t, is_adj=True)
      adj_mean = adj + drift * dt
      adj = adj_mean + diffusion[:, None, None] * np.sqrt(-dt) * z

      return adj, adj_mean

    else:
      raise NotImplementedError(f"obj {self.obj} not yet supported.")


class ReverseDiffusionPredictor(Predictor):
  def __init__(self, obj, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)
    self.obj = obj

  def update_fn(self, x, adj, flags, t):

    if self.obj == 'x':
      f, G = self.rsde.discretize(x, adj, flags, t, is_adj=False)
      z = gen_noise(x, flags, sym=False)
      x_mean = x - f
      x = x_mean + G[:, None, None] * z
      return x, x_mean

    elif self.obj == 'adj':
      f, G = self.rsde.discretize(x, adj, flags, t, is_adj=True)
      z = gen_noise(adj, flags)
      adj_mean = adj - f
      adj = adj_mean + G[:, None, None] * z
      return adj, adj_mean
    
    else:
      raise NotImplementedError(f"obj {self.obj} not yet supported.")


class NoneCorrector(Corrector):
  """An empty corrector that does nothing."""

  def __init__(self, obj, sde, score_fn, snr, scale_eps, n_steps):
    self.obj = obj
    pass

  def update_fn(self, x, adj, flags, t):
    if self.obj == 'x':
      return x, x
    elif self.obj == 'adj':
      return adj, adj
    else:
      raise NotImplementedError(f"obj {self.obj} not yet supported.")


class LangevinCorrector(Corrector):
  def __init__(self, obj, sde, score_fn, snr, scale_eps, n_steps):
    super().__init__(sde, score_fn, snr, scale_eps, n_steps)
    self.obj = obj

  def update_fn(self, x, adj, flags, t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    seps = self.scale_eps

    if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones_like(t)

    if self.obj == 'x':
      for i in range(n_steps):
        grad = score_fn(x, adj, flags, t)
        noise = gen_noise(x, flags, sym=False)
        grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        x_mean = x + step_size[:, None, None] * grad
        x = x_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * seps
      return x, x_mean

    elif self.obj == 'adj':
      for i in range(n_steps):
        grad = score_fn(x, adj, flags, t)
        noise = gen_noise(adj, flags)
        grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        adj_mean = adj + step_size[:, None, None] * grad
        adj = adj_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * seps
      return adj, adj_mean

    else:
      raise NotImplementedError(f"obj {self.obj} not yet supported")


# -------- PC sampler --------
def get_pc_sampler(sde_x, sde_adj, shape_x, shape_adj, predictor='Euler', corrector='None', 
                   snr=0.1, scale_eps=1.0, n_steps=1, 
                   probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda', strat='controlled'):

  def pc_sampler(model_x, model_adj, init_flags):

    score_fn_x = get_score_fn(sde_x, model_x, train=False, continuous=continuous)
    score_fn_adj = get_score_fn(sde_adj, model_adj, train=False, continuous=continuous)

    predictor_fn = ReverseDiffusionPredictor if predictor=='Reverse' else EulerMaruyamaPredictor 
    corrector_fn = LangevinCorrector if corrector=='Langevin' else NoneCorrector

    predictor_obj_x = predictor_fn('x', sde_x, score_fn_x, probability_flow)
    corrector_obj_x = corrector_fn('x', sde_x, score_fn_x, snr, scale_eps, n_steps)

    predictor_obj_adj = predictor_fn('adj', sde_adj, score_fn_adj, probability_flow)
    corrector_obj_adj = corrector_fn('adj', sde_adj, score_fn_adj, snr, scale_eps, n_steps)

    with torch.no_grad():
      # -------- Initial sample --------
      x = sde_x.prior_sampling(shape_x).to(device) 
      adj = sde_adj.prior_sampling_sym(shape_adj).to(device) 
      flags = init_flags
      x = mask_x(x, flags)
      adj = mask_adjs(adj, flags)
      
      # diff_steps = sde_adj.N
      diff_steps = 200
      timesteps = torch.linspace(sde_adj.T, eps, diff_steps, device=device)

      x_mid = []
      adj_mid = []
      # -------- Reverse diffusion process --------
      for i in trange(0, (diff_steps), desc = '[Sampling]', position = 1, leave=False):
        t = timesteps[i]
        vec_t = torch.ones(shape_adj[0], device=t.device) * t

        _x = x
        x, x_mean = corrector_obj_x.update_fn(x, adj, flags, vec_t)
        adj, adj_mean = corrector_obj_adj.update_fn(_x, adj, flags, vec_t)

        _x = x
        x, x_mean = predictor_obj_x.update_fn(x, adj, flags, vec_t)
        adj, adj_mean = predictor_obj_adj.update_fn(_x, adj, flags, vec_t)
        x_mid.append(x_mean.detach().clone())
        adj_mid.append(adj_mean.detach().clone())
      print(' ')
      return (x_mean if denoise else x), (adj_mean if denoise else adj), x_mid, adj_mid, diff_steps * (n_steps + 1)

  def pc_sampler_PM(model_x, model_adj, init_flags):

    score_fn_x = get_score_fn(sde_x, model_x, train=False, continuous=continuous)
    score_fn_adj = get_score_fn(sde_adj, model_adj, train=False, continuous=continuous)

    predictor_fn = ReverseDiffusionPredictor if predictor == 'Reverse' else EulerMaruyamaPredictor
    corrector_fn = LangevinCorrector if corrector == 'Langevin' else NoneCorrector

    predictor_obj_x = predictor_fn('x', sde_x, score_fn_x, probability_flow)
    corrector_obj_x = corrector_fn('x', sde_x, score_fn_x, snr, scale_eps, n_steps)

    predictor_obj_adj = predictor_fn('adj', sde_adj, score_fn_adj, probability_flow)
    corrector_obj_adj = corrector_fn('adj', sde_adj, score_fn_adj, snr, scale_eps, n_steps)

    with torch.no_grad():
      # -------- Initial sample --------
      x = sde_x.prior_sampling(shape_x).to(device)
      adj = sde_adj.prior_sampling_sym(shape_adj).to(device)
      flags = init_flags
      x = mask_x(x, flags)
      adj = mask_adjs(adj, flags)
      diff_steps = sde_adj.N
      timesteps = torch.linspace(sde_adj.T, eps, diff_steps, device=device)

      x_mid = []
      adj_mid = []
      # -------- Reverse diffusion process --------
      for i in trange(0, (diff_steps), desc='[Sampling]', position=1):
        t = timesteps[i]
        vec_t = torch.ones(shape_adj[0], device=t.device) * t

        def predict_x0_from_xt(x, adj, t):
          timestep = (t * (sde_x.N - 1) / sde_x.T).long()  # sde.N = 1000, sde.T = 1
          score_x = score_fn_x(x, adj, flags, t)
          sqrt_alpha_cumprod = sde_x.sqrt_alphas_cumprod.to(t.device)[timestep]  # \sqrt{\bar{alpha_t}}
          sqrt_1m_alpha_cumprod = sde_x.sqrt_1m_alphas_cumprod.to(t.device)[timestep]  # \sqrt{1-\bar{alpha_t}}
          x_0 = (1. / sqrt_alpha_cumprod).view(-1, 1, 1) * (
                    x + torch.square(sqrt_1m_alpha_cumprod).view(-1, 1, 1) * score_x)  # tweedie's formula
          return x_0

        def predict_a0_from_at(x, adj, t):
          timestep = (t * (sde_adj.N - 1) / sde_adj.T).long()
          score_adj = score_fn_adj(x, adj, flags, t)
          sigma = sde_adj.discrete_sigmas.to(t.device)[timestep]
          adj_0 = adj + torch.square(sigma).view(-1, 1, 1) * score_adj
          return adj_0

        _x = x
        x, x_mean = corrector_obj_x.update_fn(x, adj, flags, vec_t)
        adj, adj_mean = corrector_obj_adj.update_fn(_x, adj, flags, vec_t)

        _x = x
        x, x_mean = predictor_obj_x.update_fn(x, adj, flags, vec_t)
        adj, adj_mean = predictor_obj_adj.update_fn(_x, adj, flags, vec_t)
        if i == diff_steps - 1:
          x_mid.append(x_mean.detach().clone())
          adj_mid.append(adj_mean.detach().clone())
        else:
          x_next = predict_x0_from_xt(x, adj, t)  # Predict x_0
          a_next = predict_a0_from_at(x, adj, t)  # Predict a_0
          x_mid.append(x_next.detach().clone())
          adj_mid.append(a_next.detach().clone())

      print(' ')
      return (x_mean if denoise else x), (adj_mean if denoise else adj), x_mid, adj_mid, diff_steps * (n_steps + 1)

  def pc_sampler_controlled(model_x, model_adj, init_flags, rewardF, sample_M=20, larger_better=True):

    score_fn_x = get_score_fn(sde_x, model_x, train=False, continuous=continuous)
    score_fn_adj = get_score_fn(sde_adj, model_adj, train=False, continuous=continuous)

    predictor_fn = ReverseDiffusionPredictor if predictor=='Reverse' else EulerMaruyamaPredictor
    corrector_fn = LangevinCorrector if corrector=='Langevin' else NoneCorrector

    predictor_obj_x = predictor_fn('x', sde_x, score_fn_x, probability_flow)
    corrector_obj_x = corrector_fn('x', sde_x, score_fn_x, snr, scale_eps, n_steps)

    predictor_obj_adj = predictor_fn('adj', sde_adj, score_fn_adj, probability_flow)
    corrector_obj_adj = corrector_fn('adj', sde_adj, score_fn_adj, snr, scale_eps, n_steps)

    with torch.no_grad():
      # -------- Initial sample --------
      x = sde_x.prior_sampling(shape_x).to(device)
      adj = sde_adj.prior_sampling_sym(shape_adj).to(device)
      flags = init_flags
      x = mask_x(x, flags)
      adj = mask_adjs(adj, flags)
      diff_steps = sde_adj.N
      timesteps = torch.linspace(sde_adj.T, eps, diff_steps, device=device)

      # -------- Reverse diffusion process --------
      for i in trange(0, (diff_steps), desc = '[Sampling]', position = 1):
        t = timesteps[i]
        vec_t = torch.ones(shape_adj[0], device=t.device) * t

        def x_adj_update_fn(x, adj):
          _x = x
          x, x_mean = corrector_obj_x.update_fn(x, adj, flags.repeat(1, sample_M, 1).view(-1, flags.size(1)), vec_t.repeat(1, sample_M).view(-1))
          adj, adj_mean = corrector_obj_adj.update_fn(_x, adj, flags.repeat(1, sample_M, 1).view(-1, flags.size(1)), vec_t.repeat(1, sample_M).view(-1))

          _x = x
          x, x_mean = predictor_obj_x.update_fn(x, adj, flags.repeat(1, sample_M, 1).view(-1, flags.size(1)), vec_t.repeat(1, sample_M).view(-1))
          adj, adj_mean = predictor_obj_adj.update_fn(_x, adj, flags.repeat(1, sample_M, 1).view(-1, flags.size(1)), vec_t.repeat(1, sample_M).view(-1))
          return x_mean, adj_mean, x, adj

        # Repeat the current x and adj for batch processing
        x_batch = x.unsqueeze(1).repeat(1, sample_M, 1, 1).view(-1, x.size(1), x.size(2))
        adj_batch = adj.unsqueeze(1).repeat(1, sample_M, 1, 1).view(-1, adj.size(1), adj.size(2))

        x_mean_batch, adj_mean_batch, x_batch, adj_batch = x_adj_update_fn(x_batch, adj_batch)

        if i == diff_steps - 1:
          rewards = rewardF(x_mean_batch.detach(), adj_mean_batch.detach())
          x_list = x_mean_batch.view(-1, sample_M, x.size(1), x.size(2))
          adj_list = adj_mean_batch.view(-1, sample_M, adj.size(1), adj.size(2))
        else:
          rewards = rewardF(x_batch.detach(), adj_batch.detach())
          x_list = x_batch.view(-1, sample_M, x.size(1), x.size(2))
          adj_list = adj_batch.view(-1, sample_M, adj.size(1), adj.size(2))

        # Calculate scores using softmax
        scores = rewards.view(-1, sample_M)
        scores = torch.softmax(scores, dim=1)

        # Select the index of the highest score for each batch
        if larger_better:
          final_sample_indices = torch.argmax(scores, dim=1) # Shape [batch_size]  larger better: argmax
        else:
          final_sample_indices = torch.argmin(scores, dim=1)

        # Select the chosen samples using gathered indices
        x = x_list[torch.arange(x_list.size(0)), final_sample_indices, :, :]
        adj = adj_list[torch.arange(adj_list.size(0)), final_sample_indices, :, :]
        # x_list = []
        # adj_list = []
        # scores = []
        # for j in range(sample_M):
        #   x_mean_tmp, adj_mean_tmp, x_tmp, adj_tmp = x_adj_update_fn(x.clone(), adj.clone())
        #
        #   if i == diff_steps - 1:
        #     rew = rewardF(x_mean_tmp.detach().clone(), adj_mean_tmp.detach().clone())
        #     # if rew.size(0) != x.size(0):
        #     #   continue
        #     x_list.append(x_mean_tmp.detach().clone())
        #     adj_list.append(adj_mean_tmp.detach().clone())
        #   else:
        #     rew = rewardF(x_tmp.detach().clone(), adj_tmp.detach().clone())
        #     # if rew.size(0) != x.size(0):
        #     #   continue
        #     x_list.append(x_tmp.detach().clone())
        #     adj_list.append(adj_tmp.detach().clone())
        #   scores.append(rew)
        #
        # scores = torch.stack(scores, dim=1)
        # scores = torch.softmax(scores, dim=1)
        # # Select the index of the highest score for each batch
        # final_sample_indices = torch.argmax(scores, dim=1).squeeze()  # Shape [batch_size]   larger better: argmax
        # final_x = [x_list[final_sample_indices[j]][j, :, :] for j in range(final_sample_indices.size(0))]  # Select the chosen samples using gathered indices
        # x = torch.stack(final_x, dim=0)
        # final_adj = [adj_list[final_sample_indices[j]][j, :, :] for j in range(final_sample_indices.size(0))]  # Select the chosen samples using gathered indices
        # adj = torch.stack(final_adj, dim=0)

      print(' ')
      return x, adj, diff_steps * (n_steps + 1)

  def pc_sampler_controlled_TDS(model_x, model_adj, init_flags, rewardF, sample_M=1, larger_better=True, alpha=0.3):
    sample_M = 1 ## Important
    score_fn_x = get_score_fn(sde_x, model_x, train=False, continuous=continuous)
    score_fn_adj = get_score_fn(sde_adj, model_adj, train=False, continuous=continuous)

    predictor_fn = ReverseDiffusionPredictor if predictor=='Reverse' else EulerMaruyamaPredictor
    corrector_fn = LangevinCorrector if corrector=='Langevin' else NoneCorrector

    predictor_obj_x = predictor_fn('x', sde_x, score_fn_x, probability_flow)
    corrector_obj_x = corrector_fn('x', sde_x, score_fn_x, snr, scale_eps, n_steps)

    predictor_obj_adj = predictor_fn('adj', sde_adj, score_fn_adj, probability_flow)
    corrector_obj_adj = corrector_fn('adj', sde_adj, score_fn_adj, snr, scale_eps, n_steps)

    with torch.no_grad():
      # -------- Initial sample --------
      x = sde_x.prior_sampling(shape_x).to(device)
      adj = sde_adj.prior_sampling_sym(shape_adj).to(device)

      # x_batch = x
      # adj_batch = adj
      
      flags = init_flags
      x = mask_x(x, flags)
      adj = mask_adjs(adj, flags)
      diff_steps = sde_adj.N
      timesteps = torch.linspace(sde_adj.T, eps, diff_steps, device=device)

      # -------- Reverse diffusion process --------
      for i in trange(0, (diff_steps), desc = '[Sampling]', position = 1):
        t = timesteps[i]
        vec_t = torch.ones(shape_adj[0], device=t.device) * t

        def x_adj_update_fn(x, adj):
          _x = x
          x, x_mean = corrector_obj_x.update_fn(x, adj, flags, vec_t)
          adj, adj_mean = corrector_obj_adj.update_fn(_x, adj, flags, vec_t)

          _x = x
          x, x_mean = predictor_obj_x.update_fn(x, adj, flags, vec_t)
          adj, adj_mean = predictor_obj_adj.update_fn(_x, adj, flags, vec_t)
          return x_mean, adj_mean, x, adj

        # x_list = []
        # adj_list = []
        # scores = []

        def predict_x0_from_xt(x, adj, t): 
          timestep = (t * (sde_x.N - 1) / sde_x.T).long()  # sde.N = 1000, sde.T = 1
          score_x = score_fn_x(x, adj, flags, t)
          sqrt_alpha_cumprod = sde_x.sqrt_alphas_cumprod.to(t.device)[timestep]    # \sqrt{\bar{alpha_t}}
          sqrt_1m_alpha_cumprod = sde_x.sqrt_1m_alphas_cumprod.to(t.device)[timestep]  # \sqrt{1-\bar{alpha_t}} 
          x_0 = (1. / sqrt_alpha_cumprod).view(-1, 1, 1)*(x + torch.square(sqrt_1m_alpha_cumprod).view(-1, 1, 1)* score_x)  # tweedie's formula
          return x_0
        
        def predict_a0_from_at(x, adj, t):
          timestep = (t * (sde_adj.N - 1) / sde_adj.T).long()
          score_adj = score_fn_adj(x, adj, flags, t)
          sigma = sde_adj.discrete_sigmas.to(t.device)[timestep] 
          adj_0 = adj + torch.square(sigma).view(-1, 1, 1) * score_adj
          return adj_0

        # for j in range(sample_M):
        #   x_mean_tmp, adj_mean_tmp, x_tmp, adj_tmp = x_adj_update_fn(x.clone(), adj.clone())
        #
        #   if i == diff_steps - 1:
        #     x_list.append(x_mean_tmp.detach().clone())
        #     adj_list.append(adj_mean_tmp.detach().clone())
        #     x_next = x_mean_tmp.detach().clone() #predict_x0_from_xt()
        #     a_next = adj_mean_tmp.detach().clone() #predict_x0_from_xt()
        #   else:
        #     x_list.append(x_tmp.detach().clone())
        #     adj_list.append(adj_tmp.detach().clone())
        #     x_next = predict_x0_from_xt(x_tmp, adj_tmp, t) # Predict x_0
        #     a_next = predict_a0_from_at(x_tmp, adj_tmp, t) # Predict a_0
        #     x_next = x_next.detach().clone()
        #     a_next = a_next.detach().clone()
        #   rew =  rewardF(x_next, a_next)
        #   scores.append(rew)
        #
        # scores = torch.stack(scores, dim=1)
        # scores = torch.softmax(scores, dim=1)
        # # Select the index of the highest score for each batch
        # final_sample_indices = torch.argmax(scores, dim=1).squeeze()  # Shape [batch_size]   larger better: argmax
        # final_x = [x_list[final_sample_indices[j]][j, :, :] for j in range(final_sample_indices.size(0))]  # Select the chosen samples using gathered indices
        # x = torch.stack(final_x, dim=0)
        # final_adj = [adj_list[final_sample_indices[j]][j, :, :] for j in range(final_sample_indices.size(0))]  # Select the chosen samples using gathered indices
        # adj = torch.stack(final_adj, dim=0)

        # Repeat the current x and adj for batch processing
        
        x_batch = x
        adj_batch = adj

        ### Get Current Reward 
        '''
        Calcualte exp(v_{t}(x_{t)/alpha)
        '''
        x_next_batch = predict_x0_from_xt(x_batch, adj_batch, t)  # Predict x_0
        adj_next_batch = predict_a0_from_at(x_batch, adj_batch, t)  # Predict a_0
        reward_den = rewardF(x_next_batch.detach(), adj_next_batch.detach())
         
        ### Go to the next step
        '''
        Calcualte exp(v_{t-1}(x_{t-1})/alpha)
        '''
        x_mean_batch, adj_mean_batch, x_batch, adj_batch = x_adj_update_fn(x_batch, adj_batch)

        if i == diff_steps - 1:
          reward_num = rewardF(x_mean_batch.detach(), adj_mean_batch.detach())
        else:
          x_next_batch = predict_x0_from_xt(x_batch, adj_batch, t)  # Predict x_0
          adj_next_batch = predict_a0_from_at(x_batch, adj_batch, t)  # Predict a_0
          reward_num = rewardF(x_next_batch.detach(), adj_next_batch.detach())

        ratio = torch.exp(1.0/alpha * (reward_num - reward_den))
        # print reward_num - reward_den tensor in a line with .4f each
        # print((reward_num - reward_den).detach().cpu().numpy())
        ratio = ratio.detach().cpu().numpy()
        ratio = ratio[:,0]
        # print(ratio / ratio.sum())
        final_sample_indices = np.random.choice(reward_num.shape[0], reward_num.shape[0], p=ratio/ratio.sum())
        # print(reward_num[final_sample_indices].detach().cpu().numpy())
        if i == diff_steps - 1: 
          x = x_mean_batch[final_sample_indices]
          adj = adj_mean_batch[final_sample_indices]
        else:
          x = x_batch[final_sample_indices]
          adj = adj_batch[final_sample_indices]

      print(' ')
      return x, adj, diff_steps * (n_steps + 1) 
        
  def pc_sampler_controlled_PM(model_x, model_adj, init_flags, rewardF, sample_M=20, larger_better=True):

    score_fn_x = get_score_fn(sde_x, model_x, train=False, continuous=continuous)
    score_fn_adj = get_score_fn(sde_adj, model_adj, train=False, continuous=continuous)

    predictor_fn = ReverseDiffusionPredictor if predictor=='Reverse' else EulerMaruyamaPredictor
    corrector_fn = LangevinCorrector if corrector=='Langevin' else NoneCorrector

    predictor_obj_x = predictor_fn('x', sde_x, score_fn_x, probability_flow)
    corrector_obj_x = corrector_fn('x', sde_x, score_fn_x, snr, scale_eps, n_steps)

    predictor_obj_adj = predictor_fn('adj', sde_adj, score_fn_adj, probability_flow)
    corrector_obj_adj = corrector_fn('adj', sde_adj, score_fn_adj, snr, scale_eps, n_steps)

    with torch.no_grad():
      # -------- Initial sample --------
      x = sde_x.prior_sampling(shape_x).to(device)
      adj = sde_adj.prior_sampling_sym(shape_adj).to(device)
      flags = init_flags
      x = mask_x(x, flags)
      adj = mask_adjs(adj, flags)
      
      # diff_steps = sde_adj.N
      diff_steps = 200
      
      timesteps = torch.linspace(sde_adj.T, eps, diff_steps, device=device)

      # -------- Reverse diffusion process --------
      for i in trange(0, (diff_steps), desc = '[Sampling]', position = 1):
        t = timesteps[i]
        vec_t = torch.ones(shape_adj[0], device=t.device) * t

        def x_adj_update_fn(x, adj):
          _x = x
          x, x_mean = corrector_obj_x.update_fn(x, adj, flags.repeat(1, sample_M, 1).view(-1, flags.size(1)),
                                                vec_t.repeat(1, sample_M).view(-1))
          adj, adj_mean = corrector_obj_adj.update_fn(_x, adj, flags.repeat(1, sample_M, 1).view(-1, flags.size(1)),
                                                      vec_t.repeat(1, sample_M).view(-1))

          _x = x
          x, x_mean = predictor_obj_x.update_fn(x, adj, flags.repeat(1, sample_M, 1).view(-1, flags.size(1)),
                                                vec_t.repeat(1, sample_M).view(-1))
          adj, adj_mean = predictor_obj_adj.update_fn(_x, adj, flags.repeat(1, sample_M, 1).view(-1, flags.size(1)),
                                                      vec_t.repeat(1, sample_M).view(-1))
          return x_mean, adj_mean, x, adj

        # x_list = []
        # adj_list = []
        # scores = []

        def predict_x0_from_xt(x, adj, t): 
          timestep = (t * (sde_x.N - 1) / sde_x.T).long()  # sde.N = 1000, sde.T = 1
          score_x = score_fn_x(x, adj, flags.repeat(1, sample_M, 1).view(-1, flags.size(1)), t)
          sqrt_alpha_cumprod = sde_x.sqrt_alphas_cumprod.to(t.device)[timestep]    # \sqrt{\bar{alpha_t}}
          sqrt_1m_alpha_cumprod = sde_x.sqrt_1m_alphas_cumprod.to(t.device)[timestep]  # \sqrt{1-\bar{alpha_t}} 
          x_0 = (1. / sqrt_alpha_cumprod).view(-1, 1, 1)*(x + torch.square(sqrt_1m_alpha_cumprod).view(-1, 1, 1)* score_x)  # tweedie's formula
          return x_0
        
        def predict_a0_from_at(x, adj, t):
          timestep = (t * (sde_adj.N - 1) / sde_adj.T).long()
          score_adj = score_fn_adj(x, adj, flags.repeat(1, sample_M, 1).view(-1, flags.size(1)), t)
          sigma = sde_adj.discrete_sigmas.to(t.device)[timestep] 
          adj_0 = adj + torch.square(sigma).view(-1, 1, 1) * score_adj
          return adj_0

        # Repeat the current x and adj for batch processing
        x_batch = x.unsqueeze(1).repeat(1, sample_M, 1, 1).view(-1, x.size(1), x.size(2))
        adj_batch = adj.unsqueeze(1).repeat(1, sample_M, 1, 1).view(-1, adj.size(1), adj.size(2))

        x_mean_batch, adj_mean_batch, x_batch, adj_batch = x_adj_update_fn(x_batch, adj_batch)

        if i == diff_steps - 1:
          x_list = x_mean_batch.view(-1, sample_M, x.size(1), x.size(2))
          adj_list = adj_mean_batch.view(-1, sample_M, adj.size(1), adj.size(2))
          rewards = rewardF(x_mean_batch.detach(), adj_mean_batch.detach())
        else:
          x_list = x_batch.view(-1, sample_M, x.size(1), x.size(2))
          adj_list = adj_batch.view(-1, sample_M, adj.size(1), adj.size(2))
          x_next_batch = predict_x0_from_xt(x_batch, adj_batch, t)  # Predict x_0
          adj_next_batch = predict_a0_from_at(x_batch, adj_batch, t)  # Predict a_0
          rewards = rewardF(x_next_batch.detach(), adj_next_batch.detach())

        # Calculate scores using softmax
        scores = rewards.view(-1, sample_M)
        scores = torch.softmax(scores, dim=1)

        # Select the index of the highest score for each batch
        if larger_better:
          final_sample_indices = torch.argmax(scores, dim=1)  # Shape [batch_size]  larger better: argmax
        else:
          final_sample_indices = torch.argmin(scores, dim=1)

        # Select the chosen samples using gathered indices
        x = x_list[torch.arange(x_list.size(0)), final_sample_indices, :, :]
        adj = adj_list[torch.arange(adj_list.size(0)), final_sample_indices, :, :]

      print(' ')
      return x, adj, diff_steps * (n_steps + 1)
    
  if strat=='controlled':
    return pc_sampler_controlled
  elif strat=='controlled_tw':
    return pc_sampler_controlled_PM
  elif strat=='tds':
    return pc_sampler_controlled_TDS
  elif strat=='PM':
    return pc_sampler_PM
  else:
    return pc_sampler

def get_pc_sampler_refine(K, sde_x, sde_adj, shape_x, shape_adj, predictor='Euler', corrector='None', 
                   snr=0.1, scale_eps=1.0, n_steps=1, 
                   probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda', strat='controlled'):

  def pc_sampler(model_x, model_adj, init_flags):

    score_fn_x = get_score_fn(sde_x, model_x, train=False, continuous=continuous)
    score_fn_adj = get_score_fn(sde_adj, model_adj, train=False, continuous=continuous)

    predictor_fn = ReverseDiffusionPredictor if predictor=='Reverse' else EulerMaruyamaPredictor 
    corrector_fn = LangevinCorrector if corrector=='Langevin' else NoneCorrector

    predictor_obj_x = predictor_fn('x', sde_x, score_fn_x, probability_flow)
    corrector_obj_x = corrector_fn('x', sde_x, score_fn_x, snr, scale_eps, n_steps)

    predictor_obj_adj = predictor_fn('adj', sde_adj, score_fn_adj, probability_flow)
    corrector_obj_adj = corrector_fn('adj', sde_adj, score_fn_adj, snr, scale_eps, n_steps)

    with torch.no_grad():
      # -------- Initial sample --------
      x = sde_x.prior_sampling(shape_x).to(device) 
      adj = sde_adj.prior_sampling_sym(shape_adj).to(device) 
      flags = init_flags
      x = mask_x(x, flags)
      adj = mask_adjs(adj, flags)
      
      # diff_steps = sde_adj.N
      diff_steps = 200
      timesteps = torch.linspace(sde_adj.T, eps, diff_steps, device=device)

      x_mid = []
      adj_mid = []
      # -------- Reverse diffusion process --------
      for i in trange(0, (diff_steps), desc = '[Reward Optimization]', position = 1, leave=False):
        t = timesteps[i]
        vec_t = torch.ones(shape_adj[0], device=t.device) * t

        _x = x
        x, x_mean = corrector_obj_x.update_fn(x, adj, flags, vec_t)
        adj, adj_mean = corrector_obj_adj.update_fn(_x, adj, flags, vec_t)

        _x = x
        x, x_mean = predictor_obj_x.update_fn(x, adj, flags, vec_t)
        adj, adj_mean = predictor_obj_adj.update_fn(_x, adj, flags, vec_t)
        x_mid.append(x_mean.detach().clone())
        adj_mid.append(adj_mean.detach().clone())
      print(' ')
      return (x_mean if denoise else x), (adj_mean if denoise else adj), x_mid, adj_mid, diff_steps * (n_steps + 1)

  def pc_sampler_PM(model_x, model_adj, init_flags):

    score_fn_x = get_score_fn(sde_x, model_x, train=False, continuous=continuous)
    score_fn_adj = get_score_fn(sde_adj, model_adj, train=False, continuous=continuous)

    predictor_fn = ReverseDiffusionPredictor if predictor == 'Reverse' else EulerMaruyamaPredictor
    corrector_fn = LangevinCorrector if corrector == 'Langevin' else NoneCorrector

    predictor_obj_x = predictor_fn('x', sde_x, score_fn_x, probability_flow)
    corrector_obj_x = corrector_fn('x', sde_x, score_fn_x, snr, scale_eps, n_steps)

    predictor_obj_adj = predictor_fn('adj', sde_adj, score_fn_adj, probability_flow)
    corrector_obj_adj = corrector_fn('adj', sde_adj, score_fn_adj, snr, scale_eps, n_steps)

    with torch.no_grad():
      # -------- Initial sample --------
      x = sde_x.prior_sampling(shape_x).to(device)
      adj = sde_adj.prior_sampling_sym(shape_adj).to(device)
      flags = init_flags
      x = mask_x(x, flags)
      adj = mask_adjs(adj, flags)
      diff_steps = sde_adj.N
      timesteps = torch.linspace(sde_adj.T, eps, diff_steps, device=device)

      x_mid = []
      adj_mid = []
      # -------- Reverse diffusion process --------
      for i in trange(0, (diff_steps), desc='[Sampling]', position=1):
        t = timesteps[i]
        vec_t = torch.ones(shape_adj[0], device=t.device) * t

        def predict_x0_from_xt(x, adj, t):
          timestep = (t * (sde_x.N - 1) / sde_x.T).long()  # sde.N = 1000, sde.T = 1
          score_x = score_fn_x(x, adj, flags, t)
          sqrt_alpha_cumprod = sde_x.sqrt_alphas_cumprod.to(t.device)[timestep]  # \sqrt{\bar{alpha_t}}
          sqrt_1m_alpha_cumprod = sde_x.sqrt_1m_alphas_cumprod.to(t.device)[timestep]  # \sqrt{1-\bar{alpha_t}}
          x_0 = (1. / sqrt_alpha_cumprod).view(-1, 1, 1) * (
                    x + torch.square(sqrt_1m_alpha_cumprod).view(-1, 1, 1) * score_x)  # tweedie's formula
          return x_0

        def predict_a0_from_at(x, adj, t):
          timestep = (t * (sde_adj.N - 1) / sde_adj.T).long()
          score_adj = score_fn_adj(x, adj, flags, t)
          sigma = sde_adj.discrete_sigmas.to(t.device)[timestep]
          adj_0 = adj + torch.square(sigma).view(-1, 1, 1) * score_adj
          return adj_0

        _x = x
        x, x_mean = corrector_obj_x.update_fn(x, adj, flags, vec_t)
        adj, adj_mean = corrector_obj_adj.update_fn(_x, adj, flags, vec_t)

        _x = x
        x, x_mean = predictor_obj_x.update_fn(x, adj, flags, vec_t)
        adj, adj_mean = predictor_obj_adj.update_fn(_x, adj, flags, vec_t)
        if i == diff_steps - 1:
          x_mid.append(x_mean.detach().clone())
          adj_mid.append(adj_mean.detach().clone())
        else:
          x_next = predict_x0_from_xt(x, adj, t)  # Predict x_0
          a_next = predict_a0_from_at(x, adj, t)  # Predict a_0
          x_mid.append(x_next.detach().clone())
          adj_mid.append(a_next.detach().clone())

      print(' ')
      return (x_mean if denoise else x), (adj_mean if denoise else adj), x_mid, adj_mid, diff_steps * (n_steps + 1)

  def pc_sampler_controlled(model_x, model_adj, init_flags, rewardF, sample_M=20, larger_better=True):

    score_fn_x = get_score_fn(sde_x, model_x, train=False, continuous=continuous)
    score_fn_adj = get_score_fn(sde_adj, model_adj, train=False, continuous=continuous)

    predictor_fn = ReverseDiffusionPredictor if predictor=='Reverse' else EulerMaruyamaPredictor
    corrector_fn = LangevinCorrector if corrector=='Langevin' else NoneCorrector

    predictor_obj_x = predictor_fn('x', sde_x, score_fn_x, probability_flow)
    corrector_obj_x = corrector_fn('x', sde_x, score_fn_x, snr, scale_eps, n_steps)

    predictor_obj_adj = predictor_fn('adj', sde_adj, score_fn_adj, probability_flow)
    corrector_obj_adj = corrector_fn('adj', sde_adj, score_fn_adj, snr, scale_eps, n_steps)

    with torch.no_grad():
      # -------- Initial sample --------
      x = sde_x.prior_sampling(shape_x).to(device)
      adj = sde_adj.prior_sampling_sym(shape_adj).to(device)
      flags = init_flags
      x = mask_x(x, flags)
      adj = mask_adjs(adj, flags)
      diff_steps = sde_adj.N
      timesteps = torch.linspace(sde_adj.T, eps, diff_steps, device=device)

      # -------- Reverse diffusion process --------
      for i in trange(0, (diff_steps), desc = '[Sampling]', position = 1):
        t = timesteps[i]
        vec_t = torch.ones(shape_adj[0], device=t.device) * t

        def x_adj_update_fn(x, adj):
          _x = x
          x, x_mean = corrector_obj_x.update_fn(x, adj, flags.repeat(1, sample_M, 1).view(-1, flags.size(1)), vec_t.repeat(1, sample_M).view(-1))
          adj, adj_mean = corrector_obj_adj.update_fn(_x, adj, flags.repeat(1, sample_M, 1).view(-1, flags.size(1)), vec_t.repeat(1, sample_M).view(-1))

          _x = x
          x, x_mean = predictor_obj_x.update_fn(x, adj, flags.repeat(1, sample_M, 1).view(-1, flags.size(1)), vec_t.repeat(1, sample_M).view(-1))
          adj, adj_mean = predictor_obj_adj.update_fn(_x, adj, flags.repeat(1, sample_M, 1).view(-1, flags.size(1)), vec_t.repeat(1, sample_M).view(-1))
          return x_mean, adj_mean, x, adj

        # Repeat the current x and adj for batch processing
        x_batch = x.unsqueeze(1).repeat(1, sample_M, 1, 1).view(-1, x.size(1), x.size(2))
        adj_batch = adj.unsqueeze(1).repeat(1, sample_M, 1, 1).view(-1, adj.size(1), adj.size(2))

        x_mean_batch, adj_mean_batch, x_batch, adj_batch = x_adj_update_fn(x_batch, adj_batch)

        if i == diff_steps - 1:
          rewards = rewardF(x_mean_batch.detach(), adj_mean_batch.detach())
          x_list = x_mean_batch.view(-1, sample_M, x.size(1), x.size(2))
          adj_list = adj_mean_batch.view(-1, sample_M, adj.size(1), adj.size(2))
        else:
          rewards = rewardF(x_batch.detach(), adj_batch.detach())
          x_list = x_batch.view(-1, sample_M, x.size(1), x.size(2))
          adj_list = adj_batch.view(-1, sample_M, adj.size(1), adj.size(2))

        # Calculate scores using softmax
        scores = rewards.view(-1, sample_M)
        scores = torch.softmax(scores, dim=1)

        # Select the index of the highest score for each batch
        if larger_better:
          final_sample_indices = torch.argmax(scores, dim=1) # Shape [batch_size]  larger better: argmax
        else:
          final_sample_indices = torch.argmin(scores, dim=1)

        # Select the chosen samples using gathered indices
        x = x_list[torch.arange(x_list.size(0)), final_sample_indices, :, :]
        adj = adj_list[torch.arange(adj_list.size(0)), final_sample_indices, :, :]

      print(' ')
      return x, adj, diff_steps * (n_steps + 1)

  def pc_sampler_controlled_TDS(model_x, model_adj, init_flags, rewardF, sample_M=1, larger_better=True, alpha=0.3):
    sample_M = 1 ## Important
    score_fn_x = get_score_fn(sde_x, model_x, train=False, continuous=continuous)
    score_fn_adj = get_score_fn(sde_adj, model_adj, train=False, continuous=continuous)

    predictor_fn = ReverseDiffusionPredictor if predictor=='Reverse' else EulerMaruyamaPredictor
    corrector_fn = LangevinCorrector if corrector=='Langevin' else NoneCorrector

    predictor_obj_x = predictor_fn('x', sde_x, score_fn_x, probability_flow)
    corrector_obj_x = corrector_fn('x', sde_x, score_fn_x, snr, scale_eps, n_steps)

    predictor_obj_adj = predictor_fn('adj', sde_adj, score_fn_adj, probability_flow)
    corrector_obj_adj = corrector_fn('adj', sde_adj, score_fn_adj, snr, scale_eps, n_steps)

    with torch.no_grad():
      # -------- Initial sample --------
      x = sde_x.prior_sampling(shape_x).to(device)
      adj = sde_adj.prior_sampling_sym(shape_adj).to(device)

      # x_batch = x
      # adj_batch = adj
      
      flags = init_flags
      x = mask_x(x, flags)
      adj = mask_adjs(adj, flags)
      diff_steps = sde_adj.N
      timesteps = torch.linspace(sde_adj.T, eps, diff_steps, device=device)

      # -------- Reverse diffusion process --------
      for i in trange(0, (diff_steps), desc = '[Sampling]', position = 1):
        t = timesteps[i]
        vec_t = torch.ones(shape_adj[0], device=t.device) * t

        def x_adj_update_fn(x, adj):
          _x = x
          x, x_mean = corrector_obj_x.update_fn(x, adj, flags, vec_t)
          adj, adj_mean = corrector_obj_adj.update_fn(_x, adj, flags, vec_t)

          _x = x
          x, x_mean = predictor_obj_x.update_fn(x, adj, flags, vec_t)
          adj, adj_mean = predictor_obj_adj.update_fn(_x, adj, flags, vec_t)
          return x_mean, adj_mean, x, adj

        # x_list = []
        # adj_list = []
        # scores = []

        def predict_x0_from_xt(x, adj, t): 
          timestep = (t * (sde_x.N - 1) / sde_x.T).long()  # sde.N = 1000, sde.T = 1
          score_x = score_fn_x(x, adj, flags, t)
          sqrt_alpha_cumprod = sde_x.sqrt_alphas_cumprod.to(t.device)[timestep]    # \sqrt{\bar{alpha_t}}
          sqrt_1m_alpha_cumprod = sde_x.sqrt_1m_alphas_cumprod.to(t.device)[timestep]  # \sqrt{1-\bar{alpha_t}} 
          x_0 = (1. / sqrt_alpha_cumprod).view(-1, 1, 1)*(x + torch.square(sqrt_1m_alpha_cumprod).view(-1, 1, 1)* score_x)  # tweedie's formula
          return x_0
        
        def predict_a0_from_at(x, adj, t):
          timestep = (t * (sde_adj.N - 1) / sde_adj.T).long()
          score_adj = score_fn_adj(x, adj, flags, t)
          sigma = sde_adj.discrete_sigmas.to(t.device)[timestep] 
          adj_0 = adj + torch.square(sigma).view(-1, 1, 1) * score_adj
          return adj_0

        x_batch = x
        adj_batch = adj

        ### Get Current Reward 
        '''
        Calcualte exp(v_{t}(x_{t)/alpha)
        '''
        x_next_batch = predict_x0_from_xt(x_batch, adj_batch, t)  # Predict x_0
        adj_next_batch = predict_a0_from_at(x_batch, adj_batch, t)  # Predict a_0
        reward_den = rewardF(x_next_batch.detach(), adj_next_batch.detach())
         
        ### Go to the next step
        '''
        Calcualte exp(v_{t-1}(x_{t-1})/alpha)
        '''
        x_mean_batch, adj_mean_batch, x_batch, adj_batch = x_adj_update_fn(x_batch, adj_batch)

        if i == diff_steps - 1:
          reward_num = rewardF(x_mean_batch.detach(), adj_mean_batch.detach())
        else:
          x_next_batch = predict_x0_from_xt(x_batch, adj_batch, t)  # Predict x_0
          adj_next_batch = predict_a0_from_at(x_batch, adj_batch, t)  # Predict a_0
          reward_num = rewardF(x_next_batch.detach(), adj_next_batch.detach())

        ratio = torch.exp(1.0/alpha * (reward_num - reward_den))
        # print reward_num - reward_den tensor in a line with .4f each
        # print((reward_num - reward_den).detach().cpu().numpy())
        ratio = ratio.detach().cpu().numpy()
        ratio = ratio[:,0]
        # print(ratio / ratio.sum())
        final_sample_indices = np.random.choice(reward_num.shape[0], reward_num.shape[0], p=ratio/ratio.sum())
        # print(reward_num[final_sample_indices].detach().cpu().numpy())
        if i == diff_steps - 1: 
          x = x_mean_batch[final_sample_indices]
          adj = adj_mean_batch[final_sample_indices]
        else:
          x = x_batch[final_sample_indices]
          adj = adj_batch[final_sample_indices]

      print(' ')
      return x, adj, diff_steps * (n_steps + 1) 
        
  def pc_sampler_controlled_PM(x_s, adj_s, flags, model_x, model_adj, init_flags, rewardF, sample_M=20, larger_better=True):

    score_fn_x = get_score_fn(sde_x, model_x, train=False, continuous=continuous)
    score_fn_adj = get_score_fn(sde_adj, model_adj, train=False, continuous=continuous)

    predictor_fn = ReverseDiffusionPredictor if predictor=='Reverse' else EulerMaruyamaPredictor
    corrector_fn = LangevinCorrector if corrector=='Langevin' else NoneCorrector

    predictor_obj_x = predictor_fn('x', sde_x, score_fn_x, probability_flow)
    corrector_obj_x = corrector_fn('x', sde_x, score_fn_x, snr, scale_eps, n_steps)

    predictor_obj_adj = predictor_fn('adj', sde_adj, score_fn_adj, probability_flow)
    corrector_obj_adj = corrector_fn('adj', sde_adj, score_fn_adj, snr, scale_eps, n_steps)

    with torch.no_grad():
      # -------- Initial sample --------
      # x = sde_x.prior_sampling(shape_x).to(device)
      # adj = sde_adj.prior_sampling_sym(shape_adj).to(device)
      # flags = init_flags
      
      x = mask_x(x_s, flags)
      adj = mask_adjs(adj_s, flags)
      
      diff_steps = sde_adj.N
      # diff_steps = 200
      
      timesteps = torch.linspace(sde_adj.T, eps, diff_steps, device=device)

      # -------- Reverse diffusion process --------
      for i in trange(diff_steps-1-K, diff_steps, desc = '[Reward Optimization]', position = 1):
        t = timesteps[i]
        vec_t = torch.ones(shape_adj[0], device=t.device) * t

        def x_adj_update_fn(x, adj):
          _x = x
          x, x_mean = corrector_obj_x.update_fn(x, adj, flags.repeat(1, sample_M, 1).view(-1, flags.size(1)),
                                                vec_t.repeat(1, sample_M).view(-1))
          adj, adj_mean = corrector_obj_adj.update_fn(_x, adj, flags.repeat(1, sample_M, 1).view(-1, flags.size(1)),
                                                      vec_t.repeat(1, sample_M).view(-1))

          _x = x
          x, x_mean = predictor_obj_x.update_fn(x, adj, flags.repeat(1, sample_M, 1).view(-1, flags.size(1)),
                                                vec_t.repeat(1, sample_M).view(-1))
          adj, adj_mean = predictor_obj_adj.update_fn(_x, adj, flags.repeat(1, sample_M, 1).view(-1, flags.size(1)),
                                                      vec_t.repeat(1, sample_M).view(-1))
          return x_mean, adj_mean, x, adj

        # x_list = []
        # adj_list = []
        # scores = []

        def predict_x0_from_xt(x, adj, t): 
          timestep = (t * (sde_x.N - 1) / sde_x.T).long()  # sde.N = 1000, sde.T = 1
          score_x = score_fn_x(x, adj, flags.repeat(1, sample_M, 1).view(-1, flags.size(1)), t)
          sqrt_alpha_cumprod = sde_x.sqrt_alphas_cumprod.to(t.device)[timestep]    # \sqrt{\bar{alpha_t}}
          sqrt_1m_alpha_cumprod = sde_x.sqrt_1m_alphas_cumprod.to(t.device)[timestep]  # \sqrt{1-\bar{alpha_t}} 
          x_0 = (1. / sqrt_alpha_cumprod).view(-1, 1, 1)*(x + torch.square(sqrt_1m_alpha_cumprod).view(-1, 1, 1)* score_x)  # tweedie's formula
          return x_0
        
        def predict_a0_from_at(x, adj, t):
          timestep = (t * (sde_adj.N - 1) / sde_adj.T).long()
          score_adj = score_fn_adj(x, adj, flags.repeat(1, sample_M, 1).view(-1, flags.size(1)), t)
          sigma = sde_adj.discrete_sigmas.to(t.device)[timestep] 
          adj_0 = adj + torch.square(sigma).view(-1, 1, 1) * score_adj
          return adj_0

        # Repeat the current x and adj for batch processing
        x_batch = x.unsqueeze(1).repeat(1, sample_M, 1, 1).view(-1, x.size(1), x.size(2))
        adj_batch = adj.unsqueeze(1).repeat(1, sample_M, 1, 1).view(-1, adj.size(1), adj.size(2))

        x_mean_batch, adj_mean_batch, x_batch, adj_batch = x_adj_update_fn(x_batch, adj_batch)

        if i == diff_steps - 1:
          x_list = x_mean_batch.view(-1, sample_M, x.size(1), x.size(2))
          adj_list = adj_mean_batch.view(-1, sample_M, adj.size(1), adj.size(2))
          rewards = rewardF(x_mean_batch.detach(), adj_mean_batch.detach())
        else:
          x_list = x_batch.view(-1, sample_M, x.size(1), x.size(2))
          adj_list = adj_batch.view(-1, sample_M, adj.size(1), adj.size(2))
          x_next_batch = predict_x0_from_xt(x_batch, adj_batch, t)  # Predict x_0
          adj_next_batch = predict_a0_from_at(x_batch, adj_batch, t)  # Predict a_0
          rewards = rewardF(x_next_batch.detach(), adj_next_batch.detach())

        # Calculate scores using softmax
        scores = rewards.view(-1, sample_M)
        scores = torch.softmax(scores, dim=1)

        # Select the index of the highest score for each batch
        if larger_better:
          final_sample_indices = torch.argmax(scores, dim=1)  # Shape [batch_size]  larger better: argmax
        else:
          final_sample_indices = torch.argmin(scores, dim=1)

        # Select the chosen samples using gathered indices
        x = x_list[torch.arange(x_list.size(0)), final_sample_indices, :, :]
        adj = adj_list[torch.arange(adj_list.size(0)), final_sample_indices, :, :]

      print(' ')
      return x, adj, diff_steps * (n_steps + 1)
    
  if strat=='controlled':
    return pc_sampler_controlled
  elif strat=='controlled_tw':
    return pc_sampler_controlled_PM
  elif strat=='tds':
    return pc_sampler_controlled_TDS
  elif strat=='PM':
    return pc_sampler_PM
  else:
    return pc_sampler


def reward_PM(x, a):
  pass


# -------- S4 solver --------
def S4_solver(sde_x, sde_adj, shape_x, shape_adj, predictor='None', corrector='None', 
                        snr=0.1, scale_eps=1.0, n_steps=1, 
                        probability_flow=False, continuous=False,
                        denoise=True, eps=1e-3, device='cuda'):

  def s4_solver(model_x, model_adj, init_flags):

    score_fn_x = get_score_fn(sde_x, model_x, train=False, continuous=continuous)
    score_fn_adj = get_score_fn(sde_adj, model_adj, train=False, continuous=continuous)

    with torch.no_grad():
      # -------- Initial sample --------
      x = sde_x.prior_sampling(shape_x).to(device) 
      adj = sde_adj.prior_sampling_sym(shape_adj).to(device) 
      flags = init_flags
      x = mask_x(x, flags)
      adj = mask_adjs(adj, flags)
      diff_steps = sde_adj.N
      timesteps = torch.linspace(sde_adj.T, eps, diff_steps, device=device)
      dt = -1. / diff_steps

      # -------- Rverse diffusion process --------
      for i in trange(0, (diff_steps), desc = '[Sampling]', position = 1, leave=False):
        t = timesteps[i]
        vec_t = torch.ones(shape_adj[0], device=t.device) * t
        vec_dt = torch.ones(shape_adj[0], device=t.device) * (dt/2) 

        # -------- Score computation --------
        score_x = score_fn_x(x, adj, flags, vec_t)
        score_adj = score_fn_adj(x, adj, flags, vec_t)

        Sdrift_x = -sde_x.sde(x, vec_t)[1][:, None, None] ** 2 * score_x
        Sdrift_adj  = -sde_adj.sde(adj, vec_t)[1][:, None, None] ** 2 * score_adj

        # -------- Correction step --------
        timestep = (vec_t * (sde_x.N - 1) / sde_x.T).long()

        noise = gen_noise(x, flags, sym=False)
        grad_norm = torch.norm(score_x.reshape(score_x.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        if isinstance(sde_x, VPSDE):
          alpha = sde_x.alphas.to(vec_t.device)[timestep]
        else:
          alpha = torch.ones_like(vec_t)
      
        step_size = (snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        x_mean = x + step_size[:, None, None] * score_x
        x = x_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * scale_eps

        noise = gen_noise(adj, flags)
        grad_norm = torch.norm(score_adj.reshape(score_adj.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        if isinstance(sde_adj, VPSDE):
          alpha = sde_adj.alphas.to(vec_t.device)[timestep] # VP
        else:
          alpha = torch.ones_like(vec_t) # VE
        step_size = (snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        adj_mean = adj + step_size[:, None, None] * score_adj
        adj = adj_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * scale_eps

        # -------- Prediction step --------
        x_mean = x
        adj_mean = adj
        mu_x, sigma_x = sde_x.transition(x, vec_t, vec_dt)
        mu_adj, sigma_adj = sde_adj.transition(adj, vec_t, vec_dt) 
        x = mu_x + sigma_x[:, None, None] * gen_noise(x, flags, sym=False)
        adj = mu_adj + sigma_adj[:, None, None] * gen_noise(adj, flags)
        
        x = x + Sdrift_x * dt
        adj = adj + Sdrift_adj * dt

        mu_x, sigma_x = sde_x.transition(x, vec_t + vec_dt, vec_dt) 
        mu_adj, sigma_adj = sde_adj.transition(adj, vec_t + vec_dt, vec_dt) 
        x = mu_x + sigma_x[:, None, None] * gen_noise(x, flags, sym=False)
        adj = mu_adj + sigma_adj[:, None, None] * gen_noise(adj, flags)

        x_mean = mu_x
        adj_mean = mu_adj
      print(' ')
      return (x_mean if denoise else x), (adj_mean if denoise else adj), 0
  return s4_solver
