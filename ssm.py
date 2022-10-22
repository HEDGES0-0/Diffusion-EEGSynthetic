import math
import os

import numpy as np
import opt_einsum as oe
import torch
import torch.linalg as LA
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchaudio.functional import highpass_biquad, lowpass_biquad, resample


class ContinuousDDPMNoiseScheduler():
    """
    ref:
    Meng, C., He, Y., Song, Y., Song, J., Wu, J., Zhu, J.-Y., & Ermon, S. (2022). 
    SDEDIT: GUIDED IMAGE SYNTHESIS AND EDITING WITH STOCHASTIC DIFFERENTIAL EQUATIONS. 33.
    P21 Algorithm 4
    Li, C., Zhu, J., & Zhang, B. (2022). 
    ANALYTIC-DPM: AN ANALYTIC ESTIMATE OF THE OPTIMAL REVERSE VARIANCE IN DIFFUSION PROB- ABILISTIC MODELS. 39.
    P22
    Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). 
    Score-Based Generative Modeling through Stochastic Differential Equations. ArXiv:2011.13456 [Cs, Stat]. http://arxiv.org/abs/2011.13456
    P15
    """

    def __init__(self, beta_max=20, beta_min=0.1):
        self.beta_max = beta_max
        self.beta_min = beta_min
    
    def ALPHA(self, t):
        # x_t = sqrt(ALPHT_t) * x + sqrt(BETA_t) * z, z ~ N(0, 1)
        if isinstance(t, torch.Tensor):
            return torch.exp(-(self.beta_max - self.beta_min) * t**2 / 2 - t * self.beta_min)
        else:
            return np.exp(-(self.beta_max - self.beta_min) * t**2 / 2 - t * self.beta_min)

    def BETA(self, t):
        return 1 - self.ALPHA(t)

    def _beta(self, t):
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def G(self, t):
        if isinstance(t, torch.Tensor):
            return torch.sqrt(self._beta(t))
        else:
            return np.sqrt(self._beta(t)) 
    
    def F(self, t):
        return - self._beta(t) / 2
    
    def disturb(self, x, t, z=None):
        if isinstance(t, torch.Tensor):
            scale = self.ALPHA(t).sqrt()
            sigma = self.BETA(t).sqrt()
        else:
            scale = np.sqrt(self.ALPHA(t))
            sigma = np.sqrt(self.BETA(t))
        xt = scale * x
        z = torch.randn_like(x) if z == None else z
        xt += sigma * z
        return xt


class VariationalDMSampler():
    """
    ref:
    Meng, C., He, Y., Song, Y., Song, J., Wu, J., Zhu, J.-Y., & Ermon, S. (2022). 
    SDEDIT: GUIDED IMAGE SYNTHESIS AND EDITING WITH STOCHASTIC DIFFERENTIAL EQUATIONS. 33.
    P21 Algorithm 4
    Li, C., Zhu, J., & Zhang, B. (2022). 
    ANALYTIC-DPM: AN ANALYTIC ESTIMATE OF THE OPTIMAL REVERSE VARIANCE IN DIFFUSION PROB- ABILISTIC MODELS. 39.
    P22
    Diederik P Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models.
    arXiv preprint arXiv:2107.00630, 2021.
    """

    def __init__(self, scheduler):
        self.sch = scheduler

    def ALPHA_ts(self, t, s):
        return self.sch.ALPHA(t) / self.sch.ALPHA(s)
    
    def BETA_ts(self, t, s):
        return self.sch.BETA(t) - self.ALPHA_ts(t, s) * self.sch.BETA(s)

    def MU(self, score_model, x, t, s):
        # t > s
        dividend = self.ALPHA_ts(t, s)
        factor = 1 / torch.sqrt(dividend) if isinstance(dividend, torch.Tensor) else 1 / np.sqrt(dividend)
        grad = self.BETA_ts(t, s) * score_model(x, t)
        return factor * (x + grad)
    
    def SIGMA(self, t, s):
        # t > s
        sigma2 = self.sch.BETA(s) / self.sch.BETA(t) * self.BETA_ts(t, s)
        return torch.sqrt(sigma2) if isinstance(sigma2, torch.Tensor) else np.sqrt(sigma2)

    def sampling(self, score_model, data_shape, batch_size=2, init_point=1, num_steps=500, device='cuda'):

        eps = 1e-3
        score_model.eval()
        init_x = torch.randn(batch_size, *data_shape, device=device) * np.sqrt(self.sch.BETA(init_point))
        trajectory = torch.linspace(init_point, eps, num_steps+1, device=device)
        K = trajectory.shape[0]
        x = init_x
        with torch.no_grad():
            for i in tqdm.tqdm(range(K-1)):  
                time_step = torch.ones(batch_size, device=device)[:, None, None]
                mean_x = self.MU(score_model, x, time_step * trajectory[i], time_step * trajectory[i+1])
                sigma = self.SIGMA(time_step * trajectory[i], time_step * trajectory[i+1])
                x = mean_x + sigma * torch.randn_like(x)      
                # Do not include any noise in the last sampling step.
        return mean_x
    
    def sampling_traces(self, score_model, data_shape, batch_size=2, init_point=1, num_steps=500, device='cuda'):

        eps = 1e-3
        score_model.eval()
        init_x = torch.randn(batch_size, *data_shape, device=device) * np.sqrt(self.sch.BETA(init_point))
        trajectory = torch.linspace(init_point, eps, num_steps+1, device=device)
        K = trajectory.shape[0]
        x = init_x
        traces = []
        traces.append(x)
        with torch.no_grad():
            for i in tqdm.tqdm(range(K-1)):  
                time_step = torch.ones(batch_size, device=device)[:, None, None]
                mean_x = self.MU(score_model, x, time_step * trajectory[i], time_step * trajectory[i+1])
                sigma = self.SIGMA(time_step * trajectory[i], time_step * trajectory[i+1])
                x = mean_x + sigma * torch.randn_like(x)      
                # Do not include any noise in the last sampling step.
                traces.append(mean_x)
        return mean_x, traces


class SineWave(Dataset):
    
    def __init__(self, noise_level=0, timesteps=512, n_channels=1, freq_band=[1,10]):
        super().__init__()
        self.noise_level = noise_level
        self.timesteps = timesteps
        self.n_channels = n_channels
        self.freq_band = freq_band
    
    def __len__(self):
        return 3000

    def __getitem__(self, index):
        
        if self.n_channels == 1:
            amp = torch.rand(1) * 0.5 + 0.75 # [0.75, 1.25]
            phase = torch.rand(1) * 2 * np.pi # [0, pi]
            freq = torch.rand(1) * (self.freq_band[1] - self.freq_band[0]) + self.freq_band[0] # [1, 10]
        else:
            amp_bound = torch.rand(2).sort().values * 0.5 + 0.75 # [0.75, 1.25]
            phase_bound = torch.rand(2).sort().values * 2 * np.pi # [0, pi]
            freq_bound = torch.rand(2).sort().values * (self.freq_band[1] - self.freq_band[0]) + self.freq_band[0] # [1, 10]
            amp = torch.linspace(amp_bound[0], amp_bound[1], self.n_channels).unsqueeze(dim=1)
            phase = torch.linspace(phase_bound[0], phase_bound[1], self.n_channels).unsqueeze(dim=1)
            freq = torch.linspace(freq_bound[0], freq_bound[1], self.n_channels).unsqueeze(dim=1)
        x = torch.linspace(0, 1, self.timesteps).unsqueeze(dim=0).expand(self.n_channels, -1)
        x = amp * torch.sin((2*np.pi*freq) * x + phase)
        if self.noise_level != 0:
            x += torch.randn_like(x) * self.noise_level
        return x.T


class ECG_dataset(Dataset):
    def __init__(self, path='ECG_data_10000.npy', transposed=False, cutoff_freq=None):
        super().__init__()
        self.path = path
        self.transposed = transposed
        self.data = np.load(self.path)
        self.data = torch.from_numpy(self.data.astype(np.float32))
        freq = 100
        if cutoff_freq is not None:
            self.data = self._highpass_filter(self.data, freq, cutoff_freq)

    @staticmethod
    def _highpass_filter(data, freq, cutoff_freq):
        device = data.device
        data = highpass_biquad(data.cuda(), freq, cutoff_freq)
        return data.to(device)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        data = self.data[index] # (C, L)
        if not self.transposed: data = data.transpose(1, 0)
        return data


class MI_dataset(Dataset):
    
    def __init__(self, path, subject=None, transposed=False, new_freq=None, freq_band=None, device='cuda'):
        super().__init__()
        self.path = path
        self.transposed = transposed 
        data = self.load_dataset(path, subject)
        data = torch.from_numpy(data).to(device)

        self.freq = 128 
        if new_freq is not None: 
            data = resample(data, self.freq, new_freq)
            self.freq = new_freq
        if freq_band is not None:
            data = highpass_biquad(data, self.freq, freq_band[1])
            data = lowpass_biquad(data, self.freq, freq_band[0])

        self.data = data
            

    @staticmethod
    def load_dataset(path, subject):
        from os.path import join
        sub_list = os.listdir(path)
        if subject is not None:
            assert subject in sub_list
            sub_list = [subject]
        sub_path = [join(path,sub) for sub in sub_list]
        
        data = np.empty((0, 64, 512))
        for sub in sub_path:
            files = os.listdir(sub)
            files_path = [join(sub,file) for file in files]
            for file in files_path:
                data = np.concatenate([data, np.load(file)], axis=0)
        
        return data
        
    def __getitem__(self, index):
        x = self.data[index] # (C, L)
        if not self.transposed: x = x.transpose(1, 0)
        return x

    def __len__(self):
        return self.data.shape[0]


def loss_fn(model, x, scheduler, eps=1e-5):
    random_t = torch.rand(x.shape[0], device=x.device) * (1.-eps) + eps
    random_t = random_t[:, None, None] # (N, ) -> (N, 1, 1)
    z = torch.randn_like(x)
    std = torch.sqrt(scheduler.BETA(random_t))
    # mu = torch.sqrt(scheduler.ALPHA(random_t))
    # perturbed_x = x * mu + z * std
    perturbed_x = scheduler.disturb(x, random_t, z)
    score = model(perturbed_x, random_t)
    loss = torch.mean(
        torch.sum((score * std + z)**2, dim=(1, 2)))
    return loss


def loss_fn_forecast(model, x, *args, **kwargs):
    # x(B, L, C)
    x0, y = torch.chunk(x, 2, dim=1)
    y_pre = model(x0)
    loss = F.mse_loss(y_pre, y)
    return loss


_c2r = torch.view_as_real
_r2c = torch.view_as_complex

class S4DKernel(nn.Module):
    """Wrapper around SSKernelDiag that generates the diagonal SSM parameters
    """

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        # Generate dt
        H = d_model
        log_dt = torch.rand(H) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(_c2r(C))
        self.register("log_dt", log_dt, lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N//2))
        A_imag = math.pi * repeat(torch.arange(N//2), 'n -> h n', h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        # Materialize parameters
        dt = torch.exp(self.log_dt) # (H)
        C = _r2c(self.C) # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag # (H N)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device) # (H N L)
        C = C * (torch.exp(dtA)-1.) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real

        return K

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))


class S4D(nn.Module):

    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)

        # Pointwise
        self.activation = nn.GELU()
        dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
        # dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2*self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u, **kwargs): # absorbs return_output and transformer src mask
        """ Input and output shape (B, H, L) """ # when transposed = True
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(L=L) # (H L)

        # Convolution
        k_f = torch.fft.rfft(k, n=2*L) # (H L)
        u_f = torch.fft.rfft(u, n=2*L) # (B H L)
        y = torch.fft.irfft(u_f*k_f, n=2*L)[..., :L] # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed: y = y.transpose(-1, -2)
        return y # Return a dummy state to satisfy this repo's interface, but this can be modified


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class S4Model(nn.Module):

    def __init__(self, in_chn, num_tokens, depth, transposed=False, dropout=0, **kernel_args):
        super().__init__()
        self.input_projection = nn.Conv1d(in_chn, num_tokens, 1)

        self.s4_layers = nn.Sequential(
            *[
                nn.Sequential(
                    Residual(
                        nn.Sequential(
                            S4D(num_tokens, dropout=dropout, **kernel_args),
                            nn.Dropout1d(dropout),
                        )
                    ),
                    Rearrange('b c l -> b l c'),
                    nn.LayerNorm(num_tokens),
                    Rearrange('b l c -> b c l'),
                ) for _ in range(depth)
            ]
        )
        self.output_projection = nn.Conv1d(num_tokens, 1, 1)

        self.transposed = transposed 

    def forward(self, x):
        # x(B, L, K) -> (transposed == False), (B, K, L) -> (transposed == True)
        if not self.transposed: x = x.transpose(-1, -2)
        x = self.input_projection(x) # (B, C, L)
        x = self.s4_layers(x)
        x = self.output_projection(x)
        if not self.transposed: x = x.transpose(-1, -2)
        return x


from s4 import S4


class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):   # x is the time
        x_porj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_porj), torch.cos(x_porj)], dim = -1)


class ResidualBlock(nn.Module):

    def __init__(self, num_tokens, embedding_dim, **kwargs):
        super().__init__()
        self.diffusion_projection = nn.Sequential(
            nn.Linear(embedding_dim, num_tokens),
            Rearrange('b c -> b c 1')
        )
        
        self.input_projection = nn.Conv1d(num_tokens, 2 * num_tokens, 1)
        self.output_projection = nn.Conv1d(num_tokens, 2 * num_tokens, 1)

        self.S4_1 = S4(2 * num_tokens, **kwargs)
        self.ln1 = nn.Sequential(
            Rearrange('b c l -> b l c'),
            nn.LayerNorm(2 * num_tokens),
            Rearrange('b l c -> b c l'),
        )
        self.S4_2 = S4(2 * num_tokens, **kwargs)
        self.ln2 = nn.Sequential(
            Rearrange('b c l -> b l c'),
            nn.LayerNorm(2 * num_tokens),
            Rearrange('b l c -> b c l'),
        )

    def forward(self, x, diffusion_emb):
        # x(B, C, L)
        # diffusion_emb(B, embedding_dim)

        diffusion_emb = self.diffusion_projection(diffusion_emb) # (B, C, 1)
        y = x + diffusion_emb

        y = self.input_projection(y) # (B, 2C, L)
        y = self.S4_1(y) # (B, 2C, L)
        y = self.ln1(y)
        y = self.S4_2(y) # (B, 2C, L)
        y = self.ln2(y)

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B, C, L)
        y = self.output_projection(y) # (B, 2C, L)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip


class SSSD(nn.Module):

    def __init__(self, nscheduler, in_chn, num_tokens, depth, dropout=0, embedding_dim=128, transposed=False, **kwargs):
        super().__init__()
        self.diffusion_embedding = nn.Sequential(
            GaussianFourierProjection(embed_dim=embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU()
        )

        self.input_projection = nn.Conv1d(in_chn, num_tokens, 1)
        self.output_projection1 = nn.Conv1d(num_tokens, num_tokens, 1)
        self.output_projection2 = nn.Conv1d(num_tokens, in_chn, 1)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(num_tokens, embedding_dim, **kwargs)
                for _ in range(depth)
            ]
        )

        self.nscheduler = nscheduler
        self.transposed = transposed

    def forward(self, x, t):
        # x(B, L, K) -> (transposed == False), (B, K, L) -> (transposed == True)
        # t(B, 1, 1)
        if not self.transposed: x = x.transpose(-1, -2)
        diffusion_emb = self.diffusion_embedding(t.view(-1)) # (B, embedding_dim)
        x = self.input_projection(x) # (B, C, L)

        sum_skip = torch.zeros_like(x)
        for layer in self.residual_layers:
            x, skip = layer(x, diffusion_emb) # (B, C, L)
            sum_skip = sum_skip + skip
        
        x = sum_skip / math.sqrt(len(self.residual_layers))
        x = F.relu(self.output_projection1(x))
        x = self.output_projection2(x) 

        x = x / torch.sqrt(self.nscheduler.BETA(t))
        if not self.transposed: x = x.transpose(-1, -2)
        return x


import yaml


class YAMLLogger():
    def __init__(self, path):
        assert path[-5:] == '.yaml'
        self.path = path
        self.config = {}

    @classmethod
    def create_from_file(cls, path):
        assert path[-5:] == '.yaml'
        config = cls.load(path)
        self = cls(path)
        self.config = config
        return self
    
    @staticmethod
    def load(path):
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    @staticmethod
    def save(path, config):
        with open(path, 'w') as f:
            yaml.dump(config, f)
     
    def record_append(self, key, value):
        if key not in self.config.keys():
            self.config[key] = []        
        self.config[key].append(value)
        self.save(self.path, self.config)
    

def training(net, loss_fn, train_loader, optimizer, n_epoch, scheduler, path_ckpt, path_config):

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    nscheduler = ContinuousDDPMNoiseScheduler()

    if path_config is not None:
        try:
            logger = YAMLLogger.create_from_file(path_config)
        except:
            logger = YAMLLogger(path_config)
            logger.save(logger.path, logger.config)

    try:
        the_ckpt = torch.load(f'{path_ckpt}')
        net.load_state_dict(the_ckpt)
    except:
        pass


    net = net.to(device)
    print(f'Number of params: {sum([p.numel() for p in net.parameters()])}')

    net = net.train()
    tqdm_epoch = tqdm.trange(n_epoch)
    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for x in train_loader:
            x = x.to(device)
            
            loss = loss_fn(net, x, nscheduler)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        if path_ckpt is not None: torch.save(net.state_dict(), f'{path_ckpt}')
        if scheduler is not None: scheduler.step()
        if path_config is not None: logger.record_append('losses', avg_loss / num_items)

    net = net.eval()

if __name__ == "__main__":
    in_chn, num_tokens, depth = 12, 256, 36
    lr = 1e-4
    n_epoch = 30
    timesteps = 1000 # 1000 -> 500, 500

    nscheduler = ContinuousDDPMNoiseScheduler()
    net = SSSD(nscheduler, in_chn, num_tokens, depth, mode='diag', measure='diag-lin', bidirectional=True)
    optimizer = Adam(net.parameters(), lr=lr, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15)
    # scheduler = None

    # path_dataset = '/content/drive/MyDrive/Reproduction/State Space Models
    path_dataset = '/content/drive/MyDrive/Reproduction/State Space Models/Datasets/ECG'
    path_ckpt = f'{path_dataset}/ckpts/ckpt_c{in_chn}_t{timesteps}_ECG.pth'
    path_config = f'{path_dataset}/ckpts/config_c{in_chn}_t{timesteps}_ECG.yaml'

    # freq_band = [1, 20]
    # train_set = SineWave(n_channels=in_chn, timesteps=timesteps, freq_band=freq_band)
    train_set = ECG_dataset(path=f'{path_dataset}/ECG_data_10000.npy', cutoff_freq=0.5)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True) 
    training(net, loss_fn, train_loader, optimizer, n_epoch, scheduler, path_ckpt, path_config)
