from tqdm import tqdm
import torch

def k_t(z, t, X):
    """
    z: tensor of shape (D,)
    t: float
    X: tensor of shape (N, D)
    """

    dist_z_to_all = (z - t*X).pow(2).sum(dim=1)
    d = 2*((1 - t)**2)
    normalized_direction = torch.softmax(-dist_z_to_all / d, dim=0)
    out = (normalized_direction.unsqueeze(1) * t*X).sum(dim=0)

    return out


def smoothed_score_function(z, t, X, sigma, M):
    """
    z: tensor of shape (D, N)
    sigma: float
    t: float
    """
    noised = torch.zeros_like(z)
    for i in range(M):
        epsilon_m = torch.randn_like(z)
        noised += k_t(z + sigma * epsilon_m, t, X) - z    
    noised /= M

    return 1/((1-t)**2) * noised


def v_s_t(z, t, X, sigma, M):
    """
    z: tensor of shape (D, N)
    sigma: float
    t: float
    """

    return (1/t)*(z + (1 -t)*smoothed_score_function(z, t, X, sigma, M))


def sample_euler(z_0, steps, x, sigma, M):
    """
    z: tensor of shape (D,)
    t: float
    X: tensor of shape (N, D)
    sigma: float
    M: int
    num_steps: int
    """
    h = 1/steps

    z_n = torch.clone(z_0)
    z_all = [z_0[0]]
    for n in tqdm(range(steps)):
        t_n = n / steps
        if n == 0:
            t_n = 0.00001
        z_n = z_n + h * v_s_t(z_n, t_n, x, sigma, M, )
        z_all.append(z_n[0])
        
    return z_all
