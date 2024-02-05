
import torch
import random
from scipy.spatial.distance import cdist

CALC_SIMILARITY = False

def punish_wight(wo_batch, latent_size, alpha, method):
    if method == 'weight':
        wo_batch *= alpha
    elif method in ['alpha', 'beta', 'delete', 'soft-weight']:
        u, s, vh = torch.linalg.svd(wo_batch)
        u = u[:,:latent_size]
        zero_idx = int(latent_size * alpha)
        if method == 'alpha':
            s[:zero_idx] = 0
        elif method == 'beta':
            s[zero_idx:] = 0
        elif method == 'delete':
            s = s[zero_idx:] if zero_idx < latent_size else torch.zeros(latent_size).to(s.device)
            u = u[:, zero_idx:] if zero_idx < latent_size else u
            vh = vh[zero_idx:, :] if zero_idx < latent_size else vh
        elif method == 'soft-weight':
            if CALC_SIMILARITY:
                _s = s.clone()
                _s[zero_idx:] = 0
                _wo_batch = u @ torch.diag(_s) @ vh
                dist = cdist(wo_batch[:,0].unsqueeze(0).cpu(), _wo_batch[:,0].unsqueeze(0).cpu(), metric='cosine')
                print(f'The distance between the word embedding before and after the punishment: {dist}')
            if alpha == -.001:
                s *= (torch.exp(-.001 * s) * 1.2)  # strengthen objects (our Appendix.F)
            else:
                s *= torch.exp(-alpha*s)  # suppression EOT (our main paper)

        wo_batch = u @ torch.diag(s) @ vh
    else:
        raise ValueError('Unsupported method')
    return wo_batch

def woword_eot_context(context, token_indices, alpha, method, n):
    for i, batch in enumerate(context):
        indices = token_indices + [num for num in range(n-1, 77)]
        wo_batch = batch[indices]
        wo_batch = punish_wight(wo_batch.T, len(indices), alpha, method).T
        batch[indices] = wo_batch
    return context

def woword_reweight(attn, token_indices, alpha):
    # if attn.shape[1] > 32 ** 2:  # avoid memory overhead
    #     return attn
    latent_size = int(attn.shape[1]**0.5)
    assert latent_size**2 == attn.shape[1]
    for head_attn in attn:
        for indice in token_indices:
            wo_attn = head_attn[:, indice].reshape(latent_size, latent_size)
            wo_attn *= alpha  # same as Reweight of P2P
            head_attn[:, indice] = wo_attn.reshape(latent_size**2)
    return attn