import torch
from torch import nn
from torch.nn import functional as F

class Loss(torch.nn.Module):
    def __init__(self, loss_type='mse'):
        super(Loss, self).__init__()

        self.loss_type = loss_type

        self.loss_func = {
            'mse': torch.nn.MSELoss,
            'cosine': torch.nn.CosineSimilarity,
            'mae': torch.nn.L1Loss
        }[loss_type]()

    def forward(self, x, y):
        if self.loss_type == "cosine":
            return 1. - self.loss_func(x, y)

        return self.loss_func(x, y)

class AttnLoss(nn.Module):
    def __init__(self, device, attn_loss_type, n, token_indices, lambda_retain=1., lambda_erase=-1., lambda_self_retain=1., lambda_self_erase=-1.):
        super(AttnLoss, self).__init__()
        self.device = device
        self.prompt_n = n
        self.token_indices = token_indices

        self.lambda_retain = lambda_retain
        self.lambda_erase = lambda_erase
        self.lambda_self_retain = lambda_self_retain
        self.lambda_self_erase = lambda_self_erase

        self.retain_loss = Loss(attn_loss_type)
        self.erase_loss = Loss(attn_loss_type)
        self.self_retain_loss = Loss(attn_loss_type)
        self.self_erase_loss = Loss(attn_loss_type)

    def calc_mask(self, attn, threshold=.85):
        mask = []
        for i in [num for num in range(1, self.prompt_n-1)]:
            _attn = attn[:,:,i].clone()
            _attn = 255 * _attn / _attn.max()
            _attn = F.interpolate(_attn.unsqueeze(0).unsqueeze(0), size=(256, 256), mode='bilinear')
            if i in self.token_indices:
                _threshold = threshold
            else:
                _threshold = threshold + .1
            _attn[_attn >= _attn.max() * _threshold] = 255
            _attn[_attn < _attn.max() * _threshold] = 0
            _attn = F.interpolate(_attn, size=attn.shape[:2], mode='bilinear')
            mask += [_attn.squeeze(0).squeeze(0)]
        return mask

    def calc_retain_loss(self, attn, attn_erase):
        loss = .0
        for i in [num for num in range(1, self.prompt_n-1)]:
            if i in self.token_indices:
                continue
            loss += self.retain_loss(attn[:,:,i], attn_erase[:,:,i])
        # print(f'\n retain loss: {loss.item()}, ', end=' ')
        return loss

    def calc_erase_loss(self, attn, attn_erase):
        loss = .0
        for i in self.token_indices:
            loss += self.erase_loss(attn[:,:,i], attn_erase[:,:,i])
        # print(f'erase loss: {loss.item()}')
        return loss

    def calc_self_retain_loss(self, self_attn, self_attn_erase, mask):
        loss = .0
        h, w = mask[0].shape
        for i in [num for num in range(1, self.prompt_n-1)]:
            if i in self.token_indices:
                continue
            for j, m in enumerate(mask[i-1].reshape(h*w)):
                if m > 0:
                    loss += self.self_retain_loss(self_attn[:,:,j].view(-1).unsqueeze(0),
                                                  self_attn_erase[:,:,j].view(-1).unsqueeze(0))
        # print(f'self retain loss: {loss.item()}, ', end=' ')
        return loss

    def calc_self_erase_loss(self, self_attn, self_attn_erase, mask):
        loss = .0
        h, w = mask[0].shape
        for i in self.token_indices:
            for j, m in enumerate(mask[i-1].reshape(h*w)):
                if m > 0:
                    loss += self.self_erase_loss(self_attn[:,:,j].view(-1).unsqueeze(0),
                                                 self_attn_erase[:,:,j].view(-1).unsqueeze(0))
        # print(f'self erase loss: {loss.item()}')
        return loss

    def forward(self, attn, attn_erase, self_attn, self_attn_erase):
        attn, attn_erase, self_attn, self_attn_erase \
            = attn.to(torch.double), attn_erase.to(torch.double), self_attn.to(torch.double), self_attn_erase.to(torch.double)
        attn_loss = .0

        if self.lambda_self_retain or self.lambda_self_erase:
            mask = self.calc_mask(attn)

        h, w, seq_len = attn.shape
        attn = attn.reshape(h*w, seq_len).unsqueeze(0)
        attn_erase = attn_erase.reshape(h*w, seq_len).unsqueeze(0)

        if self.lambda_retain:
            attn_loss += self.lambda_retain * self.calc_retain_loss(attn, attn_erase)

        if self.lambda_erase:
            attn_loss += self.lambda_erase * self.calc_erase_loss(attn, attn_erase)

        if self.lambda_self_retain:
            attn_loss += self.lambda_self_retain * self.calc_self_retain_loss(self_attn, self_attn_erase, mask)

        if self.lambda_self_erase:
            attn_loss += self.lambda_self_erase * self.calc_self_erase_loss(self_attn, self_attn_erase, mask)

        loss = attn_loss
        return loss