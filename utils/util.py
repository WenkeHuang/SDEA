import os
import torch.nn as nn

def create_if_not_exists(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def save_networks(model,communication_idx):
    nets_list = model.nets_list
    model_name = model.NAME
    save_option = True

    if save_option:
        checkpoint_path = model.checkpoint_path
        model_path = os.path.join(checkpoint_path, model_name)
        model_para_path = os.path.join(model_path, 'para')
        create_if_not_exists(model_para_path)
        for net_idx,network in enumerate(nets_list):
            each_network_path = os.path.join(model_para_path, str(communication_idx) + '_' + str(net_idx) + '.ckpt')
            torch.save(network.state_dict(),each_network_path)

import torch


class ConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all'):
        super(ConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(self, features, device):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        mask = torch.eye(batch_size, dtype=torch.float32).to(device)


        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # loss
        # loss = - (self.temperature) * mean_log_prob_pos
        loss = -  mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss


def HE(probs):
    mean = probs.mean(dim=0)
    ent  = - (mean * (mean + 1e-5).log()).sum()
    return ent

def EH(probs):
    ent = - (probs * (probs + 1e-5).log()).sum(dim=1)
    mean = ent.mean()
    return mean


