import torch
import torch.nn as nn
import torch.nn.functional as F


class Entropy(nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()

    def __call__(self, logits):
        return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


class SymmetricCrossEntropy(nn.Module):
    def __init__(self, alpha=0.5):
        super(SymmetricCrossEntropy, self).__init__()
        self.alpha = alpha

    def __call__(self, x, x_ema):
        return -(1 - self.alpha) * (x_ema.softmax(1) * x.log_softmax(1)).sum(
            1
        ) - self.alpha * (x.softmax(1) * x_ema.log_softmax(1)).sum(1)


class AugCrossEntropy(nn.Module):
    def __init__(self, alpha=0.5):
        super(AugCrossEntropy, self).__init__()
        self.alpha = alpha

    def __call__(self, x, x_aug, x_ema):
        return -(1 - self.alpha) * (x.softmax(1) * x_ema.log_softmax(1)).sum(
            1
        ) - self.alpha * (x_aug.softmax(1) * x_ema.log_softmax(1)).sum(1)


class SoftLikelihoodRatio(nn.Module):
    def __init__(self, clip=0.99, eps=1e-5):
        super(SoftLikelihoodRatio, self).__init__()
        self.eps = eps
        self.clip = clip

    def __call__(self, logits):
        probs = logits.softmax(1)
        probs = torch.clamp(probs, min=0.0, max=self.clip)
        return -(
            probs * torch.log((probs / (torch.ones_like(probs) - probs)) + self.eps)
        ).sum(1)


class GeneralizedCrossEntropy(nn.Module):
    """Paper: https://arxiv.org/abs/1805.07836"""

    def __init__(self, q=0.8):
        super(GeneralizedCrossEntropy, self).__init__()
        self.q = q

    def __call__(self, logits, targets=None):
        probs = logits.softmax(1)
        if targets is None:
            targets = probs.argmax(dim=1)
        probs_with_correct_idx = probs.index_select(-1, targets).diag()
        return (1.0 - probs_with_correct_idx**self.q) / self.q


def entropy_loss(probs) -> torch.Tensor:
    """
    Calculate the entropy loss for a given probability distribution.

    Args:
        probs: The probability distribution.

    Returns:
        The entropy loss.
    """
    ent = -torch.sum(probs * torch.log(probs + 1e-16), dim=1)
    return ent.mean()


def diversity_loss(ensemble_probs) -> torch.Tensor:
    """
    Calculate the diversity loss for an ensemble of models.

    Args:
        ensemble_probs: The probability distributions of the ensemble.

    Returns:
        The diversity loss.
    """
    mean_probs = ensemble_probs.mean(dim=0)
    div = -torch.sum(mean_probs * torch.log(mean_probs + 1e-16))
    return div


def info_max_loss(probs) -> torch.Tensor:
    """
    Calculate the information maximization loss for a given probability distribution.

    Args:
        probs: The probability distribution.

    Returns:
        The information maximization loss.
    """
    ent = entropy_loss(probs)
    div = diversity_loss(probs)
    return ent - div


def orthogonal_loss(prototypes) -> torch.Tensor:
    """
    Calculate the orthogonal loss for a set of prototypes.

    Args:
        prototypes: The prototypes.

    Returns:
        The orthogonal loss.
    """
    n_prototypes = prototypes.size(0)
    prototypes = prototypes.view(n_prototypes, -1)
    return torch.mm(prototypes, prototypes.t()).pow(2).sum() / (
        n_prototypes * (n_prototypes - 1)
    )


def differential_loss(
    outputs, outputs_t1, outputs_t2, lamda, rms_norm, thresh=0.4
) -> torch.Tensor:
    """
    Calculate the differential loss between two teacher models.

    Args:
        outputs: The student model's output.
        outputs_t1: The first teacher model's output.
        outputs_t2: The second teacher model's output.
        lamda: The scaling factor.
        rms_norm: The RMSNorm layer.
        thresh: The entropy threshold.

    Returns:
        The differential loss.
    """
    # Calculate entropy for teacher 1 predictions
    prob_t1 = torch.softmax(outputs_t1, dim=-1)
    ent1 = -torch.sum(prob_t1 * torch.log(prob_t1 + 1e-16), dim=1)

    # Calculate entropy for teacher 2 predictions
    prob_t2 = torch.softmax(outputs_t2, dim=-1)
    ent2 = -torch.sum(prob_t2 * torch.log(prob_t2 + 1e-16), dim=1)

    # Select only those samples having entropies greater than threshold in both teacher's output
    mask = torch.logical_and(ent1 >= thresh, ent2 >= thresh)

    masked_prob_t1 = prob_t1[mask]
    masked_prob_t2 = prob_t2[mask]
    masked_outputs = outputs[mask]

    if len(masked_prob_t1) > 0:
        diff = masked_prob_t1 - lamda * masked_prob_t2
        output = rms_norm(diff)
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        return kl_loss(
            torch.softmax(masked_outputs, dim=-1), torch.softmax(output, dim=-1)
        )

    return torch.tensor(0.0, device=outputs.device, requires_grad=True)


class RMSNorm(nn.Module):
    """RMSNorm Implementation"""

    def __init__(self, dim, device, eps=1e-5):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim, device=device, requires_grad=True))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        x = x / (rms + self.eps)
        return self.scale * x

class RBF(nn.Module):
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (
            torch.arange(n_kernels) - n_kernels // 2
        )
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples**2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(
            -L2_distances[None, ...]
            / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers.to(X.device))[
                :, None, None
            ]
        ).sum(dim=0)


class MMDLoss(nn.Module):
    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY
    

class L2SPLoss(nn.Module):
    def __init__(self, pre_trained_weights):
        super(L2SPLoss, self).__init__()
        self.pre_trained_weights = pre_trained_weights  # source model weights

    def forward(self, model):
        loss = 0.0
        for name, param in model.named_parameters():
            loss += F.mse_loss(param, self.pre_trained_weights[name].to(param.device))
        return loss
    
class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device="cuda"):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))

    def forward(self, features, labels):
        batch_size = features.size(0)
        centers_batch = self.centers[labels]
        loss = (features - centers_batch).pow(2).sum() / batch_size
        return loss