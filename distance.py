import numpy as np
import torch

max_eps, min_eps = 1e6, 1e-6


def entropy(p: np.ndarray, q: np.ndarray = None):
    if q is None:
        q = p

    p, q = np.maximum(p, min_eps), np.maximum(q, min_eps)

    return -np.sum(p*np.log(q))


def kl_divergence(p: np.ndarray, q: np.ndarray):
    return -entropy(p) + entropy(p, q)


def js_distance(p: np.ndarray, q: np.ndarray):
    return 0.5 * kl_divergence(p, (p+q)/2) + 0.5 * kl_divergence(q, (p+q)/2)


def cosine_distance(p: np.ndarray, q: np.ndarray):
    return -np.log(np.clip(np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q)), a_min=min_eps, a_max=max_eps))


def bhattacharyya_distance(p: np.ndarray, q: np.ndarray):
    return -np.log(np.clip(np.sqrt(p * q).sum(), a_min=min_eps, a_max=max_eps))


def entropy_torch(p: torch.Tensor, q: torch.Tensor = None):
    if q is None:
        q = p

    p, q = torch.maximum(p, torch.ones(p.shape) * min_eps), torch.maximum(q, torch.ones(q.shape) * min_eps)

    return -torch.sum(p*torch.log(q))


def kl_divergence_torch(p: torch.Tensor, q: torch.Tensor):
    return -entropy_torch(p) + entropy_torch(p, q)


def js_distance_torch(p: torch.Tensor, q: torch.Tensor):
    return 0.5 * kl_divergence_torch(p, (p+q)/2) + 0.5 * kl_divergence_torch(q, (p+q)/2)


def cosine_distance_torch(p: torch.Tensor, q: torch.Tensor):
    return -torch.log(torch.clip(torch.dot(p, q) / (torch.norm(p) * torch.norm(q)), min=min_eps, max=max_eps))


def bhattacharyya_distance_torch(p: torch.Tensor, q: torch.Tensor):
    return -torch.log(torch.clip(torch.sqrt(p * q).sum(), min=min_eps, max=max_eps))


if __name__ == '__main__':
    import torch.nn.functional as F

    p = np.asarray([0.5, 0.5])
    q = np.asarray([1., 0.])

    print(cosine_distance(p,q))
    print(F.cosine_similarity(torch.from_numpy(p), torch.from_numpy(q), dim=0))
