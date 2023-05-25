import time

import numpy as np
import torch

from distance import cosine_distance, js_distance, bhattacharyya_distance, entropy, cosine_distance_torch, \
    js_distance_torch, bhattacharyya_distance_torch


def calc_conflict(a: np.ndarray, b: np.ndarray) -> float:
    assert len(a.shape) == 1, f"expected array a's shape be (N,), having {a.shape}"
    assert len(b.shape) == 1, f"expected array b's shape be (N,), having {b.shape}"
    assert a.shape[0] == b.shape[0], f"expected array a and b have same length, having a: {a.shape} and b: {b.shape}"

    a, b = np.expand_dims(a, axis=-1), np.expand_dims(b, axis=-1)

    return np.dot(a, b.T).sum() - np.dot(a.T, b).sum()


def ds_fusion_numpy(e: np.ndarray, distance: str, beta: np.ndarray, W: np.ndarray, RI=None):
    """
    improved label fusion method based on D-S Evidence Theory using subjective and objective evidence in numpy
    :param e: evidences to be fused, shape=(L, N)
    :param distance: distance metric
    :param beta: subjective weights, shape=(M, L)
    :param W: AHP comparison matrix, shape=(M+1,M+1)
    :param RI: RI，must be input when M>=10
    :return: fusion result, shape=(N,)
    """

    if distance == 'cos':
        distance_func = cosine_distance
    elif distance == 'js':
        distance_func = js_distance
    elif distance == 'bd':
        distance_func = bhattacharyya_distance
    else:
        raise Exception(f'no distance function called {distance}')

    dis_matrix = np.zeros((e.shape[0], e.shape[0]))
    for i in range(e.shape[0]):
        for j in range(i + 1, e.shape[0]):
            dis_matrix[i, j] = dis_matrix[j, i] = distance_func(e[i], e[j])

    d = np.sum(dis_matrix, axis=-1)

    alpha = 1. / d
    alpha = alpha / alpha.sum()

    if W.shape[0] > 10 and RI is None:
        raise Exception("RI required")

    RI_table = [0, 0, 0, .58, .90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49]
    if W.shape[0] <= 10:
        RI = RI_table[W.shape[0]]

    value, vector = np.linalg.eig(W)

    lambda_max, lambda_index = value.max(), value.argmax()
    CI = (lambda_max - W.shape[0]) / (W.shape[0] - 1)
    if CI / RI > 0.1:
        raise Exception("consistency check failed")

    x = vector[:, lambda_index]
    x = np.expand_dims(np.real(x / x.sum()), 0)

    delta = np.vstack((np.asarray([alpha]), beta))
    delta = delta / delta.sum(axis=-1, keepdims=True)
    delta = np.dot(x, delta)

    e_ = np.sum(delta.T * e, axis=0)

    result = e_.copy()
    for i in range(e.shape[0] - 1):
        result = result * e_ / (1 - calc_conflict(result, e_))

    return result


def calc_conflict_torch(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = torch.unsqueeze(a, dim=-1), torch.unsqueeze(b, dim=-1)

    return (a @ b.T).sum() - (a.T @ b).sum()


def ds_fusion_torch(e: torch.Tensor, distance: str, beta: torch.Tensor, W: torch.Tensor, RI=None):
    if distance == 'cos':
        distance_func = cosine_distance_torch
    elif distance == 'js':
        distance_func = js_distance_torch
    elif distance == 'bd':
        distance_func = bhattacharyya_distance_torch
    else:
        raise Exception(f'no distance function called {distance}')

    dis_matrix = torch.zeros((e.shape[0], e.shape[0]))
    for i in range(e.shape[0]):
        for j in range(i + 1, e.shape[0]):
            dis_matrix[i, j] = dis_matrix[j, i] = distance_func(e[i], e[j])

    d = torch.clip(torch.sum(dis_matrix, dim=-1), min=1e-6, max=1e6)

    alpha = 1. / d
    alpha = alpha / alpha.sum()

    if W.shape[0] > 10 and RI is None:
        raise Exception("RI required")

    RI_table = [0, 0, 0, .58, .90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49]
    if W.shape[0] <= 10:
        RI = RI_table[W.shape[0]]

    value, vector = torch.linalg.eig(W)

    lambda_max, lambda_index = torch.max(value.real).item(), torch.argmax(value.real).item()

    CI = (lambda_max - W.shape[0]) / (W.shape[0] - 1)
    if W.shape[0] > 2 and CI / RI > 0.1:
        raise Exception("consistency check failed")

    x = vector[:, lambda_index].real
    x = torch.unsqueeze(x / x.sum(), 0)

    delta = torch.vstack((torch.unsqueeze(alpha, 0), beta))
    delta = delta / torch.sum(delta, dim=-1, keepdim=True)
    delta = x @ delta

    e_ = torch.sum(delta.T * e, dim=0)

    result = torch.clone(e_)
    for i in range(e.shape[0] - 1):
        result = result * e_ / (1 - calc_conflict_torch(result, e_))

    return result


def contrast_algorithm_zhang(e: np.ndarray):
    dis_matrix = np.zeros((e.shape[0], e.shape[0]))
    for i in range(e.shape[0]):
        for j in range(e.shape[0]):
            if i != j:
                dis_matrix[i, j] = bhattacharyya_distance(e[i], e[j])

    d = np.sum(dis_matrix, axis=-1)

    alpha = 1. / d
    alpha = alpha / alpha.sum()
    alpha = np.expand_dims(alpha, axis=0)

    e_ = np.sum(alpha.T * e, axis=0)

    result = e_.copy()
    for i in range(e.shape[0] - 1):
        result = result * e_ + result * calc_conflict(result, e_)

    return result


def contrast_algorithm_Bai(e: np.ndarray, b=10):
    import pyemd

    distance_matrix = (np.ones((b, b)) - np.identity(b)) * np.sqrt(2)

    dis_matrix = np.zeros((e.shape[0], e.shape[0]))
    for i in range(e.shape[0]):
        for j in range(e.shape[0]):
            if i != j:
                dis_matrix[i, j] = pyemd.emd(e[i], e[j], distance_matrix)

    sim_matrix = np.exp(-dis_matrix)

    sup = sim_matrix.sum(axis=-1) - 1

    w_crd = sup / sup.sum()

    e = np.clip(e, 1e-6, 1e6)

    E_d = -np.sum(e * np.log(e), axis=-1)

    m_plus = e[np.argmin(E_d)]
    m_minus = e[np.argmax(E_d)]

    bmd_max = pyemd.emd(m_plus, m_minus, distance_matrix)
    D_minus = np.zeros((e.shape[0],))
    for i in range(e.shape[0]):
        D_minus[i] = pyemd.emd(e[i], m_minus, distance_matrix)

    chi_minus = D_minus / bmd_max
    chi_minus = np.clip(chi_minus, 1e-7, 1e6)
    true_mask = chi_minus <= 0.75
    false_mask = chi_minus > 0.75

    I = chi_minus * true_mask + (1 - chi_minus) * false_mask

    w_dist = I / I.sum()

    W = w_dist * w_crd
    W = W / W.sum()
    W = np.expand_dims(W, axis=0)

    e_ = np.sum(W.T * e, axis=0)

    result = e_.copy()
    for i in range(e.shape[0] - 1):
        result = result * e_ + result * calc_conflict(result, e_)

    return result


def contrast_algorithm_zhang_torch(e: torch.Tensor):
    dis_matrix = torch.zeros((e.shape[0], e.shape[0]))
    for i in range(e.shape[0]):
        for j in range(e.shape[0]):
            if i != j:
                dis_matrix[i, j] = bhattacharyya_distance_torch(e[i], e[j])

    D = torch.sum(dis_matrix, dim=-1)
    alpha = 1. / D
    alpha = alpha / alpha.sum()
    alpha = torch.unsqueeze(alpha, dim=0)

    e_ = torch.sum(alpha.T * e, dim=0)

    result = e_.clone()
    for i in range(e.shape[0] - 1):
        result = result * e_ + result * calc_conflict_torch(result, e_)

    return result


def contrast_algorithm_jiang(e: np.ndarray):
    def jousselme_distance(p: np.ndarray, q: np.ndarray):
        return np.sqrt(0.5 * np.dot(np.expand_dims(p - q, axis=0), np.expand_dims(p - q, axis=-1)))

    dis_matrix = np.zeros((e.shape[0], e.shape[0]))
    for i in range(e.shape[0]):
        for j in range(e.shape[0]):
            if i != j:
                dis_matrix[i, j] = jousselme_distance(e[i], e[j])

    d_bar = dis_matrix.sum(axis=-1) / (e.shape[0] - 1)
    d = d_bar.sum() / e.shape[0]
    credible_mask = d_bar <= d
    incredible_mask = d_bar > d

    e = np.clip(e, a_min=1e-6, a_max=1e6)

    E_d = -np.sum(e * np.log(e), axis=-1)
    E_d = E_d / E_d.sum()

    alpha = credible_mask * np.exp(-E_d) + incredible_mask * np.exp(-(np.max(E_d) + 1 - E_d))
    w = np.expand_dims(alpha / alpha.sum(), axis=0)

    e_ = np.sum(w.T * e, axis=0)
    e_ = e_ / e_.sum()

    result = e_.copy()
    for i in range(e.shape[0] - 1):
        result = result * e_ / (1 - calc_conflict(result, e_))

    return result


def contrast_algorithm_jiang_torch(e: torch.Tensor):
    def jousselme_distance(p: torch.Tensor, q: torch.Tensor):
        return torch.sqrt(0.5 * (torch.unsqueeze(p - q, dim=0) @ torch.unsqueeze(p - q, dim=-1)))

    dis_matrix = torch.zeros((e.shape[0], e.shape[0]))
    for i in range(e.shape[0]):
        for j in range(e.shape[0]):
            if i != j:
                dis_matrix[i, j] = jousselme_distance(e[i], e[j])

    d_bar = dis_matrix.sum(dim=-1) / (e.shape[0] - 1)
    d = d_bar.sum() / e.shape[0]
    credible_mask = d_bar <= d
    incredible_mask = d_bar > d

    e = torch.clip(e, min=1e-6, max=1e6)

    E_d = -torch.sum(e * torch.log(e), dim=-1)
    E_d = E_d / E_d.sum()

    alpha = credible_mask * torch.exp(-E_d) + incredible_mask * torch.exp(-(torch.max(E_d) + 1 - E_d))
    w = torch.unsqueeze(alpha / alpha.sum(), dim=0)

    e_ = torch.sum(w.T * e, dim=0)
    e_ = e_ / e_.sum()

    result = e_.clone()
    for i in range(e.shape[0] - 1):
        result = result * e_ / (1 - calc_conflict_torch(result, e_))

    return result


def contrast_algorithm_bai_torch(e: torch.Tensor, b=10):
    import pyemd

    distance_matrix = (np.ones((b, b)) - np.identity(b)) * np.sqrt(2)

    dis_matrix = torch.zeros((e.shape[0], e.shape[0]))
    for i in range(e.shape[0]):
        for j in range(e.shape[0]):
            if i != j:
                dis_matrix[i, j] = pyemd.emd(e[i].numpy().astype(np.float64), e[j].numpy().astype(np.float64), distance_matrix)

    sim_matrix = torch.exp(-dis_matrix)

    sup = sim_matrix.sum(dim=-1) - 1

    w_crd = sup / sup.sum()

    E_d = -torch.sum(e * torch.log(e), dim=-1)

    m_plus = e[torch.argmin(E_d)]
    m_minus = e[torch.argmax(E_d)]

    bmd_max = pyemd.emd(m_plus.numpy().astype(np.float64), m_minus.numpy().astype(np.float64), distance_matrix)
    D_minus = torch.zeros((e.shape[0],))
    for i in range(e.shape[0]):
        D_minus[i] = pyemd.emd(e[i].numpy().astype(np.float64), m_minus.numpy().astype(np.float64), distance_matrix)

    chi_minus = D_minus / bmd_max
    true_mask = chi_minus <= 0.75
    false_mask = chi_minus > 0.75
    I = chi_minus * true_mask + (1 - chi_minus) * false_mask

    w_dist = I / I.sum()

    W = w_dist * w_crd
    W = W / W.sum()
    W = torch.unsqueeze(W, dim=0)

    e_ = torch.sum(W.T * e, dim=0)

    result = e_.clone()
    for i in range(e.shape[0] - 1):
        result = result * e_ + result * calc_conflict_torch(result, e_)

    return result


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    a = np.asarray([1.04e-6, 4.20e-5, 1.09e-2, 4.00e-1, 1.23e-2, 5.71e-1, 3.16e-3, 2.05e-3, 1.56e-5, 1.20e-5])
    b = np.asarray([6.61e-6, 1.37e-5, 7.01e-5, 7.34e-3, 1.01e-5, 9.92e-1, 5.03e-5, 1.27e-5, 5.70e-6, 3.66e-6])
    c = np.asarray([2.22e-8, 1.97e-7, 4.99e-4, 6.07e-1, 1.29e-4, 3.84e-1, 7.93e-3, 1.02e-4, 8.10e-8, 6.29e-6])

    cos_result = ds_fusion_numpy(np.asarray([a, b, c]), distance='cos',
                                 beta=np.asarray([[0.8317, 0.8726, 0.7857],
                                                  [0.8366, 0.8735, 0.7828],
                                                  [1, 1, 0.8]]),
                                 W=np.asarray([[1, 4, 5, 9],
                                               [1 / 4, 1, 1, 5],
                                               [1 / 5, 1, 1, 5],
                                               [1 / 9, 1 / 5, 1 / 5, 1]]))

    cos_result_torch = ds_fusion_torch(torch.from_numpy(np.asarray([a, b, c])), distance='cos',
                                       beta=torch.from_numpy(np.asarray([[0.8317, 0.8726, 0.7857],
                                                                         [0.8366, 0.8735, 0.7828],
                                                                         [1, 1, 0.8]])),
                                       W=torch.from_numpy(np.asarray([[1, 4, 5, 9],
                                                                      [1 / 4, 1, 1, 5],
                                                                      [1 / 5, 1, 1, 5],
                                                                      [1 / 9, 1 / 5, 1 / 5, 1]])))

    js_result = ds_fusion_numpy(np.asarray([a, b, c]), distance='js',
                                beta=np.asarray([[0.8317, 0.8726, 0.7857],
                                                 [0.8366, 0.8735, 0.7828],
                                                 [1, 1, 0.8]]),
                                W=np.asarray([[1, 4, 5, 9],
                                              [1 / 4, 1, 1, 5],
                                              [1 / 5, 1, 1, 5],
                                              [1 / 9, 1 / 5, 1 / 5, 1]]))

    bd_result = ds_fusion_numpy(np.asarray([a, b, c]), distance='bd',
                                beta=np.asarray([[0.8317, 0.8726, 0.7857],
                                                 [0.8366, 0.8735, 0.7828],
                                                 [1, 1, 0.8]]),
                                W=np.asarray([[1, 4, 5, 9],
                                              [1 / 4, 1, 1, 5],
                                              [1 / 5, 1, 1, 5],
                                              [1 / 9, 1 / 5, 1 / 5, 1]]))

    zhang_result = contrast_algorithm_zhang(np.asarray([a, b, c]))

    jiang_result = contrast_algorithm_jiang(np.asarray([a, b, c]))

    # bai_result = contrast_algorithm_bai_torch(torch.from_numpy(np.asarray([a, b, c])))
    bai_result = contrast_algorithm_Bai(np.asarray([a, b, c]))

    print(cos_result, js_result, bd_result, zhang_result, jiang_result, bai_result, sep='\n')

    entropy_list = [entropy(jiang_result, jiang_result),
                    entropy(bai_result, bai_result),
                    entropy(cos_result, cos_result),
                    entropy(js_result, js_result),
                    entropy(bd_result, bd_result),
                    ]
    prob_list = [
        jiang_result.max(),
        bai_result.max(),
        cos_result.max(),
        js_result.max(),
        bd_result.max(),
    ]

    l = [i for i in range(5)]

    plt.bar(l, prob_list, color='yellow', label='probability', alpha=0.5, width=0.5)
    plt.xlabel("methods")
    plt.ylabel("probability")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])

    ax1 = plt.twinx()
    ax1.plot(l, entropy_list, 'r', marker='.', label='entropy')
    ax1.set_ylabel("entropy")
    ax1.set_ylim([0, 1])
    plt.legend(loc="upper right")

    for i, en in zip(l, entropy_list):
        plt.text(i, en, '%.3f'%en, ha='center', va='bottom', fontsize=12)

    plt.xticks(l, ["Jiang's", "Bai's", "ours(cos)", "ours(js)", "ours(bhattacharyya)"])
    plt.show()
