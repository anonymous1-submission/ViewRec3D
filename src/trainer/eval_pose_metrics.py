import numpy as np

# 计算RDA（位置）误差
def compute_RDA(t1: np.ndarray, t2: np.ndarray):
    # 计算欧几里得距离
    distances = np.linalg.norm(t1 - t2, axis=1)
    return distances

# 计算RRA（四元数）误差
def compute_RRA(q1: np.ndarray, q2: np.ndarray, eps=1e-8):
    norm_q1 = np.linalg.norm(q1, axis=1, keepdims=True)
    norm_q2 = np.linalg.norm(q2, axis=1, keepdims=True)

    q1 = q1 / (norm_q1 + eps)
    q2 = q2 / (norm_q2 + eps)

    # 如果输入是 xyzw，这样转成 wxyz
    q1 = np.stack([q1[:, 3], q1[:, 0], q1[:, 1], q1[:, 2]], axis=-1)
    q2 = np.stack([q2[:, 3], q2[:, 0], q2[:, 1], q2[:, 2]], axis=-1)

    d = np.sum(q1 * q2, axis=-1)
    d = np.clip(np.abs(d), -1.0, 1.0)

    rra_err = 2 * np.arccos(d)
    rra_err = rra_err * 180.0 / np.pi
    return rra_err

# 计算RTA（直接法）误差
def compute_RTA(t1: np.ndarray, t2: np.ndarray, eps=1e-8):
    n1 = np.linalg.norm(t1, axis=1, keepdims=True)
    n2 = np.linalg.norm(t2, axis=1, keepdims=True)

    # 避免除0
    t1 = t1 / (n1 + eps)
    t2 = t2 / (n2 + eps)

    cos_theta = np.sum(t1 * t2, axis=-1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    rta_err = np.arccos(cos_theta)
    rta_err = rta_err * 180.0 / np.pi
    return rta_err

# 计算RDA的召回率
def compute_recall_rate(value, threshold: float):
    # 计算距离小于 threshold 的比例
    recall_rate = np.mean(value <= threshold)
    return recall_rate


# RRA 与 RTA
def mAA_RRA_and_RTA(rra_err: np.ndarray, rta_err: np.ndarray, min_threshold: int = 1, max_threshold: int = 30):
    max_err = np.maximum(rra_err, rta_err)

    thresholds = np.arange(min_threshold, max_threshold + 1)

    acc = np.mean(max_err[:, None] < thresholds[None, :], axis=0)

    return np.mean(acc)

# RDA
def mAA_RDA(rda: np.ndarray, min_threshold: int = 0.1, max_threshold: int = 1.0):

    thresholds = np.arange(min_threshold, max_threshold + 1, step=0.1)

    acc = np.mean(rda[:, None] < thresholds[None, :], axis=0)

    return np.mean(acc)