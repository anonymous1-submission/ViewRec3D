import numpy as np

from src.trainer.eval_pose_metrics import mAA_RRA_and_RTA, mAA_RDA, compute_RRA, compute_RTA, compute_recall_rate, \
    compute_RDA


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def compute_binary_classification_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
    prefix: str = "",
) -> dict:
    if logits.ndim > 1:
        if logits.shape[-1] == 1:
            logits = np.squeeze(logits, axis=-1)
    logits = np.reshape(logits, -1)
    labels = np.reshape(labels, -1)

    probs = sigmoid(logits)
    preds = (probs >= threshold).astype(np.int64)
    labels = labels.astype(np.int64)

    tp = np.sum((preds == 1) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))

    accuracy = (tp + tn) / max(len(labels), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        f"{prefix}accuracy": float(accuracy),
        f"{prefix}precision": float(precision),
        f"{prefix}recall": float(recall),
        f"{prefix}f1": float(f1),
    }

def compute_pose_metrics(pose_pred: np.ndarray, pose_gt: np.ndarray, prefix: str = "pose_") -> dict:

    diff = pose_pred - pose_gt
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(diff))

    trans_pred = pose_pred[:, :3]
    trans_gt = pose_gt[:, :3]
    rot_pred = pose_pred[:, 3:7]
    rot_gt = pose_gt[:, 3:7]
    fov_pred = pose_pred[:, 7:9]
    fov_gt = pose_gt[:, 7:9]

    fov_diff = fov_pred - fov_gt
    fov_mse = np.mean(fov_diff ** 2)
    fov_rmse = np.sqrt(fov_mse)
    fov_mae = np.mean(np.abs(fov_diff))

    rra_err = compute_RRA(rot_pred, rot_gt)
    rra_median = np.median(rra_err)
    rra_mean = np.mean(rra_err)

    rta_err = compute_RTA(trans_pred, trans_gt)
    rta_median = np.median(rta_err)
    rta_mean = np.mean(rta_err)

    rda_err = compute_RDA(trans_pred, trans_gt)
    rda_median = np.median(rda_err)
    rda_mean = np.mean(rda_err)

    rra_5 = compute_recall_rate(rra_err, threshold=5.0)
    rra_15 = compute_recall_rate(rra_err, threshold=15.0)
    rra_30 = compute_recall_rate(rra_err, threshold=30.0)
    rta_5 = compute_recall_rate(rta_err, threshold=5.0)
    rta_15 = compute_recall_rate(rta_err, threshold=15.0)
    rta_30 = compute_recall_rate(rta_err, threshold=30.0)
    rda_01 = compute_recall_rate(rda_err, threshold=0.1)
    rda_05 = compute_recall_rate(rda_err, threshold=0.5)
    rda_1 = compute_recall_rate(rda_err, threshold=1.0)

    mAA_RRA_and_RTA_score = mAA_RRA_and_RTA(rra_err, rta_err)
    mAA_RDA_score = mAA_RDA(rda_err)

    pose_metrics = {
        f"{prefix}mse": float(mse),
        f"{prefix}rmse": float(rmse),
        f"{prefix}mae": float(mae),
        f"{prefix}rra_median": rra_median,
        f"{prefix}rra_mean": rra_mean,
        f"{prefix}rra@5": rra_5,
        f"{prefix}rra@15": rra_15,
        f"{prefix}rra@30": rra_30,
        f"{prefix}rta_median": rta_median,
        f"{prefix}rta_mean": rta_mean,
        f"{prefix}rta@5": rta_5,
        f"{prefix}rta@15": rta_15,
        f"{prefix}rta@30": rta_30,
        f"{prefix}rda_median": rda_median,
        f"{prefix}rda_mean": rda_mean,
        f"{prefix}rda@0.1": rda_01,
        f"{prefix}rda@0.5": rda_05,
        f"{prefix}rda@1.0": rda_1,
        f"{prefix}mAA_RDA": mAA_RDA_score,
        f"{prefix}mAA_RRA_and_RTA": mAA_RRA_and_RTA_score,
        f"{prefix}scale_mse": float(fov_mse),
        f"{prefix}scale_rmse": float(fov_rmse),
        f"{prefix}scale_mae": float(fov_mae),
    }
    return pose_metrics

def build_compute_metrics(
    change_orientation_threshold: float = 0.5,
):

    def compute_metrics(eval_pred):
        predictions = eval_pred.predictions
        label_ids = eval_pred.label_ids
        change_orientation_pred, pose_pred = predictions
        change_orientation_gt, pose_gt = label_ids

        change_orientation_pred = np.asarray(change_orientation_pred)
        pose_pred = np.asarray(pose_pred, dtype=np.float32)

        change_orientation_gt = np.asarray(change_orientation_gt)
        pose_gt = np.asarray(pose_gt, dtype=np.float32)

        metrics = {}

        metrics.update(
            compute_binary_classification_metrics(
                logits=change_orientation_pred,
                labels=change_orientation_gt,
                threshold=change_orientation_threshold,
                prefix="change_orientation_",
            )
        )

        metrics.update(
            compute_pose_metrics(
                pose_pred=pose_pred,
                pose_gt=pose_gt,
                prefix="pose_",
            )
        )

        return metrics

    return compute_metrics