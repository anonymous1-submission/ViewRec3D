import torch
from transformers import Trainer
from src.trainer.metrics import build_compute_metrics


import torch
from transformers import Trainer


class VGGTTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 用来暂存最近一次训练 step 的分支 loss，供 log() 写入 TensorBoard
        self._extra_train_logs = {}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs["loss"]

        extra_logs = {}

        if "loss_change_orientation" in outputs:
            extra_logs["loss_change_orientation"] = float(
                outputs["loss_change_orientation"].detach().cpu().item()
            )

        if "loss_pose" in outputs:
            extra_logs["loss_pose"] = float(
                outputs["loss_pose"].detach().cpu().item()
            )

        if "loss_rays" in outputs:
            extra_logs["loss_rays"] = float(
                outputs["loss_rays"].detach().cpu().item()
            )
        self._extra_train_logs = extra_logs

        if return_outputs:
            return loss, outputs
        return loss

    def log(self, logs, start_time=None):
        logs = dict(logs)

        # 只在训练日志中追加，避免污染 eval_* 指标
        if "loss" in logs and len(self._extra_train_logs) > 0:
            logs.update(self._extra_train_logs)

        super().log(logs, start_time=start_time)

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        has_labels = all(
            key in inputs for key in ["change_orientation", "pose"]
        )

        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            outputs = model(**inputs)

        loss = None
        if has_labels and "loss" in outputs:
            loss = outputs["loss"].mean().detach()

        if prediction_loss_only:
            return (loss, None, None)

        change_orientation_pred = outputs["change_orientation"].detach()
        pose_pred = outputs["pose"].detach()

        if change_orientation_pred.ndim > 1 and change_orientation_pred.shape[-1] == 1:
            change_orientation_pred = change_orientation_pred.squeeze(-1)

        predictions = (
            change_orientation_pred,
            pose_pred,
        )

        label_ids = None
        if has_labels:
            change_orientation_gt = inputs["change_orientation"].detach()
            pose_gt = inputs["pose"].detach()

            label_ids = (
                change_orientation_gt,
                pose_gt,
            )

        return (loss, predictions, label_ids)

def build_trainer(
    model,
    training_args,
    train_dataset=None,
    eval_dataset=None,
    data_collator=None,
    change_orientation_threshold: float = 0.5,
):
    """
    构建适配三头输出的 Trainer。

    输入:
        model: 你的 VGGT 模型
        training_args: VGGTTrainingArguments
        train_dataset: 训练集
        eval_dataset: 验证集
        data_collator: collator
        change_orientation_threshold: change_orientation 二分类阈值

    输出:
        trainer: VGGTTrainer
    """
    compute_metrics = build_compute_metrics(
        change_orientation_threshold=change_orientation_threshold,
    )

    trainer = VGGTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    return trainer