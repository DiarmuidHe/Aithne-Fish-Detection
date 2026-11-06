"""SAM 3 image detector only. No video predictor, tracker, or label export."""

from __future__ import annotations

from pathlib import Path

from scripts.sam3_autolabel import SAM3_CHECKPOINT, build_predictor


class FishDetector:
    def __init__(self, checkpoint: Path = SAM3_CHECKPOINT):
        import torch

        self.torch = torch
        self.predictor, self.provenance = build_predictor(
            checkpoint, mode="image", imgsz=1008, fp16=True, conf=0.25
        )

    def detect(self, image) -> tuple[list[list[float]], list[float], float]:
        """Return source-pixel boxes and scores; duration is synchronized CUDA wall time."""
        import torchvision
        from ultralytics.utils import ops

        p, torch = self.predictor, self.torch
        height, width = image.shape[:2]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        import time

        started = time.perf_counter()
        with torch.inference_mode():
            tensor = p.preprocess([image])
            preds = p._inference_features(p.get_im_features(tensor), text=["fish"])
            scores = (
                preds["pred_logits"].sigmoid() * preds["presence_logit_dec"].sigmoid().unsqueeze(1)
            ).squeeze(-1)
            keep = scores > p.args.conf
            boxes = ops.xywh2xyxy(preds["pred_boxes"][keep].float())
            scores = scores[keep].float()
            order = torchvision.ops.nms(boxes, scores, p.args.iou)
            boxes, scores = boxes[order].cpu().numpy(), scores[order].cpu().numpy()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] * width).clip(0, width)
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] * height).clip(0, height)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        return boxes[valid].round(2).tolist(), [round(float(s), 5) for s in scores[valid]], elapsed
