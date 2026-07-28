from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from coffee26n_experiment.model_utils import setup_local_imports


def test_confusion_matrix_updates_when_plots_are_disabled():
    setup_local_imports()
    from ultralytics.models.yolo.detect.val import DetectionValidator

    calls = []
    validator = object.__new__(DetectionValidator)
    validator.seen = 0
    validator.args = SimpleNamespace(
        plots=False,
        visualize=False,
        save_json=False,
        save_txt=False,
        single_cls=False,
        conf=0.25,
    )
    validator.metrics = SimpleNamespace(update_stats=lambda payload: None)
    validator.confusion_matrix = SimpleNamespace(
        process_batch=lambda pred, batch, conf: calls.append((pred, batch, conf))
    )
    validator._prepare_batch = lambda si, batch: {
        "cls": torch.tensor([1.0]),
        "bboxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "im_file": "sample.jpg",
    }
    validator._prepare_pred = lambda pred: pred
    validator._process_batch = lambda pred, batch: {
        "tp": np.zeros((1, 10), dtype=bool),
        "tp_metrics": np.zeros((1, 10), dtype=bool),
        "tp_iou": np.zeros(1),
    }
    pred = {
        "cls": torch.tensor([0.0]),
        "conf": torch.tensor([0.9]),
        "bboxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
    }
    DetectionValidator.update_metrics(validator, [pred], {"unused": True})
    assert len(calls) == 1
    assert calls[0][2] == 0.25
