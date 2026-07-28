from __future__ import annotations

import pytest

from coffee26n_experiment.train_coffee26n import _parser, _select_fitness, _training_overrides


def test_hpc_training_defaults_and_parameter_propagation():
    args = _parser().parse_args(["--data", "dataset.yaml"])
    assert args.variant == "full"
    assert args.profile == "auto_full"
    assert args.loss == "coffee_l4"
    assert args.bgmix == "auto"
    assert args.bgmix_close_epoch == 150
    assert args.imgsz == "960"
    assert args.epochs == 300
    assert args.batch == 96
    assert args.workers == 8
    assert args.seed == 0
    assert args.device == "0"
    assert args.optimizer == "MuSGD"
    assert args.fitness_metric == "map50-95"
    training = _training_overrides(args, (960, 960))
    assert training == {
        "epochs": 300,
        "imgsz": 960,
        "batch": 96,
        "workers": 8,
        "device": "0",
        "seed": 0,
        "optimizer": "MuSGD",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "warmup_epochs": 4.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "patience": 80,
        "nbs": 64,
        "mosaic": 0.7,
        "close_mosaic": 150,
        "hsv_h": 0.005,
        "hsv_s": 0.2,
        "hsv_v": 0.1,
        "scale": 0.35,
        "erasing": 0.0,
        "cache": False,
        "amp": True,
        "deterministic": True,
        "cos_lr": False,
    }


def test_monitor_headers_leave_native_results_csv_to_trainer(tmp_path):
    from coffee26n_experiment.train_coffee26n import _write_monitor_headers

    _write_monitor_headers(tmp_path)
    assert not (tmp_path / "results.csv").exists()
    assert (tmp_path / "per_class_metrics.csv").is_file()


def test_fitness_metric_selects_requested_validation_value():
    metrics = {
        "metrics/mAP50(B)": 0.71,
        "metrics/mAP50-95(B)": 0.44,
    }
    assert _select_fitness(metrics, "map50") == pytest.approx(0.71)
    assert _select_fitness(metrics, "map50-95") == pytest.approx(0.44)


def test_fitness_metric_cli_rejects_unknown_value():
    with pytest.raises(SystemExit):
        _parser().parse_args(["--data", "dataset.yaml", "--fitness-metric", "precision"])
