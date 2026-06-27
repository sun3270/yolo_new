from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml
from PIL import Image


def write_image(path: Path, size: tuple[int, int] = (128, 128), checker: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, (48, 96, 48))
    if checker:
        px = image.load()
        for y in range(size[1]):
            for x in range(size[0]):
                value = 245 if (x + y) % 2 == 0 else 12
                px[x, y] = (value, value, value)
    image.save(path)


def write_label(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "coffee_self_sum"
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)

    write_image(root / "images" / "train" / "tail_source.jpg")
    write_label(root / "labels" / "train" / "tail_source.txt", ["2 0.250000 0.250000 0.125000 0.125000"])
    write_image(root / "images" / "train" / "background.jpg")
    write_label(root / "labels" / "train" / "background.txt", ["0 0.700000 0.700000 0.150000 0.150000"])

    write_image(root / "images" / "val" / "edge_small.jpg")
    write_label(root / "labels" / "val" / "edge_small.txt", ["2 0.020000 0.500000 0.040000 0.040000"])

    write_image(root / "images" / "val" / "multi.jpg")
    write_label(
        root / "labels" / "val" / "multi.txt",
        [
            "0 0.200000 0.200000 0.100000 0.100000",
            "0 0.400000 0.200000 0.100000 0.100000",
            "1 0.600000 0.200000 0.100000 0.100000",
            "2 0.800000 0.200000 0.100000 0.100000",
        ],
    )

    write_image(root / "images" / "val" / "clutter.jpg", checker=True)
    write_label(root / "labels" / "val" / "clutter.txt", ["1 0.500000 0.500000 0.200000 0.200000"])

    write_image(root / "images" / "test" / "plain.jpg")
    write_label(root / "labels" / "test" / "plain.txt", ["0 0.500000 0.500000 0.300000 0.300000"])

    data_yaml = root / "coffee_self_sum.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "nc": 3,
                "names": {0: "ANT_AB", 1: "CPD", 2: "SM_CD"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return data_yaml


def test_dataset_diagnostics_flags_recall_hard_cases(tmp_path: Path) -> None:
    from coffee_v4_training.dataset_diagnostics import analyze_dataset

    data_yaml = make_dataset(tmp_path)
    output = tmp_path / "diagnostics"

    report = analyze_dataset(
        data_yaml,
        output,
        splits=("val",),
        tail_class_ids={2},
        small_area_threshold=0.01,
        edge_margin=0.05,
        multi_instance_threshold=3,
        clutter_edge_density_threshold=0.20,
    )

    assert report["splits"]["val"]["images"] == 3
    assert report["splits"]["val"]["instances"] == 6
    assert report["class_counts"]["2"]["count"] == 2

    rows = {row["image"]: row for row in csv.DictReader((output / "hard_cases.csv").open(encoding="utf-8"))}
    assert "small_edge" in rows["images/val/edge_small.jpg"]["hard_case_tags"]
    assert "bad_border" in rows["images/val/edge_small.jpg"]["hard_case_tags"]
    assert "tail_class" in rows["images/val/edge_small.jpg"]["hard_case_tags"]
    assert "multi_instance" in rows["images/val/multi.jpg"]["hard_case_tags"]
    assert "clutter_bg" in rows["images/val/clutter.jpg"]["hard_case_tags"]

    saved_report = json.loads((output / "dataset_report.json").read_text(encoding="utf-8"))
    assert saved_report["hard_case_counts"]["small_edge"] == 1
    assert saved_report["hard_case_counts"]["clutter_bg"] == 1
    assert saved_report["hard_case_images"]["small_edge"] == ["images/val/edge_small.jpg"]
    assert saved_report["image_object_count"]["images/val/multi.jpg"] == 4
    assert saved_report["label_quality"]["warning_count"] == 0

    paper_report = (output / "paper_requirements_report.md").read_text(encoding="utf-8")
    assert "标注边缘噪声" in paper_report
    assert "复杂背景干扰" in paper_report
    assert "多主体遮挡" in paper_report
    assert "small_edge" in paper_report


def test_copy_paste_builder_writes_derived_dataset_without_touching_source(tmp_path: Path) -> None:
    from coffee_v4_training.build_recall_datasets import build_copy_paste_dataset

    data_yaml = make_dataset(tmp_path)
    original_train_images = sorted((data_yaml.parent / "images" / "train").glob("*.jpg"))

    output_yaml = build_copy_paste_dataset(
        data_yaml,
        tmp_path / "derived" / "coffee_self_sum_copypaste",
        target_class_ids={2},
        copies_per_source=1,
        seed=3,
        overwrite=True,
    )

    assert output_yaml.exists()
    assert sorted((data_yaml.parent / "images" / "train").glob("*.jpg")) == original_train_images
    generated_images = sorted((output_yaml.parent / "images" / "train").glob("cp_*.jpg"))
    generated_labels = sorted((output_yaml.parent / "labels" / "train").glob("cp_*.txt"))
    assert len(generated_images) == 1
    assert len(generated_labels) == 1
    assert all(0.0 <= float(v) <= 1.0 for line in generated_labels[0].read_text().splitlines() for v in line.split()[1:5])

    manifest = json.loads((output_yaml.parent / "recall_dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["variant"] == "copypaste"
    assert manifest["source_dataset_yaml"] == str(data_yaml.resolve())
    assert manifest["parameters"]["target_class_ids"] == [2]
    assert manifest["outputs"]["generated_train_images"] == 1
    assert manifest["source_mutation"] == "read_only"
    derived_cfg = yaml.safe_load(output_yaml.read_text(encoding="utf-8"))
    assert isinstance(derived_cfg["train"], list)
    assert str((data_yaml.parent / "images" / "train").resolve()).replace("\\", "/") in derived_cfg["train"]
    assert str((output_yaml.parent / "images" / "train").resolve()).replace("\\", "/") in derived_cfg["train"]
    assert derived_cfg["val"] == str((data_yaml.parent / "images" / "val").resolve()).replace("\\", "/")


def test_dataset_diagnostics_handles_list_train_entries_outside_dataset_root(tmp_path: Path) -> None:
    from coffee_v4_training.build_recall_datasets import build_copy_paste_dataset
    from coffee_v4_training.dataset_diagnostics import analyze_dataset

    data_yaml = make_dataset(tmp_path)
    output_yaml = build_copy_paste_dataset(
        data_yaml,
        tmp_path / "derived" / "coffee_self_sum_copypaste",
        target_class_ids={2},
        copies_per_source=1,
        seed=3,
        overwrite=True,
    )

    report = analyze_dataset(output_yaml, tmp_path / "diagnostics", splits=("train",), tail_class_ids={2})

    assert report["splits"]["train"]["images"] == 3
    rows = list(csv.DictReader((tmp_path / "diagnostics" / "hard_cases.csv").open(encoding="utf-8")))
    assert rows
    assert all(row["image"] for row in rows)


def test_context_copy_paste_uses_weighted_targets_and_records_rejections(tmp_path: Path) -> None:
    from coffee_v4_training.build_recall_datasets import build_context_copy_paste_dataset

    data_yaml = make_dataset(tmp_path)
    original_train_images = sorted((data_yaml.parent / "images" / "train").glob("*.jpg"))

    output_yaml = build_context_copy_paste_dataset(
        data_yaml,
        tmp_path / "derived" / "coffee_self_sum_copypaste_context",
        target_class_ids={2},
        copy_class_weights={2: 2},
        copies_per_source=1,
        max_paste_iou=0.15,
        max_paste_attempts=40,
        seed=7,
        overwrite=True,
    )

    assert output_yaml.exists()
    assert sorted((data_yaml.parent / "images" / "train").glob("*.jpg")) == original_train_images
    generated_images = sorted((output_yaml.parent / "images" / "train").glob("cpc_*.jpg"))
    generated_labels = sorted((output_yaml.parent / "labels" / "train").glob("cpc_*.txt"))
    assert len(generated_images) == 2
    assert len(generated_labels) == 2
    assert all(0.0 <= float(v) <= 1.0 for line in generated_labels[0].read_text().splitlines() for v in line.split()[1:5])

    manifest = json.loads((output_yaml.parent / "recall_dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["variant"] == "copypaste_context"
    assert manifest["parameters"]["copy_class_weights"] == {"2": 2}
    assert manifest["parameters"]["max_paste_iou"] == 0.15
    assert manifest["parameters"]["max_paste_attempts"] == 40
    assert manifest["outputs"]["accepted_pastes"] == 2
    assert "rejected_paste_attempts" in manifest["outputs"]
    assert manifest["source_mutation"] == "read_only"


def test_sahi_builder_writes_valid_tiles_and_labels(tmp_path: Path) -> None:
    from coffee_v4_training.build_recall_datasets import build_sahi_dataset

    data_yaml = make_dataset(tmp_path)

    output_yaml = build_sahi_dataset(
        data_yaml,
        tmp_path / "derived" / "coffee_self_sum_sahi",
        tile_size=64,
        overlap=0.25,
        splits=("train",),
        min_visibility=0.25,
        overwrite=True,
    )

    assert output_yaml.exists()
    tile_images = sorted((output_yaml.parent / "images" / "train").glob("*.jpg"))
    tile_labels = sorted((output_yaml.parent / "labels" / "train").glob("*.txt"))
    assert len(tile_images) > 2
    assert any(path.read_text(encoding="utf-8").strip() for path in tile_labels)

    manifest = json.loads((output_yaml.parent / "recall_dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["variant"] == "sahi"
    assert manifest["parameters"]["tile_size"] == 64
    assert manifest["outputs"]["generated_train_images"] == len(tile_images)


def test_sahi_trainonly_references_source_eval_splits(tmp_path: Path) -> None:
    from coffee_v4_training.build_recall_datasets import build_sahi_trainonly_dataset

    data_yaml = make_dataset(tmp_path)

    output_yaml = build_sahi_trainonly_dataset(
        data_yaml,
        tmp_path / "derived" / "coffee_self_sum_sahi_trainonly",
        tile_size=64,
        overlap=0.25,
        min_visibility=0.25,
        overwrite=True,
    )

    cfg = yaml.safe_load(output_yaml.read_text(encoding="utf-8"))
    assert cfg["train"] == str((output_yaml.parent / "images" / "train").resolve()).replace("\\", "/")
    assert cfg["val"] == str((data_yaml.parent / "images" / "val").resolve()).replace("\\", "/")
    assert cfg["test"] == str((data_yaml.parent / "images" / "test").resolve()).replace("\\", "/")
    assert len(list((output_yaml.parent / "images" / "train").glob("*.jpg"))) > 2
    assert not list((output_yaml.parent / "images" / "val").glob("*.jpg"))
    manifest = json.loads((output_yaml.parent / "recall_dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["variant"] == "sahi_trainonly"
    assert manifest["outputs"]["val_images"] == "source_reference"


def test_roi_builder_references_source_eval_splits(tmp_path: Path) -> None:
    from coffee_v4_training.build_recall_datasets import build_roi_dataset

    data_yaml = make_dataset(tmp_path)

    output_yaml = build_roi_dataset(
        data_yaml,
        tmp_path / "derived" / "coffee_self_sum_roi",
        overwrite=True,
    )

    cfg = yaml.safe_load(output_yaml.read_text(encoding="utf-8"))
    assert cfg["train"] == str((output_yaml.parent / "images" / "train").resolve()).replace("\\", "/")
    assert cfg["val"] == str((data_yaml.parent / "images" / "val").resolve()).replace("\\", "/")
    assert cfg["test"] == str((data_yaml.parent / "images" / "test").resolve()).replace("\\", "/")
    assert len(list((output_yaml.parent / "images" / "train").glob("*.jpg"))) == 2
    assert not list((output_yaml.parent / "images" / "val").glob("*.jpg"))


def test_roi_leafmask_preserves_labels_and_source_eval_splits(tmp_path: Path) -> None:
    from coffee_v4_training.build_recall_datasets import build_roi_leafmask_dataset

    data_yaml = make_dataset(tmp_path)

    output_yaml = build_roi_leafmask_dataset(
        data_yaml,
        tmp_path / "derived" / "coffee_self_sum_roi_leafmask",
        overwrite=True,
    )

    cfg = yaml.safe_load(output_yaml.read_text(encoding="utf-8"))
    assert cfg["train"] == str((output_yaml.parent / "images" / "train").resolve()).replace("\\", "/")
    assert cfg["val"] == str((data_yaml.parent / "images" / "val").resolve()).replace("\\", "/")
    assert cfg["test"] == str((data_yaml.parent / "images" / "test").resolve()).replace("\\", "/")
    source_labels = sorted((data_yaml.parent / "labels" / "train").glob("*.txt"))
    derived_labels = sorted((output_yaml.parent / "labels" / "train").glob("*.txt"))
    assert len(derived_labels) == len(source_labels)
    assert [path.name for path in derived_labels] == [path.name for path in source_labels]
    manifest = json.loads((output_yaml.parent / "recall_dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["variant"] == "roi_leafmask"
    assert manifest["parameters"]["preserve_gt_boxes"] is True


def test_sanitize_labels_clips_out_of_bounds_boxes_without_mutating_source(tmp_path: Path) -> None:
    from coffee_v4_training.build_recall_datasets import LabelBox, build_sanitize_labels_dataset, format_label

    data_yaml = make_dataset(tmp_path)
    source_label = data_yaml.parent / "labels" / "train" / "tail_source.txt"
    source_label.write_text("2 0.020000 0.500000 0.100000 0.200000\n", encoding="utf-8")

    output_yaml = build_sanitize_labels_dataset(
        data_yaml,
        tmp_path / "derived" / "coffee_self_sum_sanitize_labels",
        overwrite=True,
    )

    assert source_label.read_text(encoding="utf-8") == "2 0.020000 0.500000 0.100000 0.200000\n"
    fixed_label = output_yaml.parent / "labels" / "train" / "tail_source.txt"
    values = [float(v) for v in fixed_label.read_text(encoding="utf-8").split()[1:]]
    x, y, w, h = values
    assert x - w / 2 >= 0.0
    assert y - h / 2 >= 0.0
    assert x + w / 2 <= 1.0
    assert y + h / 2 <= 1.0
    assert (output_yaml.parent / "images" / "train" / "tail_source.jpg").exists()

    manifest = json.loads((output_yaml.parent / "recall_dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["variant"] == "sanitize_labels"
    assert manifest["outputs"]["clipped_boxes"] == 1
    assert manifest["source_mutation"] == "read_only"

    serialized = format_label([LabelBox(cls=0, x=0.925390625, y=0.5, w=0.14921875, h=0.2)])
    _, sx, sy, sw, sh = serialized.split()
    x, y, w, h = (float(sx), float(sy), float(sw), float(sh))
    assert x - w / 2 >= 0.0
    assert y - h / 2 >= 0.0
    assert x + w / 2 <= 1.0
    assert y + h / 2 <= 1.0


def test_dataset_path_remap_resolves_absolute_yaml_entries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from coffee_v4_training.dataset_diagnostics import load_dataset, resolve_split_dirs

    data_yaml = make_dataset(tmp_path)
    actual_root = data_yaml.parent.resolve()
    old_root = "C:/Users/1/Desktop/coffee/datasets/coffee_self_sum"
    portable_yaml = tmp_path / "portable.yaml"
    portable_yaml.write_text(
        yaml.safe_dump(
            {
                "path": old_root,
                "train": f"{old_root}/images/train",
                "val": "images/val",
                "test": f"{old_root}/images/test",
                "nc": 3,
                "names": {0: "ANT_AB", 1: "CPD", 2: "SM_CD"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("COFFEE_PATH_REMAP", f"{old_root}=>{actual_root.as_posix()}")
    info = load_dataset(portable_yaml)

    assert info.root == actual_root
    assert resolve_split_dirs(info, "train") == [actual_root / "images" / "train"]
    assert resolve_split_dirs(info, "val") == [actual_root / "images" / "val"]
    assert resolve_split_dirs(info, "test") == [actual_root / "images" / "test"]


def test_train_materializes_path_remapped_yaml(tmp_path: Path) -> None:
    from coffee_v4_training.train import materialize_remapped_dataset_yaml

    data_yaml = make_dataset(tmp_path)
    old_root = "C:/Users/1/Desktop/coffee/datasets/coffee_self_sum"
    actual_root = data_yaml.parent.resolve()
    portable_yaml = tmp_path / "portable_train.yaml"
    portable_yaml.write_text(
        yaml.safe_dump(
            {
                "path": old_root,
                "train": f"{old_root}/images/train",
                "val": "images/val",
                "test": f"{old_root}/images/test",
                "nc": 3,
                "names": {0: "ANT_AB", 1: "CPD", 2: "SM_CD"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    remapped = materialize_remapped_dataset_yaml(
        portable_yaml,
        tmp_path / "runs" / "train",
        "portable_run",
        f"{old_root}=>{actual_root.as_posix()}",
    )
    cfg = yaml.safe_load(remapped.read_text(encoding="utf-8"))

    assert remapped != portable_yaml
    assert cfg["path"] == actual_root.as_posix()
    assert cfg["train"] == f"{actual_root.as_posix()}/images/train"
    assert cfg["val"] == "images/val"
    assert cfg["test"] == f"{actual_root.as_posix()}/images/test"


def test_sam2_review_writes_review_csv_and_preview_without_mutating_source(tmp_path: Path) -> None:
    from coffee_v4_training.sam2_refine_labels import build_sam2_review_dataset

    data_yaml = make_dataset(tmp_path)
    source_label = data_yaml.parent / "labels" / "val" / "edge_small.txt"
    original_text = source_label.read_text(encoding="utf-8")

    output_yaml = build_sam2_review_dataset(
        data_yaml,
        tmp_path / "derived" / "coffee_self_sum_sam_refined",
        mode="review",
        iou_gate=0.90,
        overwrite=True,
        max_previews=10,
    )

    assert output_yaml.exists()
    assert source_label.read_text(encoding="utf-8") == original_text
    cfg = yaml.safe_load(output_yaml.read_text(encoding="utf-8"))
    assert cfg["train"] == "images/train"
    assert cfg["val"] == "images/val"
    assert cfg["test"] == "images/test"

    review_csv = output_yaml.parent / "review_candidates.csv"
    rows = list(csv.DictReader(review_csv.open(encoding="utf-8")))
    assert rows
    assert {
        "split",
        "image",
        "class_id",
        "class_name",
        "original_x",
        "candidate_x",
        "iou",
        "needs_review",
        "review_reason",
        "sam_status",
    }.issubset(rows[0])
    assert any(row["needs_review"] == "1" and "border_touching_box" in row["review_reason"] for row in rows)
    assert list((output_yaml.parent / "review_previews").glob("*.jpg"))

    manifest = json.loads((output_yaml.parent / "sam2_refine_manifest.json").read_text(encoding="utf-8"))
    assert manifest["variant"] == "sam_refined"
    assert manifest["source_mutation"] == "read_only"
    assert manifest["parameters"]["mode"] == "review"
    assert manifest["outputs"]["review_csv"] == str(review_csv.resolve())


def test_inference_sweep_configs_cover_nms_and_sahi_modes() -> None:
    from coffee_v4_training.inference_recall_sweep import build_sweep_configs, effective_probe_conf

    configs = build_sweep_configs(
        confs=[0.05, 0.25],
        ious=[0.5],
        max_dets=[100, 300],
        nms_modes=["standard", "soft"],
        sahi_tile_sizes=[0, 640],
    )

    assert len(configs) == 16
    assert any(config.nms_mode == "soft" for config in configs)
    assert any(config.sahi_tile_size == 640 for config in configs)
    assert any(config.conf == 0.05 and config.max_det == 300 for config in configs)
    assert effective_probe_conf(configs[-1], probe_conf=0.03) == pytest.approx(0.03)
    assert effective_probe_conf(configs[0], probe_conf=0.10) == pytest.approx(0.05)


def test_inference_sweep_matches_predictions_to_ground_truth(tmp_path: Path) -> None:
    from coffee_v4_training.inference_recall_sweep import DetectionBox, evaluate_image_predictions, write_recall_summary

    data_yaml = make_dataset(tmp_path)
    image = data_yaml.parent / "images" / "val" / "multi.jpg"
    predictions = [
        DetectionBox(cls=0, conf=0.90, x=0.200000, y=0.200000, w=0.100000, h=0.100000),
        DetectionBox(cls=1, conf=0.70, x=0.600000, y=0.200000, w=0.100000, h=0.100000),
        DetectionBox(cls=2, conf=0.02, x=0.800000, y=0.200000, w=0.100000, h=0.100000),
        DetectionBox(cls=0, conf=0.80, x=0.900000, y=0.900000, w=0.100000, h=0.100000),
    ]

    result = evaluate_image_predictions(
        image=image,
        predictions=predictions,
        nc=3,
        match_iou=0.50,
        candidate_conf_floor=0.03,
    )

    assert result.gt_total == 4
    assert result.detection_total == 3
    assert result.matched_total == 2
    assert result.false_positive_total == 1
    assert result.precision == pytest.approx(2 / 3)
    assert result.recall == pytest.approx(0.5)
    assert result.low_conf_candidate_total == 1
    assert result.per_class["0"]["detections"] == 2
    assert result.per_class["0"]["matched"] == 1
    assert result.per_class["0"]["false_positives"] == 1
    assert result.per_class["1"]["matched"] == 1
    assert result.per_class["2"]["missed"] == 1

    out = tmp_path / "sweep"
    path = write_recall_summary({"cfg": [result]}, out, ["ANT_AB", "CPD", "SM_CD"], {"cfg": "standard"})
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["configs"]["cfg"]["recall"] == pytest.approx(0.5)
    assert summary["configs"]["cfg"]["precision"] == pytest.approx(2 / 3)
    assert summary["configs"]["cfg"]["false_positives"] == 1
    assert summary["configs"]["cfg"]["classes"]["SM_CD"]["low_conf_candidates"] == 1
    assert summary["configs"]["cfg"]["classes"]["ANT_AB"]["precision"] == pytest.approx(0.5)


def test_copy_paste_default_targets_low_recall_classes() -> None:
    from coffee_v4_training.build_recall_datasets import DEFAULT_CONTEXT_COPY_CLASS_WEIGHTS, DEFAULT_COPY_PASTE_TARGET_CLASSES, parse_args

    assert DEFAULT_COPY_PASTE_TARGET_CLASSES == "1,2,3,5"
    args = parse_args(["--variant", "copypaste", "--data", "dataset.yaml"])
    assert args.target_classes == DEFAULT_COPY_PASTE_TARGET_CLASSES
    args = parse_args(["--variant", "copypaste_context", "--data", "dataset.yaml"])
    assert args.copy_class_weights == DEFAULT_CONTEXT_COPY_CLASS_WEIGHTS


def test_soft_nms_configs_are_explicitly_marked_unsupported(tmp_path: Path) -> None:
    from coffee_v4_training.inference_recall_sweep import SweepConfig, write_recall_summary

    path = write_recall_summary(
        {"soft_cfg": []},
        tmp_path / "sweep",
        ["ANT_AB"],
        {"soft_cfg": SweepConfig(0.05, 0.50, 100, nms_mode="soft").nms_status},
    )
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["configs"]["soft_cfg"]["nms_status"] == "unsupported_soft_nms_requires_raw_predictions"


def test_recall_sweep_uses_edgelite_local_ultralytics() -> None:
    import ultralytics

    import coffee_v4_training.inference_recall_sweep as sweep

    local_ultralytics = Path(sweep.LOCAL_ULTRALYTICS).resolve()
    imported_path = Path(ultralytics.__file__).resolve()
    assert imported_path.is_relative_to(local_ultralytics)


def test_recall_experiments_are_registered() -> None:
    from coffee_v4_training.train import EXPERIMENTS, RECALL_EXPERIMENTS, STRONG_EXPERIMENTS

    assert RECALL_EXPERIMENTS == (
        "edgelite_bibridge_wiou_sahi",
        "edgelite_bibridge_wiou_copypaste",
        "edgelite_bibridge_wiou_roi",
        "edgelite_bibridge_wiou_sanitize_labels",
        "edgelite_bibridge_wiou_sanitize_copypaste_context",
        "edgelite_bibridge_wiou_sam_refined",
        "edgelite_bibridge_wiou_copypaste_context",
        "edgelite_bibridge_wiou_roi_leafmask",
        "edgelite_bibridge_nwd",
    )
    assert EXPERIMENTS["edgelite_bibridge_wiou_sahi"].preprocess == "sahi"
    assert EXPERIMENTS["edgelite_bibridge_wiou_copypaste"].preprocess == "copypaste"
    assert EXPERIMENTS["edgelite_bibridge_wiou_roi"].preprocess == "roi"
    assert EXPERIMENTS["edgelite_bibridge_wiou_sanitize_labels"].preprocess == "sanitize_labels"
    assert EXPERIMENTS["edgelite_bibridge_wiou_sanitize_copypaste_context"].preprocess == "sanitize_copypaste_context"
    assert EXPERIMENTS["edgelite_bibridge_wiou_sam_refined"].preprocess == "sam_refined"
    assert EXPERIMENTS["edgelite_bibridge_wiou_copypaste_context"].preprocess == "copypaste_context"
    assert EXPERIMENTS["edgelite_bibridge_wiou_sahi_trainonly"].preprocess == "sahi_trainonly"
    assert EXPERIMENTS["edgelite_bibridge_wiou_roi_leafmask"].preprocess == "roi_leafmask"
    assert EXPERIMENTS["edgelite_bibridge_nwd"].loss == "wiou_progloss_nwd"

    assert STRONG_EXPERIMENTS == (
        "edgelite_bibridge_wiou_y26_recipe",
        "edgelite_bibridge_wiou_stal_lite",
        "edgelite_bibridge_wiou_p2",
        "edgelite_hyperace_wiou",
        "edgelite_hyperace_p2_wiou",
    )
    assert EXPERIMENTS["edgelite_bibridge_wiou_y26_recipe"].train_overrides["optimizer"] == "MuSGD"
    assert EXPERIMENTS["edgelite_bibridge_wiou_stal_lite"].loss_overrides["small_box_topk_boost"] is True
    assert EXPERIMENTS["edgelite_bibridge_wiou_p2"].cfg.name == "yolo26n_edgelite_bibridge_p2.yaml"
    assert EXPERIMENTS["edgelite_hyperace_wiou"].cfg.name == "yolo26n_edgelite_hyperace.yaml"
    assert EXPERIMENTS["edgelite_hyperace_p2_wiou"].cfg.name == "yolo26n_edgelite_hyperace_p2.yaml"


def test_train_auto_resume_prefers_existing_last_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from argparse import Namespace

    from coffee_v4_training.train import resolve_resume_checkpoint

    project = tmp_path / "runs" / "train"
    last = project / "existing_run" / "weights" / "last.pt"
    last.parent.mkdir(parents=True)
    last.write_bytes(b"checkpoint")

    args = Namespace(resume=False, resume_from=None, fresh=False, auto_resume=None)
    monkeypatch.delenv("COFFEE_AUTO_RESUME", raising=False)
    assert resolve_resume_checkpoint(args, project, "existing_run") == last

    args.fresh = True
    assert resolve_resume_checkpoint(args, project, "existing_run") is None

    args.fresh = False
    monkeypatch.setenv("COFFEE_AUTO_RESUME", "0")
    assert resolve_resume_checkpoint(args, project, "existing_run") is None

    args.resume = True
    assert resolve_resume_checkpoint(args, project, "existing_run") == last
