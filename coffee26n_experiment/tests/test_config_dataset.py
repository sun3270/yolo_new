from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from coffee26n_experiment.config import deep_merge, resolve_profile, resolve_profile_runtime
from coffee26n_experiment.dataset import parse_dataset
from coffee26n_experiment.rebuild_strict_dataset import rebuild_dataset


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("nc", (1, 6, 80))
def test_fixture_dataset_locks(nc):
    dataset = parse_dataset(ROOT / "tests" / "fixtures" / f"nc{nc}" / "data.yaml")
    assert dataset["nc"] == nc
    assert dataset["data_lock_id"]
    assert dataset["splits"]["train"] and dataset["splits"]["val"]


def test_unknown_profile_keys_fail_closed():
    with pytest.raises(ValueError, match="Unknown profile/config key"):
        deep_merge({"name": "generic"}, {"unknown": True})


def test_named_profile_resolves_to_runtime_ids():
    profile, _ = resolve_profile(ROOT / "configs" / "profiles" / "coffee_ant6.yaml")
    profile_names = list(profile["expected_names"])
    profile_counts = profile["tail_bce"]["counts"]
    dataset = {"names": profile_names, "class_counts": {profile_names.index(name): count for name, count in profile_counts.items()}}
    runtime = resolve_profile_runtime(profile, dataset)
    assert runtime["class_counts"][profile_names.index(profile_names[1])] == float(profile_counts[profile_names[1]])
    assert (profile_names.index(profile_names[1]), profile_names.index(profile_names[0])) in runtime["pair_ids"]


def test_named_profile_count_drift_fails_closed():
    profile, _ = resolve_profile(ROOT / "configs" / "profiles" / "coffee_ant6.yaml")
    dataset = {"names": list(profile["expected_names"]), "class_counts": {i: 1 for i in range(len(profile["expected_names"]))}}
    with pytest.raises(ValueError, match="do not match current train labels"):
        resolve_profile_runtime(profile, dataset)


def test_auto_full_profile_uses_current_dataset_statistics():
    profile, _ = resolve_profile("auto_full")
    dataset = {
        "names": ["dominant", "medium", "rare"],
        "class_counts": {0: 1000, 1: 500, 2: 100},
    }
    runtime = resolve_profile_runtime(profile, dataset)
    assert runtime["class_counts"] == {0: 1000.0, 1: 500.0, 2: 100.0}
    assert runtime["pair_ids"] == [(1, 0), (2, 0)]


def test_auto_full_profile_does_not_invent_pairs_for_balanced_data():
    profile, _ = resolve_profile("auto_full")
    runtime = resolve_profile_runtime(profile, {"names": ["a", "b"], "class_counts": {0: 10, 1: 10}})
    assert runtime["pair_ids"] == []


def test_dataset_rejects_noncontiguous_name_mapping(tmp_path):
    path = tmp_path / "data.yaml"
    path.write_text("nc: 2\nnames: {0: first, 2: third}\ntrain: images\nval: images\n", encoding="utf-8")
    with pytest.raises(ValueError, match="contiguous"):
        parse_dataset(path)


def test_data_lock_changes_with_image_content_and_rejects_cross_split_duplicate(tmp_path):
    source = ROOT / "tests" / "fixtures" / "nc1"
    target = tmp_path / "fixture"
    shutil.copytree(source, target)
    original = parse_dataset(target / "data.yaml")
    image = target / "images" / "train" / "bus_a.jpg"
    image.write_bytes(image.read_bytes() + b"lock-change")
    changed = parse_dataset(target / "data.yaml")
    assert changed["data_lock_id"] != original["data_lock_id"]
    duplicate = target / "images" / "val" / "duplicate.jpg"
    duplicate.write_bytes(image.read_bytes())
    (target / "labels" / "val" / "duplicate.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Cross-split duplicate"):
        parse_dataset(target / "data.yaml")


def test_missing_label_file_is_a_valid_background_image(tmp_path):
    source = ROOT / "tests" / "fixtures" / "nc1"
    target = tmp_path / "fixture"
    shutil.copytree(source, target)
    label = target / "labels" / "train" / "bus_a.txt"
    label.unlink()
    dataset = parse_dataset(target / "data.yaml")
    record = next(item for item in dataset["records"]["train"] if item["image"].endswith("bus_a.jpg"))
    assert record["label_exists"] is False
    assert record["boxes"] == []
    assert record["image"] in dataset["empty_label_images"]["train"]


def test_clean_dataset_lock_matches_hpc_snapshot_when_dataset_is_available():
    yaml_path = ROOT.parent / "coffee_self_sum_5cls_yolo_grouped_701515_clean" / "coffee_self_sum_5cls_yolo_grouped_701515_clean.yaml"
    if not yaml_path.is_file():
        pytest.skip("Full clean coffee dataset is not installed in this workspace.")
    dataset = parse_dataset(yaml_path)
    assert dataset["data_lock_id"] == "aa319dc094e763abfa7fab56504842c09ba494a62cece06fee58bb0401ed581e"


def test_strict_dataset_rebuild_uses_manifest_hardlinks(tmp_path):
    source = tmp_path / "coffee_self_sum_5cls_yolo_grouped_701515"
    for split in ("train", "val", "test"):
        (source / "images" / split).mkdir(parents=True)
        (source / "labels" / split).mkdir(parents=True)
    (source / "coffee_self_sum_5cls_yolo_grouped_701515.yaml").write_text(
        "train: images/train\nval: images/val\ntest: images/test\nnc: 5\n"
        "names: {0: ANT_AB, 1: ANT_CD, 2: BLS, 3: CR, 4: SM}\n",
        encoding="utf-8",
    )
    rows = []
    expected = {"train": 2096, "val": 423, "test": 416}
    index = 0
    for final_split, count in expected.items():
        for _ in range(count):
            source_split = ("train", "val", "test")[index % 3]
            filename = f"sample_{index:04d}.jpg"
            image = source / "images" / source_split / filename
            label = source / "labels" / source_split / Path(filename).with_suffix(".txt")
            image.write_bytes(f"image-{index}".encode())
            label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            rows.append(f"{filename},True,{source_split},{final_split}")
            index += 1
    manifest = tmp_path / "split_manifest.csv"
    manifest.write_text("image_filename,included,source_final_split,final_split\n" + "\n".join(rows) + "\n", encoding="utf-8")
    output = tmp_path / "clean"
    report = rebuild_dataset(source, manifest, output)
    assert report["splits"] == expected
    first_source = source / "images" / "train" / "sample_0000.jpg"
    first_target = output / "images" / "train" / "sample_0000.jpg"
    assert first_source.stat().st_ino == first_target.stat().st_ino
    assert parse_dataset(output / "coffee_self_sum_5cls_yolo_grouped_701515_clean.yaml")["nc"] == 5
