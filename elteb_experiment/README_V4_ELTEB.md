# V4 ELTEB Experiment

V4 targets early disease symptoms whose texture and edge cues are weak and easily confused.

## Module

`ELTEB` replaces the P3 `C3k2` block while preserving the original layer index. It reads the original RGB input
through a special YAML source `-2`, resizes it to the P3 feature scale, builds fixed texture cues, compresses them
with a learnable `1x1` path, and fuses them back into P3 as a residual.

- Full `ELTEB`: RGB + luminance + Sobel + Laplacian + unsharp, additive residual fusion.
- `ELTEB-Lite`: luminance + Sobel + Laplacian, additive residual fusion.
- Texture fusion is initialized at `0.0`, so the model starts close to the original P3 path and learns texture
  corrections during training.

## Ablation Names

| Model | Change | Suggested run name |
| --- | --- | --- |
| Native YOLO + ELTEB | Original YOLO26n with learnable P3 texture branch | `v4_native_elteb_coffee_self_sum` |
| EdgeLite + ELTEB | Original EdgeLite with learnable P3 texture branch | `v4_edgelite_elteb_coffee_self_sum` |
| EdgeLite-BiBridge + ELTEB | Latest bidirectional EdgeLite with learnable P3 texture branch | `v4_edgelite_bibridge_elteb_coffee_self_sum` |
| EdgeLite-BiBridge + WIoU/ProgLoss | Current V4 ablation baseline | `v4_edgelite_bibridge_wiou_progloss_baseline_coffee_self_sum` |
| EdgeLite-BiBridge + ELTEB + WIoU/ProgLoss | Texture branch plus new loss | `v4_edgelite_bibridge_wiou_progloss_elteb_coffee_self_sum` |

## Training

Unified V4 ablation entry:

```powershell
$env:V4_DATA="E:\ultralytics-8.4.43\coffee_self_sum\coffee_self_sum.yaml"
$env:V4_ARCH="edgelite_bibridge" # native, edgelite, edgelite_bibridge
$env:V4_VARIANT="baseline"      # baseline, clahe, laplacian, elteb, elteb_lite
python elteb_experiment/train_coffee_v4_ablation.py
```

The training scripts count train-label instances from the selected dataset YAML
and pass those counts to ProgLoss automatically.

`clahe` and `laplacian` automatically build preprocessing-only dataset copies under:

```text
elteb_experiment/preprocessed_data/
```

Build preprocessing datasets directly:

```powershell
python elteb_experiment/build_preprocessed_dataset.py --variant clahe
python elteb_experiment/build_preprocessed_dataset.py --variant laplacian
```

Legacy ELTEB-only entry:

```powershell
$env:V4_ELTEB_ARCH="native"      # native, edgelite, edgelite_bibridge
$env:V4_ELTEB_VARIANT="elteb"    # elteb, elteb_lite
$env:V4_ELTEB_LOSS="native"      # native, wiou_progloss
python elteb_experiment/train_coffee_v4_elteb.py
```

Lite variant:

```powershell
$env:V4_ELTEB_ARCH="edgelite_bibridge"
$env:V4_ELTEB_VARIANT="elteb_lite"
python elteb_experiment/train_coffee_v4_elteb.py
```

Smoke test:

```powershell
$env:V4_EPOCHS="3"
$env:V4_VARIANT="clahe"
python elteb_experiment/train_coffee_v4_ablation.py
```
