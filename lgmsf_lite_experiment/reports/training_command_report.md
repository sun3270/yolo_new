# LGMSF-Lite Training Command Report

- Status: `not run`
- Config YAML: `lgmsf_lite_experiment/configs/yolo26n_lgmsf_lite.yaml`
- Default data YAML: `coffee3000/coffee3000.yaml`
- Default project: `lgmsf_lite_experiment/runs_lgmsf_pc`
- Default name: `yolo26n_lgmsf_lite`

Recommended command:

```bash
python lgmsf_lite_experiment/06_train_lgmsf_pc.py --data coffee3000/coffee3000.yaml --epochs 150 --imgsz 640 --batch 16 --device 0 --workers 8 --pretrained yolo26n.pt
```

This file will be overwritten with actual run settings when `06_train_lgmsf_pc.py` starts training.
