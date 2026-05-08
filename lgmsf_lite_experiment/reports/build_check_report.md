# LGMSF-Lite Build Check Report

- Status: `passed`
- Config: `E:\计算机比赛\urp\ultralytics-src-8.4.14\lgmsf_lite_experiment\configs\yolo26n_lgmsf_lite.yaml`
- Parameters: `2468190`
- FLOPs: `7.314329600000001`
- Contains LDSConv: `True`
- Contains LGMSFBridge: `True`
- Dummy forward: `passed`

## Output Summary

```json
[
  [
    1,
    300,
    6
  ],
  {
    "one2many": {
      "boxes": [
        1,
        4,
        8400
      ],
      "scores": [
        1,
        6,
        8400
      ],
      "feats": [
        [
          1,
          64,
          80,
          80
        ],
        [
          1,
          128,
          40,
          40
        ],
        [
          1,
          256,
          20,
          20
        ]
      ]
    },
    "one2one": {
      "boxes": [
        1,
        4,
        8400
      ],
      "scores": [
        1,
        6,
        8400
      ],
      "feats": [
        [
          1,
          64,
          80,
          80
        ],
        [
          1,
          128,
          40,
          40
        ],
        [
          1,
          256,
          20,
          20
        ]
      ]
    }
  }
]
```
