# YOLO26n-EdgeLite Build Compare Report

| model | params | GFLOPs | layers | EdgeBridge | SimAM | dummy forward |
|---|---:|---:|---:|---|---|---|
| native | 2506140 | 5.782528 | 260 | False | False | passed |
| edgelite | 1951870 | 4.6626304 | 324 | True | False | passed |
| edgelite_simam | 1951870 | 4.6626304 | 324 | True | True | passed |

## JSON

```json
[
  {
    "name": "native",
    "cfg": "E:\\\u8ba1\u7b97\u673a\u6bd4\u8d5b\\urp\\ultralytics-src-8.4.14\\edgelite_experiment\\configs\\yolo26n_original_copy.yaml",
    "layers": 260,
    "params": 2506140,
    "trainable": 2506140,
    "flops": 5.782528,
    "has_ldsconv": false,
    "has_texture_stream_p3": false,
    "has_semantic_stream_p5": false,
    "has_fast_norm_fuse2": false,
    "has_simam_module": false,
    "has_edge_bridge": false,
    "output": [
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
  },
  {
    "name": "edgelite",
    "cfg": "E:\\\u8ba1\u7b97\u673a\u6bd4\u8d5b\\urp\\ultralytics-src-8.4.14\\edgelite_experiment\\configs\\yolo26n_edgelite.yaml",
    "layers": 324,
    "params": 1951870,
    "trainable": 1951870,
    "flops": 4.6626304,
    "has_ldsconv": true,
    "has_texture_stream_p3": true,
    "has_semantic_stream_p5": true,
    "has_fast_norm_fuse2": true,
    "has_simam_module": false,
    "has_edge_bridge": true,
    "output": [
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
  },
  {
    "name": "edgelite_simam",
    "cfg": "E:\\\u8ba1\u7b97\u673a\u6bd4\u8d5b\\urp\\ultralytics-src-8.4.14\\edgelite_experiment\\configs\\yolo26n_edgelite_simam.yaml",
    "layers": 324,
    "params": 1951870,
    "trainable": 1951870,
    "flops": 4.6626304,
    "has_ldsconv": true,
    "has_texture_stream_p3": true,
    "has_semantic_stream_p5": true,
    "has_fast_norm_fuse2": true,
    "has_simam_module": true,
    "has_edge_bridge": true,
    "output": [
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
  }
]
```
