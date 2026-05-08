# tasks.py LGMSF-Lite Notes

- Imported `LDSConv` and `LGMSFBridge` from `ultralytics.nn.modules`.
- Added `LDSConv` to `base_modules` so it receives standard `c1, c2` parsing.
- Added a dedicated `LGMSFBridge` parse branch for two-input channel handling.
