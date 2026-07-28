# Coffee26n V5 generation guidance

This directory preserves the complete design and code-generation inputs used for the Coffee26n V5 experiment package.

| Document | Role |
|---|---|
| `coffee26n_codegen_multiagent_prompt.md` | Generation contract: package boundaries, artifacts, validation gates and multi-agent work split. |
| `yolo26n_module_architecture_guide.md` | Module-level architecture contract: tensor interfaces, insertion positions, gates, ablations and Detect constraints. |
| `coffee26n_optimization_plan.md` | Research plan: loss, augmentation, data-lock, monitoring, deployment and experiment sequencing requirements. |

The executable result is `../../coffee26n_experiment/`. The guidance files are preserved as provenance, not as runtime dependencies. When guidance and executable behavior are compared, the package source, resolved configuration and source manifest are the auditable implementation record.

Coffee3000 is the first intended V5 high-compute dataset, but it is not bundled here and is not claimed as locally validated. The cluster job must parse the actual `coffee3000.yaml`, inspect labels and split integrity, and write a fresh data lock before any result is treated as comparable.
