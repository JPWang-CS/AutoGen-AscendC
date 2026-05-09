# AscendC Ops Expert Memory Index

- [ops-transformer Architecture](ops-transformer-architecture.md) -- ops-transformer 仓库架构、目录结构、构建系统、平台适配机制、Tiling 框架的完整参考
- [A5 GMM Finalize Routing Data Types](a5-gmm-finalize-routing-dtypes.md) -- Data types, quant modes, format restrictions for grouped_matmul_finalize_routing on A5/950
- [GMMFR Deterministic Cgmct Analysis](gmmfr-deterministic-cgmct-analysis.md) -- Cgmct vs A3 prototype differences for deterministic migration: workspace usage, Cube output path, Epilogue scatter behavior
- [GMMFR Deterministic Prologue/Epilogue Conflict](gmmfr-deterministic-prologue-epilogue-conflict.md) -- Root cause and fix for Prologue/Epilogue writing to same workspace buffer in deterministic mode
- [GMMFR Deterministic Implementation](gmmfr-deterministic-implementation.md) -- Final implementation: data flow, design decisions, file changes, SyncAll counts
