---
name: Project Structure
description: Repository layout, key directories, operator projects, and reference code locations
type: project
---

## Repository Root: m:\Desktop\tmp\AgentTest\AscendC

### Key Directories
- `project/` -- Individual operator development projects (one subdir per operator)
- `ops-transformer_AI/` -- Upstream reference code repo (independent git repo), contains ~50+ implemented Transformer operators
  - `attention/` (~40+ operators): flash_attention, sparse_flash_attention, mla_prolog, nsa_compress, etc.
  - `gmm/` (8 operators): grouped_matmul, grouped_matmul_finalize_routing and variants
  - `ffn/`, `moe/`, `mc2/`, `mhc/` -- other categories
  - `common/` -- shared components
- `template/` -- Operator dev templates: `aic_only/`, `aiv_only/`, `aiv_aic_mixed/`
- `skills/` -- Agent knowledge docs (API guide, platform differences, programming model, operator structure, GMMFR analysis)

### Important Notes
- `ops-transformer_AI/` is an independent git repo, NOT tracked by the main AscendC repo
- The main repo has only 2 commits (as of 2026-05-08), project is in early stage
- `gmm/grouped_matmul_finalize_routing/` in ops-transformer_AI already has `arch35` examples and `950_checker` files

### Platform Architecture Reference
- A2 (910B) / A3 (910C): Code generally shared, same programming paradigm
- A5 (950): arch35 in ops-transformer_AI, MUST be developed independently from A2/A3
