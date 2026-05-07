---
name: A5 GMM Finalize Routing Data Type Support
description: Detailed analysis of data types, quantization modes, and format restrictions for grouped_matmul_finalize_routing on A5/950 (arch35)
type: reference
---

# A5 (950) Grouped Matmul Finalize Routing - Data Type Support

Source: `ops-transformer_AI/gmm/grouped_matmul_finalize_routing/`

## Two Quantization Modes

1. **MX (MicroScaling) Quantization** - QuantMode::MX = 2
2. **PerToken Quantization** - QuantMode::PERTOEKN = 0

Mode is determined by `scale` dtype: if scale is DT_FLOAT8_E8M0 -> MX; if scale is DT_FLOAT/DT_BF16 -> PerToken.

## MX Quant Mode - Supported Data Types

| Parameter | Supported Types |
|-----------|----------------|
| x (activation) | DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_FLOAT4_E2M1 |
| weight | DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_FLOAT4_E2M1 |
| scale | DT_FLOAT8_E8M0 |
| pertoken_scale | DT_FLOAT8_E8M0 (required, not optional) |
| bias | DT_BF16 |
| groupList | DT_INT64 |
| sharedInput | DT_BF16 |
| logit | DT_FLOAT |
| rowIndex | DT_INT64 |
| output | DT_FLOAT |

Constraints: x and weight must be BOTH FP8 or BOTH FP4.

## PerToken Quant Mode - Supported Data Types

| Parameter | Supported Types |
|-----------|----------------|
| x (activation) | DT_INT8, DT_FLOAT8_E4M3FN, DT_HIFLOAT8 |
| weight | DT_INT8, DT_FLOAT8_E4M3FN, DT_HIFLOAT8 |
| scale | DT_FLOAT, DT_BF16 |
| pertoken_scale | DT_FLOAT (optional) |
| bias | DT_BF16 |
| groupList | DT_INT64 |
| sharedInput | DT_BF16 |
| logit | DT_FLOAT |
| rowIndex | DT_INT64 or DT_INT32 (INT32 only when x is INT8) |
| output | DT_FLOAT |

Constraints: x and weight must match dtype, except both can be different FP8 variants.

## Weight Quant Mode (separate tiling path) - MX-A8W4

| Parameter | Supported Types |
|-----------|----------------|
| x | DT_FLOAT8_E4M3FN |
| weight | DT_FLOAT4_E2M1 |
| scale | DT_FLOAT8_E8M0 |
| pertoken_scale | DT_FLOAT8_E8M0 |
| weight format | FRACTAL_NZ or FRACTAL_NZ_C0_32 |

K/N alignment: must be divisible by 32. Minimum K/N: 32.

## Format Restrictions (Critical A5 Difference)

- **MX mode**: weight format must be **ND** (non-private format)
- **PerToken mode**: weight format must be **FRACTAL_NZ**
- All other tensors: ND format only (no private formats)
- x and pertoken_scale: must be ND format

## FP4 Specific Shape Constraints

- k must be even (divisible by 2)
- k cannot be 2
- n must be even when weight is not transposed

## Key Differences from A3

1. A5 (arch35) has its own separate tiling files under `op_tiling/arch35/`
2. Weight quant tiling only supports MX-A8W4-WEIGHT-NZ scenario on A5
3. The weight quant path forces transpose_w=true and transpose_x=false
4. A5 introduces DT_HIFLOAT8 support in PerToken mode (not seen on A3)
5. PerToken mode requires FRACTAL_NZ weight format (vs ND on A3 MX mode)
