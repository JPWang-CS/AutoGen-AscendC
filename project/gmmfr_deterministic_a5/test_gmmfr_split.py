"""
GMMFR 拆分测试脚本：将 npu_grouped_matmul_finalize_routing 拆分为
  Step 1: torch_npu.npu_grouped_matmul   (分组矩阵乘 + per-token 反量化)
  Step 2: torch_npu.npu_moe_finalize_routing (路由聚合)
  Step 3: 手动添加 shared_input 贡献

API 参考:
  - npu_grouped_matmul (per-token 量化, group_list=Tensor, 单单单模式):
    https://www.hiascend.com/document/detail/zh/Pytorch/700/apiref/apilist/ptaoplist_000160.html
  - npu_moe_finalize_routing:
    https://www.hiascend.com/document/detail/zh/Pytorch/600/apiref/apilist/ptaoplist_000159.html
"""

import torch
import torch_npu
import numpy as np
from scipy.special import softmax
import argparse

torch.npu.set_device(1)


def prepare_inputs(groupNum, topK, m, k, n):
    """准备 GMMFR 所需的全部输入数据"""
    batch = m // groupNum

    # 基础输入
    x = torch.from_numpy(np.random.randint(-10, 10, [m, k]).astype(np.int8)).npu()
    weight = torch.from_numpy(np.random.randint(-10, 10, [groupNum, k, n]).astype(np.int8)).npu()
    bias = torch.zeros((groupNum, n), dtype=torch.bfloat16, device="npu")
    weightNz = torch_npu.npu_format_cast(weight, 29)  # NZ 格式

    # 量化参数
    scale = torch.from_numpy(np.random.normal(0, 0.01, (groupNum, n)).astype(np.float32)).npu()
    perTokenScale = torch.from_numpy(np.random.normal(0, 0.01, (m, 1)).astype(np.float32)).npu()
    perTokenScale = perTokenScale.reshape(m).npu()

    # 分组列表: 每个 group 的行数
    groupList = torch.from_numpy(np.array([batch] * groupNum, dtype=np.int64)).npu()

    # 路由相关
    logits_ori = np.random.normal(0, 0.1, [batch, groupNum]).astype(np.float32)
    routing = np.argsort(logits_ori, axis=1, kind="stable")[:, -topK:].astype(np.int32)
    logits_np = softmax(
        logits_ori[np.arange(batch).reshape(-1, 1).repeat(topK, axis=1), routing], axis=1
    ).astype(np.float32)
    logits = torch.from_numpy(logits_np.reshape(m)).npu()

    # 共享输入
    shared_input = np.random.normal(0, 0.1, (batch // 4, n)).astype(np.float32)
    shared_input = torch.from_numpy(shared_input).to(torch.bfloat16).npu()

    # 行索引: 每个 expanded row 的目标输出行
    row_index = torch.from_numpy(
        (np.argsort(routing.reshape(-1), kind="stable") // topK).astype(np.int64)
    ).npu()

    share_input_offset = batch // 2

    return {
        "x": x, "weight": weight, "weightNz": weightNz, "bias": bias,
        "scale": scale, "perTokenScale": perTokenScale,
        "groupList": groupList, "logits": logits,
        "shared_input": shared_input, "row_index": row_index,
        "share_input_offset": share_input_offset,
        "groupNum": groupNum, "topK": topK, "batch": batch,
        "m": m, "k": k, "n": n,
    }


def gmmfr_fused(inputs):
    """原始融合调用: npu_grouped_matmul_finalize_routing"""
    torch_npu.npu_grouped_matmul_finalize_routing(
        inputs["x"], inputs["weightNz"], inputs["groupList"],
        scale=inputs["scale"],
        bias=inputs["bias"],
        pertoken_scale=inputs["perTokenScale"],
        shared_input=inputs["shared_input"],
        logit=inputs["logits"],
        row_index=inputs["row_index"],
        shared_input_weight=1.0,
        shared_input_offset=inputs["share_input_offset"],
        output_bs=inputs["batch"],
    )
    torch_npu.npu.synchronize()


def gmmfr_split(inputs):
    """
    拆分调用:
      Step 1: npu_grouped_matmul     (分组矩阵乘 + per-token 反量化 → FP16)
      Step 2: npu_moe_finalize_routing (路由聚合, 不含 shared_input)
      Step 3: 手动添加 shared_input 贡献

    参数映射:
      GMMFR fused                           → Split
      ─────────────────────────────────────────────────────────────────
      x, weight, scale, pertoken_scale      → npu_grouped_matmul(x, weight, scale, per_token_scale)
      logit (路由权重)                      → npu_moe_finalize_routing(scales)
      row_index (目标行)                    → npu_moe_finalize_routing(expanded_src_to_dst_row)
      shared_input + offset + weight        → 手动 FR 输出切片 += shared_input * weight
    """
    batch = inputs["batch"]
    topK = inputs["topK"]
    m = inputs["m"]
    n = inputs["n"]

    # ===================== Step 1: Grouped MatMul =====================
    # per-token 量化: INT8 × INT8 → INT32, 再 INT32 × scale × per_token_scale → FP16
    # 文档约束: x=INT8[M,K], weight=INT8[G,K,N](3D), scale=FP32, per_token_scale=FP32
    #           单单单模式: split_item=3, group_type=0, group_list_type=1
    gmm_out = torch_npu.npu_grouped_matmul(
        inputs["x"],                          # [m, k] INT8
        inputs["weightNz"],                   # [groupNum, k, n] INT8 NZ
        scale=inputs["scale"],                # [groupNum, n] FP32
        per_token_scale=inputs["perTokenScale"],  # [m] FP32
        group_list=inputs["groupList"],        # [groupNum] INT64 (每组大小)
        split_item=3,
        group_type=0,
        group_list_type=1,
        output_dtype=torch.float16,
    )
    gmm_result = gmm_out[0]                   # [m, n] FP16
    torch_npu.npu.synchronize()

    # ===================== Step 2: Finalize Routing =====================
    # npu_moe_finalize_routing 参数说明 (PyTorch 6.0 文档):
    #   expanded_permuted_rows: [NUM_ROWS*K, H] FP16/BF16/FP32  (GMM 反量化结果)
    #   skip1:                  [NUM_ROWS, H] 与输出同形状       (此处为 None, shared_input 单独处理)
    #   skip2:                  [NUM_ROWS, H] 或 None            (skip1=None 时 skip2 必须 None)
    #   bias:                   [E, H] 每个专家的偏置             (此处 None)
    #   scales:                 [NUM_ROWS, K] 路由权重           (logits reshape)
    #   expanded_src_to_dst_row:[NUM_ROWS*K] INT32 目标行索引    (row_index 转换)
    #   expert_for_source_row:  [NUM_ROWS, K] INT32 专家索引     (此处 None)
    #   drop_pad_mode:          0
    #
    # 注意: skip2=None 时 skip1 必须也为 None, 因此 shared_input 在 Step 3 手动处理

    expanded_permuted_rows = gmm_result.to(torch.float32)   # [m, n] FP32
    scales = inputs["logits"].reshape(batch, topK)           # [batch, topK] FP32
    expanded_src_to_dst_row = inputs["row_index"].to(torch.int32)  # [m] INT32

    fr_out = torch_npu.npu_moe_finalize_routing(
        expanded_permuted_rows,     # expanded_permuted_rows: [m, n] FP32
        None,                       # skip1: None (shared_input 单独处理)
        None,                       # skip2: None
        None,                       # bias: None
        scales,                     # scales: [batch, topK] FP32
        expanded_src_to_dst_row,    # expanded_src_to_dst_row: [m] INT32
        None,                       # expert_for_source_row: None
        0,                          # drop_pad_mode: 0
    )
    torch_npu.npu.synchronize()

    # ===================== Step 3: 手动添加 shared_input =====================
    # GMMFR: out[offset : offset+len] += shared_input * shared_input_weight
    shared_input_weight = 1.0
    offset = inputs["share_input_offset"]
    shared_input_fp32 = inputs["shared_input"].to(torch.float32)
    fr_out[offset:offset + shared_input_fp32.shape[0]] += shared_input_fp32 * shared_input_weight

    return fr_out


def warmup():
    """L2 cache 预热"""
    mm1 = torch.rand((10240, 10240), dtype=torch.float16).npu()
    mm2 = torch.rand((10240, 10240), dtype=torch.float16).npu()
    reduce_input = torch.rand((96, 1024, 1024), dtype=torch.float16).npu()
    _ = torch.matmul(mm1, mm2)
    torch_npu.npu.synchronize()
    _ = torch.max(reduce_input)
    torch_npu.npu.synchronize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GMMFR 拆分测试 (npu_grouped_matmul + npu_moe_finalize_routing)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--groupNum", type=int, default=8)
    parser.add_argument("--topK", type=int, default=8)
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--n", type=int, default=7168)
    parser.add_argument("--times", type=int, default=12)
    parser.add_argument("--mode", type=str, default="split",
                        choices=["fused", "split", "both"],
                        help="fused=仅融合算子, split=仅拆分, both=两者对比")
    args = parser.parse_args()

    warmup()

    for i in range(args.times):
        inputs = prepare_inputs(args.groupNum, args.topK, args.m, args.k, args.n)

        # L2 cache activate
        inputs["x"] = inputs["x"] + torch.zeros_like(inputs["x"])

        if args.mode == "fused":
            gmmfr_fused(inputs)
            print(f"[{i}] fused done")
        elif args.mode == "split":
            result = gmmfr_split(inputs)
            print(f"[{i}] split done, output shape: {result.shape}")
        else:  # both
            result_split = gmmfr_split(inputs)
            gmmfr_fused(inputs)
            print(f"[{i}] both done, split output shape: {result_split.shape}")

    print("All iterations completed.")
