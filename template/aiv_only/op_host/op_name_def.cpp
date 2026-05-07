/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it
 * under the terms and conditions of CANN Open Software License Agreement
 * Version 2.0 (the "License").
 * Please refer to the License for details.
 */

/*!
 * \file op_name_def.cpp
 * \brief 纯 AIV (Vector Only) 算子定义模板
 */

#include "register/op_def_registry.h"

namespace ops {
class OpNameTemplate : public OpDef {
public:
    explicit OpNameTemplate(const char* name) : OpDef(name)
    {
        // === 输入定义 ===
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        // === 输出定义 ===
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        // === 属性定义 ===
        // this->Attr("attr_name").AttrType(OPTIONAL).Int(0);

        // === A2/A3 平台配置 (arch32) ===
        OpAICoreConfig aicConfig;
        aicConfig.DynamicCompileStaticFlag(true)
                .DynamicFormatFlag(true)
                .DynamicRankSupportFlag(true)
                .DynamicShapeSupportFlag(true);
        this->AICore().AddConfig("ascend910b", aicConfig);
        this->AICore().AddConfig("ascend910_93", aicConfig);

        // === A5 平台配置 (arch35) ===
        OpAICoreConfig config950;
        config950.Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        config950.Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        config950.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("opFile.value", "op_name_apt");
        this->AICore().AddConfig("ascend950", config950);
    }
};

OP_ADD(OpNameTemplate);
}
