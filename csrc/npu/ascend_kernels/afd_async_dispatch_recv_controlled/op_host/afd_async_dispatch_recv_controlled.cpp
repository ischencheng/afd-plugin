// SPDX-License-Identifier: MIT
// Copyright contributors to the AFD plugin project
#include "afd_async_dispatch_recv.h"
#include "../op_kernel/afd_async_dispatch_recv_tiling.h"
#include "../op_kernel/cam_ffn_status.h"
#include "ops_error.h"

namespace optiling {
static ge::graphStatus ControlledTiling(gert::TilingContext *context)
{
    if (GraphDispatchRecvTiling(context) != ge::GRAPH_SUCCESS) { return ge::GRAPH_FAILED; }
    const auto *stop = context->GetInputShape(2);
    const auto *layers = context->GetInputShape(3);
    OPS_ERR_IF(stop == nullptr || layers == nullptr ||
        stop->GetStorageShape().GetDimNum() != 1 ||
        stop->GetStorageShape().GetDim(0) != AfdCam::CONTROL_WORDS ||
        layers->GetStorageShape().GetDimNum() != 1 || layers->GetStorageShape().GetDim(0) <= 0,
        OPS_LOG_E(context->GetNodeName(), "stop control[2] and a nonempty layer table are required"),
        return ge::GRAPH_FAILED);
    auto &info = context->GetTilingData<Cam::AfdAsyncDispatchRecvTilingData>()->moeDistributeDispatchInfo;
    OPS_ERR_IF(info.maxSeqLen / info.tpSize > AfdCam::TOKEN_OFFSET_LIMIT,
        OPS_LOG_E(context->GetNodeName(), "per-rank token capacity exceeds uint16 CAM offsets"),
        return ge::GRAPH_FAILED);
    info.numLayers = layers->GetStorageShape().GetDim(0);
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus ControlledShape(gert::InferShapeContext *context)
{
    if (GraphDispatchRecvShape(context) != GRAPH_SUCCESS) { return GRAPH_FAILED; }
    context->GetOutputShape(4)->SetDimNum(1);
    context->GetOutputShape(4)->SetDim(0, 1);
    return GRAPH_SUCCESS;
}
static graphStatus ControlledDtype(gert::InferDataTypeContext *context)
{
    if (GraphDispatchRecvDtype(context) != GRAPH_SUCCESS) { return GRAPH_FAILED; }
    context->SetOutputDataType(4, DT_INT64);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class AfdAsyncDispatchRecvControlled : public AfdAsyncDispatchRecv {
public:
    explicit AfdAsyncDispatchRecvControlled(const char *name) : AfdAsyncDispatchRecv(name)
    {
        for (const char *input : {"stop", "validLayers"}) {
            this->Input(input).ParamType(REQUIRED)
                .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        }
        this->Output("slot").ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::ControlledShape).SetInferDataType(ge::ControlledDtype);
        this->AICore().SetTiling(optiling::ControlledTiling);
    }
};
OP_ADD(AfdAsyncDispatchRecvControlled);
}
