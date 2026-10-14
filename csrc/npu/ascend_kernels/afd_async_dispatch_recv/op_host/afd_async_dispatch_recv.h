// SPDX-License-Identifier: MIT
// Copyright contributors to the AFD plugin project
#ifndef AFD_ASYNC_DISPATCH_RECV_HOST_H
#define AFD_ASYNC_DISPATCH_RECV_HOST_H
#include "register/op_def_registry.h"
#include "ops_log.h"
namespace ops {
class AfdAsyncDispatchRecv : public OpDef {
public:
    explicit AfdAsyncDispatchRecv(const char *name);
};
}
namespace optiling {
ge::graphStatus GraphDispatchRecvTiling(gert::TilingContext *context);
}
namespace ge {
graphStatus GraphDispatchRecvShape(gert::InferShapeContext *context);
graphStatus GraphDispatchRecvDtype(gert::InferDataTypeContext *context);
}
#endif
