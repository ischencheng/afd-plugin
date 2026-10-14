// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.

#include <string.h>
#include "graph/types.h"
#include "aclnn_afd_async_dispatch_recv_controlled.h"
#include "aclnnInner_afd_async_dispatch_recv_controlled.h"

namespace {
static constexpr int32_t NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0;
static constexpr int32_t NNOPBASE_HCCL_SERVER_TYPE_MTE = 1;
static constexpr int32_t NNOPBASE_HCCL_SERVER_TYPE_END = 2;
} // namespace
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, int32_t sType);

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnAfdAsyncDispatchRecvControlledGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *commArgs,
    const aclTensor *stop,
    const aclTensor *validLayers,
    int64_t magic,
    int64_t maxSeqLen,
    int64_t hiddenSize,
    int64_t topk,
    int64_t moeRankNum,
    int64_t attnRankNum,
    int64_t routeExpertNumPerMoe,
    int64_t moeRankId,
    int64_t worldSize,
    int64_t tpSize,
    int64_t dynamicQuant,
    char *hcclGroupName,
    const aclTensor *expandXOut,
    const aclTensor *dynamicScalesOut,
    const aclTensor *batchInfoOut,
    const aclTensor *epRecvCountRoutedOut,
    const aclTensor *slot,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    return aclnnInnerAfdAsyncDispatchRecvControlledGetWorkspaceSize(x, commArgs, stop, validLayers,
        magic, maxSeqLen, hiddenSize, topk, moeRankNum, attnRankNum, routeExpertNumPerMoe, moeRankId, worldSize,
        tpSize, dynamicQuant, hcclGroupName,
        expandXOut, dynamicScalesOut, batchInfoOut, epRecvCountRoutedOut, slot,
        workspaceSize, executor);
}

aclnnStatus aclnnAfdAsyncDispatchRecvControlled(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    return aclnnInnerAfdAsyncDispatchRecvControlled(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
