// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the AFD plugin project
#ifndef AFD_CAM_FFN_STATUS_H
#define AFD_CAM_FFN_STATUS_H
#include <cstdint>

// Graph-local decisions; the compact CAM wire header is unchanged.
// Nonnegative slots select weights. Negative slots skip FFN and CS.
namespace AfdCam {
constexpr int64_t RUN = 0;
constexpr int64_t REQUEST_STOP = 1;
constexpr int64_t STOP = -1;
constexpr int64_t INVALID_LAYER = -2;
constexpr int64_t CAPACITY_EXCEEDED = -3;
constexpr int64_t INSUFFICIENT_UB = -4;
// One stable control tensor with disjoint host STOP and device ERROR writers.
constexpr uint32_t CONTROL_WORDS = 2;
constexpr uint32_t ERROR_WORD = 1;
constexpr uint32_t TOKEN_OFFSET_LIMIT = 65536;
} // namespace AfdCam
#endif
