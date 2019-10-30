// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {
template <typename T>
class Attention : public OpKernel {
 public:
  explicit Attention(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;

 private:
  int num_heads_;  // number of attention heads
};
}  // namespace contrib
}  // namespace onnxruntime
