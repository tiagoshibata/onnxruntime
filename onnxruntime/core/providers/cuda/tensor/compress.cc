// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "compress.h"
#include "core/providers/cpu/tensor/utils.h"
#include "compress_impl.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    Compress,
    kOnnxDomain,
    9,
    kCudaExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
                      .TypeConstraint("T1", DataTypeImpl::GetTensorType<bool>()),
    Compress);

Status Compress::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* input_tensor = ctx->Input<Tensor>(0);
  ORT_ENFORCE(input_tensor);
  size_t rank = input_tensor->Shape().NumDimensions();
  auto& input_dimensions = input_tensor->Shape().GetDims();
  if (has_axis_) {
    ORT_ENFORCE(axis_ < static_cast<int64_t>(rank), "axis greater than input data dimension!");
  }

  const Tensor* condition = ctx->Input<Tensor>(1);
  ORT_ENFORCE(condition);
  auto condition_length = condition->Shape().Size();
  auto condition_data = condition->template Data<bool>();

  // if has axis, we need to compress on dimension[axis], otherwise compress on the flattened input data
  int64_t input_size = input_tensor->Shape().Size();
  int64_t compress_input_length = has_axis_ ? input_dimensions[axis_] : input_size;
  int64_t valid_condition_length = compress_input_length < condition_length ? compress_input_length : condition_length;

  auto condition_cumulative_sum_buffer = GetScratchBuffer<int32_t>(valid_condition_length);
  auto condition_cumulative_sum = condition_cumulative_sum_buffer.get();
  PrefixSumImpl(reinterpret_cast<const int8_t*>(condition_data), condition_cumulative_sum, valid_condition_length);
  
  int32_t positive_condition_count = 0;
  CUDA_RETURN_IF_ERROR(cudaMemcpy(&positive_condition_count, condition_cumulative_sum + valid_condition_length - 1, sizeof(int32_t), cudaMemcpyDeviceToHost));

  std::vector<int64_t> output_dims(input_dimensions);
  if (has_axis_) {
    output_dims[axis_] = positive_condition_count;
  } else {
    output_dims.resize(1);
    output_dims[0] = positive_condition_count;
  }

  TensorShape output_shape(output_dims);
  auto output_tensor = ctx->Output(0, output_shape);
  if (positive_condition_count <= 0) {
    return Status::OK();
  }

  auto element_bytes = input_tensor->DataType()->Size();

  int64_t axis_right_stride = 1;
  if (has_axis_) {
    for (auto i = static_cast<size_t>(axis_ + 1); i < rank; ++i) {
      axis_right_stride *= input_dimensions[i];
    }
  }

  ORT_RETURN_IF_ERROR(CompressImpl(element_bytes,
                                           gsl::narrow_cast<int32_t>(valid_condition_length),
                                           gsl::narrow_cast<int32_t>(axis_right_stride),
                                           has_axis_ ? gsl::narrow_cast<int32_t>(input_dimensions[axis_]) : gsl::narrow_cast<int32_t>(input_size),
                                           gsl::narrow_cast<int32_t>(positive_condition_count),
                                           condition_cumulative_sum,
                                           condition_data,
                                           input_tensor->DataRaw(),
                                           output_tensor->MutableDataRaw(),
                                           input_size));

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
