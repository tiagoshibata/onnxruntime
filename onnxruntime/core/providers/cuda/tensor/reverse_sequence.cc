// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "reverse_sequence.h"
#include "reverse_sequence_impl.h"

#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/common.h"
#include "core/framework/utils.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    ReverseSequence,
    kOnnxDomain,
    10,
    kCudaExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    ReverseSequenceOp);

#define ReverseSequenceCallCudaImplIfType(T)                                                         \
  if (data_type == DataTypeImpl::GetType<T>()) {                                                     \
    CUDA_RETURN_IF_ERROR(ReverseSequenceCudaImpl(                                                    \
        reinterpret_cast<const typename ToCudaType<T>::MappedType*>(X.template Data<T>()),           \
        seq_lengths.Data<int64_t>(),                                                                 \
        reinterpret_cast<typename ToCudaType<T>::MappedType*>(Y.template MutableData<T>()),          \
        gsl::narrow<int>(batch_size), gsl::narrow<int>(max_seq_len), gsl::narrow<int>(element_size), \
        time_major_));                                                                               \
    return Status::OK();                                                                             \
  }

Status ReverseSequenceOp::ComputeInternal(OpKernelContext* context) const {
  const auto& X = *context->Input<Tensor>(0);
  const auto data_type = X.DataType();
  const auto& dims = X.Shape();

  const auto batch_size = time_major_ ? dims[1] : dims[0];
  const auto max_seq_len = time_major_ ? dims[0] : dims[1];
  const auto element_size = dims.SizeFromDimension(2);

  const auto& seq_lengths = *context->Input<Tensor>(1);
  const auto& seq_len_shape = seq_lengths.Shape();

  if (seq_len_shape.NumDimensions() != 1 || seq_len_shape[0] != batch_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "sequence_lens shape must be {batch_size}. Got:",
                           seq_len_shape, ". batch_size=", batch_size);
  }
  auto& Y = *context->Output(0, dims);

  ReverseSequenceCallCudaImplIfType(float);
  ReverseSequenceCallCudaImplIfType(MLFloat16);
  ReverseSequenceCallCudaImplIfType(int32_t);
  ReverseSequenceCallCudaImplIfType(uint32_t);
  ReverseSequenceCallCudaImplIfType(int16_t);
  ReverseSequenceCallCudaImplIfType(uint16_t);
  ReverseSequenceCallCudaImplIfType(int8_t);
  ReverseSequenceCallCudaImplIfType(uint8_t);
  ReverseSequenceCallCudaImplIfType(double);
  ReverseSequenceCallCudaImplIfType(bool);
  ReverseSequenceCallCudaImplIfType(int64_t);
  ReverseSequenceCallCudaImplIfType(uint64_t);

  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                         "Type for ", data_type, " is not supported yet in ReverseSequence.");
}

}  // namespace cuda
}  // namespace onnxruntime
