// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "matmul_bnb4.cuh"
#include "dequantize_blockwise_bnb4.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {
using namespace onnxruntime::cuda;

template <typename T>
class MatMulBnb4 final : public CudaKernel {
 public:
  MatMulBnb4(const OpKernelInfo& info) : CudaKernel(info) {
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("K", &K_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("N", &N_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("blocksize", &blocksize_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("quant_type", &quant_type_));
    ORT_ENFORCE(quant_type_ == FP4 || quant_type_ == NF4, "Invalid quant_type, only 0 (FP4) and 1 (NF4) are supported.");
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t K_;
  int64_t N_;
  int64_t blocksize_;
  int64_t quant_type_;
};

template <typename T>
Status MatMulBnb4<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* a = ctx->Input<Tensor>(0);
  const Tensor* b_quant = ctx->Input<Tensor>(1);
  const Tensor* absmax = ctx->Input<Tensor>(2);
  const Tensor* quant_map = ctx->Input<Tensor>(3);

  const auto* a_data = a->Data<T>();
  const uint8_t* b_quant_data = b_quant->Data<uint8_t>();
  const auto* quant_map_data = quant_map->Data<T>();
  const float_t* absmax_data = absmax->Data<float_t>();

  typedef typename ToCudaType<T>::MappedType CudaT;

  constexpr bool transa = false;
  constexpr bool transb = true;
  MatMulComputeHelper helper;
  TensorShape b_shape({N_, K_});
  ORT_RETURN_IF_ERROR(
      helper.Compute(a->Shape(), b_shape, transa, transb));

  Tensor* Y = ctx->Output(0, helper.OutputShape());
  // Bail out early if the output is going to be empty
  if (Y->Shape().Size() == 0) return Status::OK();

  bool is_4bit_done = TryMatMulBnb4(
      reinterpret_cast<CudaT*>(Y->MutableData<T>()),
      reinterpret_cast<const CudaT*>(a_data),
      b_quant_data,
      absmax_data,
      reinterpret_cast<const CudaT*>(quant_map_data),
      SafeInt<int>(helper.M()),
      SafeInt<int>(helper.N()),
      SafeInt<int>(helper.K()),
      SafeInt<int>(blocksize_),
      static_cast<cudaStream_t>(ctx->GetComputeStream()->GetHandle()));

  if (!is_4bit_done) {
    IAllocatorUniquePtr<T> b_dequant_ptr = GetScratchBuffer<T>(N_ * K_, ctx->GetComputeStream());
    auto* b_dequant_data = b_dequant_ptr.get();
    ORT_RETURN_IF_ERROR(DequantizeBnb4<CudaT>(
        SafeInt<int>(quant_type_),
        reinterpret_cast<CudaT*>(b_dequant_data),
        b_quant_data,
        absmax_data,
        SafeInt<int>(blocksize_),
        SafeInt<int>(N_ * K_),
        static_cast<cudaStream_t>(ctx->GetComputeStream()->GetHandle())));

    const CudaT alpha = ToCudaType<T>::FromFloat(1.f);
    const CudaT zero = ToCudaType<T>::FromFloat(0.f);

    CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
        GetCublasHandle(ctx),
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        SafeInt<int>(helper.N()),
        SafeInt<int>(helper.M()),
        SafeInt<int>(helper.K()),
        &alpha,
        reinterpret_cast<const CudaT*>(b_dequant_data),
        SafeInt<int>(K_),
        reinterpret_cast<const CudaT*>(a_data),
        helper.Lda(transa),
        &zero,
        reinterpret_cast<CudaT*>(Y->MutableData<T>()),
        helper.Ldc(),
        GetDeviceProp()));
  }

  return Status::OK();
}

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MatMulBnb4,
    kMSDomain,
    1,
    float,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>()),
    MatMulBnb4<float>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MatMulBnb4,
    kMSDomain,
    1,
    MLFloat16,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>()),
    MatMulBnb4<MLFloat16>);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime