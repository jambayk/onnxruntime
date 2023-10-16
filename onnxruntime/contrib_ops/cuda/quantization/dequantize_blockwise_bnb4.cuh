// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

typedef enum Bnb_DataType_t
{
  FP4 = 0,
  NF4 = 1,
} Bnb_DataType_t;

template <class T>
Status DequantizeBnb4(
    int quant_type,
    T* output,
    const unsigned char* quant_data, 
    const float* absmax,
    int block_size,
    int numel,
    cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime