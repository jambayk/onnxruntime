// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "blockwise_quant_block_bnb4.h"

#include <vector>

#include "core/common/safeint.h"
#include "core/framework/float16.h"
#include "core/platform/threadpool.h"
#include <iostream>

namespace onnxruntime {
namespace contrib {

template <typename T, int32_t block_size>
void QuantizeBlockwiseBnb4(
    uint8_t* dst,          // shape: [(N * K + 1) / 2] 
    const T* src,          // shape: [N, K]
    float* scale,              // shape: [(N * K + block_size - 1) / block_size]
    int32_t N,
    int32_t K,
    onnxruntime::concurrency::ThreadPool* thread_pool) {
  int32_t numel = N * K;
  int32_t total_block_count = (numel + block_size - 1) / block_size;

  concurrency::ThreadPool::TryBatchParallelFor(
      thread_pool,
      total_block_count,
      [&](ptrdiff_t block_idx) {
        QuantizeBlockBnb4<T, block_size>(src, dst, scale[block_idx], block_idx, numel);
      },
      0);
}

template <typename T>
void QuantizeBlockwiseBnb4(
    uint8_t* dst,          // shape: [(N * K + 1) / 2]
    const T* src,          // shape: [N, K]
    float* scale,              // shape: [(N * K + block_size - 1) / block_size]
    int32_t block_size,
    int32_t N,
    int32_t K,
    onnxruntime::concurrency::ThreadPool* thread_pool) {

  if (16 == block_size) {
    QuantizeBlockwiseBnb4<T, 16>(dst, src, scale, N, K, thread_pool);
  } else if (32 == block_size) {
    QuantizeBlockwiseBnb4<T, 32>(dst, src, scale, N, K, thread_pool);
  } else if (64 == block_size) {
    QuantizeBlockwiseBnb4<T, 64>(dst, src, scale, N, K, thread_pool);
  } else if (128 == block_size) {
    QuantizeBlockwiseBnb4<T, 128>(dst, src, scale, N, K, thread_pool);
  } else if (256 == block_size) {
    QuantizeBlockwiseBnb4<T, 256>(dst, src, scale, N, K, thread_pool);
  } else {
    ORT_NOT_IMPLEMENTED("only block size 16, 32, 64, 128, 256 are supported.");
  }
}

template <typename T, int32_t block_size>
void DequantizeBlockwiseBnb4(
    T* dst,                      // shape: [N, K]
    const uint8_t* src,          // shape: [(N * K + 1) / 2)]
    const float* scale,              // shape: [(N * K + block_size - 1) / block_size]
    int32_t N,
    int32_t K,
    onnxruntime::concurrency::ThreadPool* thread_pool) {
  int32_t numel = N * K;
  int32_t total_block_count = (numel + block_size - 1) / block_size;

  concurrency::ThreadPool::TryBatchParallelFor(
      thread_pool,
      total_block_count,
      [&](ptrdiff_t block_idx) {
        DeQuantizeBlockBnb4<T, block_size>(src, dst, scale[block_idx], block_idx, numel);
      },
      0);
}

template <typename T>
void DequantizeBlockwiseBnb4(
    T* dst,                      // shape: [N, K]
    const uint8_t* src,          // shape: [(N * K + 1) / 2)]
    const float* scale,              // shape: [(N * K + block_size - 1) / block_size]
    int32_t block_size,
    int32_t N,
    int32_t K,
    onnxruntime::concurrency::ThreadPool* thread_pool) {

  if (16 == block_size) {
    DequantizeBlockwiseBnb4<T, 16>(dst, src, scale, N, K, thread_pool);
  } else if (32 == block_size) {
    DequantizeBlockwiseBnb4<T, 32>(dst, src, scale, N, K, thread_pool);
  } else if (64 == block_size) {
    DequantizeBlockwiseBnb4<T, 64>(dst, src, scale, N, K, thread_pool);
  } else if (128 == block_size) {
    DequantizeBlockwiseBnb4<T, 128>(dst, src, scale, N, K, thread_pool);
  } else if (256 == block_size) {
    DequantizeBlockwiseBnb4<T, 256>(dst, src, scale, N, K, thread_pool);
  } else {
    ORT_NOT_IMPLEMENTED("only block size 16, 32, 64, 128, 256 are supported.");
  }
}



}  // namespace contrib
}  // namespace onnxruntime