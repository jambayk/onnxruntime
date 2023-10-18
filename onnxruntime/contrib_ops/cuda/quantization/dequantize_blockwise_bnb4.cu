// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include "core/providers/cuda/cuda_common.h"
#include "dequantize_blockwise_bnb4.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {


__device__ float dDequantizeFP4Tree(unsigned char val, float scale)
{
  float sign = (val & 0b1000) == 8 ? -1.0f : 1.0f;
  if((val & 0b0100) == 4) // 0
    if((val & 0b0010) == 2) //01
      if((val & 0b0001) == 1) // 111
        return 0.25000000f*scale*sign; // 1111
      else
        return 0.16666667f*scale*sign; // 1110
    else
      if((val & 0b0001) == 1) // 110
        return 0.50000000f*scale*sign; // 1101
      else
        return 0.33333333f*scale*sign; // 1100
  else
    if((val & 0b0010) == 2) //10
      if((val & 0b0001) == 1) // 101
        return 1.00000000f*scale*sign; // 1011
      else
        return 0.66666667f*scale*sign; // 1010
    else
      if((val & 0b0001) == 1) // 100
        return 5.208333333e-03f*scale*sign; // 1001
      else
        return 0.00000000f*scale*sign; // 1000
}

__device__ float dDequantizeNF4(unsigned char val)
{
  if((val & 0b1000) == 8)
    if((val & 0b0100) == 4) // 1
      if((val & 0b0010) == 2) // 11
        if((val & 0b0001) == 1) // 111
          return 1.0f;
        else
          return 0.7229568362236023f;
      else
        if((val & 0b0001) == 1) // 110
          return 0.5626170039176941f;
        else
          return 0.44070982933044434f;
    else
      if((val & 0b0010) == 2) //10
        if((val & 0b0001) == 1) // 101
          return 0.33791524171829224f;
        else
          return 0.24611230194568634f;
      else
        if((val & 0b0001) == 1) // 100
          return 0.16093020141124725f;
        else
          return 0.07958029955625534f;

  else
    if((val & 0b0100) == 4) // 0
      if((val & 0b0010) == 2) //01
        if((val & 0b0001) == 1) // 011
          return 0.0f;
        else
          return -0.09105003625154495f;
      else
        if((val & 0b0001) == 1) // 010
          return -0.18477343022823334f;
        else
          return -0.28444138169288635f;
    else
      if((val & 0b0010) == 2) //00
        if((val & 0b0001) == 1) // 001
          return -0.39491748809814453f;
        else
          return -0.5250730514526367f;
      else
        if((val & 0b0001) == 1) // 000
          return -0.6961928009986877f;
        else
          return -1.0f;
}


template<typename T, int TILE_SIZE, int THREADS, int NUM_PER_TH, int DATA_TYPE>
__global__ void kDequantizeBlockwise(T *output, const unsigned char *quant_data, const float *scale, const int blocksize, const int n)
{
  const int n_load = (gridDim.x * TILE_SIZE);
  int valid_items_load = 0;
  int valid_items_store = 0;
  const int base_idx = (blockIdx.x * TILE_SIZE);

  T vals[NUM_PER_TH*2];
  unsigned char qvals[NUM_PER_TH];
  float local_abs_max = -FLT_MAX;

  typedef cub::BlockLoad<unsigned char, THREADS, NUM_PER_TH, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadChar;
  typedef cub::BlockStore<T, THREADS, NUM_PER_TH*2, cub::BLOCK_STORE_WARP_TRANSPOSE> StoreT;

  __shared__ typename LoadChar::TempStorage loadchar;
  __shared__ typename StoreT::TempStorage storet;

  for (unsigned int i = base_idx; i < n_load; i += gridDim.x*TILE_SIZE)
  {
    valid_items_load = (n+1)/2 - i > TILE_SIZE ? TILE_SIZE : (n+1)/2 - i;
    valid_items_store = n - i*2 > TILE_SIZE*2 ? TILE_SIZE*2 : n - i*2;

    local_abs_max = __ldg(&scale[(i+threadIdx.x*NUM_PER_TH)/(blocksize)]);

    __syncthreads();
    LoadChar(loadchar).Load(&(quant_data[i]), qvals, valid_items_load, 128);

    switch(DATA_TYPE)
    {
      case FP4:
        #pragma unroll NUM_PER_TH
        for(int j = 0; j < NUM_PER_TH; j++)
        {
          vals[j*2] = dDequantizeFP4Tree(qvals[j] >> 4, local_abs_max);
          vals[j*2 + 1] = dDequantizeFP4Tree(qvals[j] & 0x0F, local_abs_max);
        }
        break;
      case NF4:
        #pragma unroll NUM_PER_TH
        for(int j = 0; j < NUM_PER_TH; j++)
        {
          vals[j*2] = dDequantizeNF4(qvals[j] >> 4)* local_abs_max;
          vals[j*2 + 1] = dDequantizeNF4(qvals[j] & 0x0F)* local_abs_max;
        }
        break;
    }

    __syncthreads();
    StoreT(storet).Store(&(output[i*2]), vals, valid_items_store);
  }
}


template<class T>
Status DequantizeBnb4(int quant_type, T *output, const unsigned char *quant_data, const float *scale, int blocksize, int numel, cudaStream_t stream)
{
  ORT_ENFORCE(quant_type == FP4 || quant_type == NF4, "Unsupported quantization type");

  int tile_size = 1024;

  switch (quant_type) {
    case FP4:
      kDequantizeBlockwise<T, 512, 64, 8, FP4><<<(numel+tile_size-1)/tile_size, 64, 0, stream>>>(output, quant_data, scale, blocksize/2, numel);
      break;
    case NF4:
      kDequantizeBlockwise<T, 512, 64, 8, NF4><<<(numel+tile_size-1)/tile_size, 64, 0, stream>>>(output, quant_data, scale, blocksize/2, numel);
      break;
  }
    
  return Status::OK();
}

template Status DequantizeBnb4<float>(int quant_type, float *output, const unsigned char *quant_data, const float *scale,  int blocksize, int numel, cudaStream_t stream);

template Status DequantizeBnb4<half>(int quant_type, half *output, const unsigned char *quant_data, const float *scale,  int blocksize, int numel, cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime