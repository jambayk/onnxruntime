// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <algorithm>
#include <cmath>

namespace onnxruntime {
namespace contrib {

#if defined(_MSC_VER)
#define FORCEINLINE __forceinline
#else
#define FORCEINLINE __attribute__((always_inline)) inline
#endif

// TODO(jambayk): Does this improve performance?
FORCEINLINE uint8_t QuantizeNF4(float x)
{
  if(x > 0.03979014977812767f)
    if(x > 0.3893125355243683f) // 1
      if(x > 0.6427869200706482f) // 11
        if(x > 0.8614784181118011f) // 111
          return 0b1111;
        else
          return 0b1110;
      else
        if(x > 0.5016634166240692f) // 110
          return 0b1101;
        else
          return 0b1100;
    else
      if(x > 0.2035212516784668f) // 10
        if(x > 0.2920137718319893f) // 101
          return 0b1011;
        else
          return 0b1010;
      else
        if(x > 0.1202552504837513f) // 100
          return 0b1001;
        else
          return 0b1000;
  else
    if(x > -0.33967943489551544f) // 0
      if(x > -0.13791173323988914f) // 01
        if(x > -0.045525018125772476f) // 011
          return 0b0111;
        else
          return 0b0110;
      else
        if(x > -0.23460740596055984f) // 010
          return 0b0101;
        else
          return 0b0100;
    else
      if(x > -0.6106329262256622f) // 00
        if(x > -0.4599952697753906f) // 001
          return 0b0011;
        else
          return 0b0010;
      else
        if(x > -0.8480964004993439f) // 000
          return 0b0001;
        else
          return 0b0000;
}

template <typename T, int32_t block_size>
FORCEINLINE void QuantizeBlockBnb4(const T* src, uint8_t* dst, float& scale_block, int32_t block_idx, int32_t numel){
  float absmax = 0.0f;

  int32_t block_len = std::min(block_size, numel - block_idx * block_size);
  int32_t src_offset = block_idx * block_size;
  int32_t dst_offset = block_idx * block_size / 2;

  for (int32_t idx = 0; idx < block_len; idx++) {
    const float v = static_cast<float>(src[src_offset + idx]);
    absmax = fmaxf(absmax, fabsf(v));
  }

  scale_block = absmax;
  const float reciprocal_scale = absmax ? 1.0f / absmax : 0.0f;

  for (int32_t idx = 0; idx < block_len; idx += 2) {
    const float v0 = src[src_offset + idx] * reciprocal_scale;
    const uint8_t vi0 = QuantizeNF4(v0);

    const float v1 = (idx + 1 < block_len) ? src[src_offset + idx + 1] * reciprocal_scale : 0;
    const uint8_t vi1 = QuantizeNF4(v1);

    dst[dst_offset + idx / 2] =  (vi0 << 4) | vi1;
  }
}

FORCEINLINE float DequantizeNF4(uint8_t val)
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

template <typename T, int32_t block_size>
FORCEINLINE void DequantizeBlockBnb4(const uint8_t* src, T* dst, float scale_block, int32_t block_idx, int32_t numel){
  int32_t block_len = std::min(block_size, numel - block_idx * block_size);
  int32_t src_offset = block_idx * block_size / 2;
  int32_t dst_offset = block_idx * block_size;

  for (int32_t idx = 0; idx < block_len; idx += 2) {
    const uint8_t val = src[src_offset + idx / 2];

    dst[dst_offset + idx] = static_cast<T>(DequantizeNF4(val >> 4) * scale_block);
    if(idx + 1 < block_len)
      dst[dst_offset + idx + 1] = static_cast<T>(DequantizeNF4(val & 0xF) * scale_block);
  }
}

}  // namespace contrib
}  // namespace onnxruntime
