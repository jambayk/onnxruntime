# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import sys
from dataclasses import dataclass

import kernel_explorer as ke
import numpy as np
from utils import dtype_to_bytes


def dtype_to_funcs(dtype):
    type_map = {
        "float16": list(filter(lambda x: "MatrixFloatBnb4_half" in x, dir(ke))),
        "float32": list(filter(lambda x: "MatrixFloatBnb4_float" in x, dir(ke))),
    }
    return type_map[dtype]


def dtype_to_funcs_cublas(dtype):
    type_map = {
        "float16": list(filter(lambda x: "GemmBenchmark_half" in x, dir(ke))),
        "float32": list(filter(lambda x: "GemmBenchmark_float" in x, dir(ke))),
    }
    return type_map[dtype]


quant_enums = {"FP4": 0, "NF4": 1}
quant_maps = {
    "FP4": [0, 0.0625, 8.0, 12.0, 4.0, 6.0, 2.0, 3.0, -0, -0.0625, -8.0, -12.0, -4.0, -6.0, -2.0, -3.0],
    "NF4": [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
}


dtypes = ["float16", "float32"]
quant_types = ["FP4", "NF4"]


@dataclass
class MatrixMulMetric(ke.BandwidthMetric):
    m: int
    n: int
    k: int

    def report(self):
        return (
            f"{self.duration:6.2f} us {self.gbps:5.2f} GB/s {self.dtype} m={self.m} n={self.n} k={self.k} {self.name}"
        )


@dataclass
class MatrixFpBnb4Metric(MatrixMulMetric):
    quant_type: str

    def report(self):
        return (
            f"{self.duration:6.2f} us {self.gbps:5.2f} GB/s"
            f" {self.quant_type} {self.dtype} m={self.m} n={self.n} k={self.k} {self.name}"
        )


def profile_matmul_fp_bnb4_func(quant_type, m, n, k, dtype, func):
    np.random.seed(0)
    block_size = 64
    numel = n * k
    output = np.random.rand(m, n).astype(dtype)
    a = np.random.rand(m, k).astype(dtype)
    b = np.random.randint(low=0, high=255, size=(numel + 1) // 2).astype("uint8")
    # absmax = np.random.rand((numel + block_size - 1) // block_size).astype(dtype)
    absmax = np.random.rand((numel + block_size - 1) // block_size).astype("float32")
    quant_map = np.array(quant_maps[quant_type]).astype("float32")
    quant_map /= np.max(np.abs(quant_map))

    output_d = ke.DeviceArray(output)
    a_d = ke.DeviceArray(a)
    b_d = ke.DeviceArray(b)
    absmax_d = ke.DeviceArray(absmax)
    quant_map_d = ke.DeviceArray(quant_map)
    f = getattr(ke, func)

    my_op = f(output_d, a_d, b_d, absmax_d, quant_map_d, m, n, k)
    duration_ms = my_op.Profile()
    total_bytes = (m * k + n * k + m * n) * (dtype_to_bytes(dtype))

    ke.report(MatrixFpBnb4Metric(func, dtype, duration_ms, total_bytes, m, n, k, quant_type))


def profile_gemm_func(m, n, k, dtype, func):
    np.random.seed(0)
    output = np.random.rand(m, n).astype(dtype)
    a = np.random.rand(m, k).astype(dtype)
    b = np.random.rand(k, n).astype(dtype)

    output_d = ke.DeviceArray(output)
    a_d = ke.DeviceArray(a)
    b_d = ke.DeviceArray(b)
    f = getattr(ke, func)
    my_op = f(output_d, a_d, b_d, m, n, k)
    duration_ms = my_op.Profile()
    total_bytes = (m * k + n * k + m * n) * (dtype_to_bytes(dtype))

    ke.report(MatrixMulMetric(func, dtype, duration_ms, total_bytes, m, n, k))


def profile_with_args(qt, m, n, k, dtype, sort):
    with ke.benchmark(sort):
        for func in dtype_to_funcs(dtype):
            profile_matmul_fp_bnb4_func(qt, m, n, k, dtype, func)

        for func in dtype_to_funcs_cublas(dtype):
            profile_gemm_func(m, n, k, dtype, func)


def profile():
    dims_m = [1]
    for qt in quant_types:
        for dt in dtypes:
            for m in dims_m:
                for n, k in ((4096, 4096), (4096, 12288), (12288, 4096)):
                    profile_with_args(qt, m, n, k, dt, False)
                    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    group = parser.add_argument_group("profile with args")
    group.add_argument("m", type=int)
    group.add_argument("n", type=int)
    group.add_argument("k", type=int)
    group.add_argument("quant_type", choices=quant_types)
    group.add_argument("dtype", choices=dtypes)
    group.add_argument("--sort", action="store_true")

    if len(sys.argv) == 1:
        profile()
    else:
        args = parser.parse_args()
        profile_with_args(args.quant_type, args.m, args.n, args.k, args.dtype, args.sort)
