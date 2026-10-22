# ForgeServe CUDA vector-add benchmark

## Purpose

This is the first reproducible GPU measurement in ForgeServe. It keeps the
workload intentionally simple: each output element is `C[i] = A[i] + B[i]`.
The benchmark separates allocation and host-device copies from the timed
kernel region, performs warm-up launches, uses CUDA events for GPU timing, and
checks every output against a host reference.

## Build and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
./build/vector_add
```

The CUDA target includes `-lineinfo` so profiler source mapping is available
without turning the Release build into a debug build.

The executable accepts optional `elements`, `warmup_iterations`, and
`timed_iterations` arguments. Nsight profiling should use the same executable
and arguments:

```bash
nsys profile --trace=cuda,nvtx,osrt --output=reports/vector_add \
  ./build/vector_add
ncu --set full --launch-skip 10 --launch-count 1 \
  --export reports/vector_add_ncu --force \
  ./build/vector_add
```

The `--launch-skip` value skips the warm-up launches, so Nsight Compute
measures one steady-state kernel launch.

On the Ubuntu package of Nsight Systems 2023.4.4, `nsys profile` may report
that the importer binary is unavailable even though the importer is installed.
Import the generated trace explicitly:

```bash
/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter \
  -i reports/vector_add.qdstrm \
  -o reports/vector_add.nsys-rep \
  -f
nsys stats --report cuda_gpu_kern_sum --report cuda_api_sum \
  reports/vector_add.nsys-rep
```

## Initial environment

Captured on 2026-10-22:

| Component | Result |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB |
| Driver | 595.99.02 |
| Driver CUDA compatibility | 13.2 |
| CUDA toolkit | 12.4.131 (`nvcc`) |
| GCC | 15.2.0 |
| Nsight Systems | 2023.4.4.54 |
| Nsight Compute | 2024.1.1.0 |

The build targets compute capability 8.9, matching the RTX 4050. The actual
device-reported capability is printed by the executable and should be checked
before changing the architecture setting on another machine.

## Measurement notes

Vector addition moves three 4-byte values per element and performs one
floating-point addition. Its arithmetic intensity is therefore very low:
`1 FLOP / 12 bytes`. The expected first bottleneck is global-memory bandwidth,
not arithmetic throughput. Nsight Systems establishes launch and
synchronization behavior; Nsight Compute provides the kernel-level evidence
for memory throughput, occupancy, and instruction mix.

Do not treat the effective bandwidth as hardware peak bandwidth: it is a
useful workload metric that includes the kernel's actual load/store behavior.
Record the command, build configuration, GPU clocks/utilization, and profiler
report alongside each result in `reports/`.

## First baseline

The initial run used the default 16,777,216 elements, 10 warm-up launches, and
100 timed launches:

| Metric | Result |
| --- | ---: |
| Validation | PASS |
| Average kernel time | 1.167 ms |
| Effective bandwidth | 172.5 GB/s |
| Maximum absolute error | 0.000 |

The measured bandwidth and low arithmetic intensity make global memory traffic
the first optimization hypothesis. Nsight Compute results below support that
hypothesis.

Nsight Systems produced [`reports/vector_add_fixed.nsys-rep`](../reports/vector_add_fixed.nsys-rep).
Its kernel summary contains 110 launches averaging 1.177 ms. The installed
`nsys` CLI did not discover its packaged importer automatically, so the report
was generated with the explicit `QdstrmImporter` command above. The raw trace
remains available at
[`reports/vector_add_fixed.qdstrm`](../reports/vector_add_fixed.qdstrm).

## Profiler verification

GPU performance-counter access is available (`RmProfilingAdminOnly=0`).
Nsight Compute completed successfully and produced
[`reports/vector_add_ncu_fixed.ncu-rep`](../reports/vector_add_ncu_fixed.ncu-rep).
Its steady-state launch measured 1.08 ms, with DRAM and compute-memory
throughput at 93.15% of peak sustained elapsed throughput.

## Nsight Systems application profile

The complete application was captured with sampling disabled:

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none \
  --output=reports/vector_add_step5 ./build/vector_add
```

The installed CLI generated the raw trace
[`reports/vector_add_step5.qdstrm`](../reports/vector_add_step5.qdstrm).
Because this package does not automatically discover its importer, it was
converted explicitly to
[`reports/vector_add_step5.nsys-rep`](../reports/vector_add_step5.nsys-rep).
The generated SQLite export is
[`reports/vector_add_step5.sqlite`](../reports/vector_add_step5.sqlite).

The Systems summary reported 110 sequential kernel launches averaging
1.151 ms. The three `cudaMemcpy` calls occurred outside the repeated timed
kernel sequence, while the timed region was followed by one
`cudaEventSynchronize`. The warm-up was followed by one
`cudaDeviceSynchronize`. This matches the benchmark's intended measurement
boundaries; the Systems run itself is diagnostic and not the authoritative
performance measurement.
