# ForgeServe vector-add baseline

Captured 2026-10-22 on an NVIDIA GeForce RTX 4050 Laptop GPU (compute
capability 8.9), using CUDA toolkit 12.4.131 and a Release build.

Command:

```bash
./build/vector_add
```

Result:

```text
validation=PASS
average_kernel_time_ms=1.167
effective_bandwidth_gb_per_second=172.527
max_error=0.000
```

The first optimization hypothesis is global-memory bandwidth: vector addition
performs one floating-point operation while reading two values and writing one,
so the workload has only `1 FLOP / 12 bytes` of arithmetic intensity. This
hypothesis is evaluated below with Nsight Compute after performance-counter
access was enabled.

## Profiler verification

GPU performance-counter access is now available (`RmProfilingAdminOnly=0`).
Nsight Compute completed successfully and produced
[`vector_add_ncu_fixed.ncu-rep`](./vector_add_ncu_fixed.ncu-rep). Its
steady-state launch measured 1.08 ms, with DRAM and compute-memory throughput
at 93.15% of peak sustained elapsed throughput.

Nsight Systems also produced
[`vector_add_fixed.nsys-rep`](./vector_add_fixed.nsys-rep). Its kernel summary
contains 110 launches averaging 1.177 ms. The installed `nsys` CLI did not
discover its packaged importer automatically, so the report was generated
with:

```bash
/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter \
  -i reports/vector_add_fixed.qdstrm \
  -o reports/vector_add_fixed.nsys-rep \
  -f
```
