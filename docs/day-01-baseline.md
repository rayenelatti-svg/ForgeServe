# ForgeServe - Day 1 Baseline

## Environment

- GPU: NVIDIA GeForce RTX 4050 Laptop GPU
- VRAM: 6141 MiB reported by `nvidia-smi`
- Compute capability: 8.9
- Driver: 595.99.02
- Driver CUDA compatibility: 13.2
- CUDA toolkit: 12.4.131
- C++ compiler: GCC 15.2.0
- CMake: unavailable on `PATH` during this run; the existing build directory
  was created with CMake 4.4.4 metadata
- Nsight Systems: 2023.4.4.54
- Nsight Compute: 2024.1.1.0
- OS: Ubuntu, Linux kernel 7.0.0-38-generic, x86_64
- GPU state at collection: 43 C, P3, 2130 MHz SM clock, 7001 MHz memory clock

## Workload

- Operation: vector addition, `C[i] = A[i] + B[i]`
- Number of elements: 4,194,304
- Threads per block: 256
- Number of blocks: 16,384
- Warm-up launches: 10
- Timed launches: 100
- Logical traffic: 3 floats per element, or 50,331,648 bytes per launch
- Build flags used for profiling: `-O3 -std=c++17 -arch=sm_89
  --expt-relaxed-constexpr -lineinfo`

## Correctness

- Result: PASS on all five ordinary runs and the Nsight Compute run
- Maximum absolute error: 0.000
- Edge cases tested: positive element count, warm-up count, and timed count
  parsing; invalid and zero values are rejected by the executable
- Known limitations: the benchmark uses pageable host memory and one CUDA
  stream, and it does not measure end-to-end latency in its reported kernel
  timing

## Baseline measurements

Five ordinary runs used the same executable and arguments:
`./build/vector_add 4194304 10 100`.

| Run | Average kernel time (ms) | Effective bandwidth (GB/s) | Validation |
| ---: | ---: | ---: | --- |
| 1 | 0.292 | 172.465 | PASS |
| 2 | 0.293 | 171.950 | PASS |
| 3 | 0.292 | 172.566 | PASS |
| 4 | 0.292 | 172.343 | PASS |
| 5 | 0.291 | 172.730 | PASS |

- Median average kernel time: 0.292 ms
- Time range: 0.291-0.293 ms
- Median effective bandwidth: 172.465 GB/s
- Bandwidth range: 171.950-172.730 GB/s

The narrow spread is useful for this run, but it is not a claim that laptop
GPU timing will remain stable across power, thermal, or competing-workload
conditions.

## Nsight Systems

- Report: [`vector_add_step5.nsys-rep`](../reports/vector_add_step5.nsys-rep)
- Kernel launch behavior: 110 sequential launches (10 warm-up and 100 timed)
- Average kernel duration in the Systems trace: 1.151 ms
- Synchronization observed: one `cudaDeviceSynchronize` after warm-up and one
  `cudaEventSynchronize` after the timed loop
- Unexpected CPU/GPU activity: none in the repeated kernel sequence
- Memory transfers observed: three `cudaMemcpy` calls before the timed
  sequence; no transfers inside the repeated timed kernel region

The Systems run is diagnostic and includes profiler overhead; its duration is
not used as the authoritative baseline.

## Nsight Compute

- Report: [`vector_add_step6.ncu-rep`](../reports/vector_add_step6.ncu-rep)
- Collection: `ncu --set basic --launch-skip 10 --launch-count 1
  --export reports/vector_add_step6 --force <profile-build> 4194304 10 100`
- Grid and block: 16,384 blocks by 256 threads
- Registers per thread: 16
- Shared memory: 0 bytes static and dynamic; 1.02 KiB driver allocation
- Theoretical occupancy: 100%
- Achieved occupancy: 86.40%
- Achieved active warps per SM: 41.47
- DRAM throughput: 96.29% of peak sustained elapsed throughput
- Compute-memory throughput: 96.29% of peak sustained elapsed throughput
- Compute/SM throughput: 13.36%
- L1/TEX throughput: 17.18%
- L2 throughput: 22.75%
- Measured single-launch duration: 223.14 microseconds
- Source correlation: available in the profiling build because `-lineinfo` was
  enabled

### Conclusion and confidence

The kernel appears to be limited primarily by global-memory throughput:
DRAM throughput is 96.29% while compute throughput is only 13.36%, matching
the low arithmetic intensity of vector addition. Achieved occupancy is below
theoretical occupancy, but register and shared-memory usage do not indicate an
obvious occupancy limiter. Confidence: medium-high for this workload and
configuration. Nsight Compute replay changes execution conditions, so the
ordinary repeated baseline remains the performance comparison.

## Next experiment

- Hypothesis: changing the block size may alter scheduling efficiency or
  achieved occupancy, but should not materially improve a bandwidth-bound
  vector addition.
- One variable to change: threads per block, from 256 to 128 or 512
- Metric to compare: median average kernel time and effective bandwidth over
  five ordinary runs; inspect DRAM throughput for the best candidate
- Correctness check: require `validation=PASS` and `max_error=0.000` on every
  run
- Control conditions: same GPU, input size, warm-up/timed counts, build mode,
  power mode, and no concurrent GPU workload

