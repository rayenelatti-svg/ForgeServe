#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kDefaultElements = 1 << 24;
constexpr int kDefaultWarmupIterations = 10;
constexpr int kDefaultTimedIterations = 100;

__global__ void vector_add(const float* a, const float* b, float* c,
                           std::size_t n) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < n) {
    c[index] = a[index] + b[index];
  }
}

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    std::cerr << operation << " failed: " << cudaGetErrorString(status)
              << '\n';
    std::exit(EXIT_FAILURE);
  }
}

std::size_t parse_size(const char* value, const char* name) {
  try {
    const auto parsed = std::stoull(value);
    if (parsed == 0) {
      throw std::invalid_argument("zero");
    }
    return parsed;
  } catch (const std::exception&) {
    std::cerr << name << " must be a positive integer\n";
    std::exit(EXIT_FAILURE);
  }
}

int parse_iterations(const char* value, const char* name) {
  const auto parsed = parse_size(value, name);
  if (parsed > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    std::cerr << name << " is too large\n";
    std::exit(EXIT_FAILURE);
  }
  return static_cast<int>(parsed);
}

}  // namespace

int main(int argc, char** argv) {
  const std::size_t elements =
      argc > 1 ? parse_size(argv[1], "elements") : kDefaultElements;
  const int warmup_iterations =
      argc > 2 ? parse_iterations(argv[2], "warmup iterations")
               : kDefaultWarmupIterations;
  const int timed_iterations =
      argc > 3 ? parse_iterations(argv[3], "timed iterations")
               : kDefaultTimedIterations;

  int device = 0;
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, device),
             "cudaGetDeviceProperties");
  check_cuda(cudaSetDevice(device), "cudaSetDevice");

  const std::size_t bytes = elements * sizeof(float);
  std::vector<float> host_a(elements);
  std::vector<float> host_b(elements);
  std::vector<float> host_c(elements, 0.0f);
  std::mt19937 generator(12345);
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
  for (std::size_t i = 0; i < elements; ++i) {
    host_a[i] = distribution(generator);
    host_b[i] = distribution(generator);
  }

  float* device_a = nullptr;
  float* device_b = nullptr;
  float* device_c = nullptr;
  check_cuda(cudaMalloc(&device_a, bytes), "cudaMalloc(device_a)");
  check_cuda(cudaMalloc(&device_b, bytes), "cudaMalloc(device_b)");
  check_cuda(cudaMalloc(&device_c, bytes), "cudaMalloc(device_c)");
  check_cuda(cudaMemcpy(device_a, host_a.data(), bytes, cudaMemcpyHostToDevice),
             "cudaMemcpy(device_a)");
  check_cuda(cudaMemcpy(device_b, host_b.data(), bytes, cudaMemcpyHostToDevice),
             "cudaMemcpy(device_b)");

  constexpr int threads_per_block = 256;
  const int blocks = static_cast<int>(
      (elements + threads_per_block - 1) / threads_per_block);
  for (int i = 0; i < warmup_iterations; ++i) {
    vector_add<<<blocks, threads_per_block>>>(device_a, device_b, device_c,
                                               elements);
  }
  check_cuda(cudaGetLastError(), "vector_add warm-up launch");
  check_cuda(cudaDeviceSynchronize(), "warm-up synchronization");

  cudaEvent_t start{};
  cudaEvent_t stop{};
  check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
  check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");
  check_cuda(cudaEventRecord(start), "cudaEventRecord(start)");
  for (int i = 0; i < timed_iterations; ++i) {
    vector_add<<<blocks, threads_per_block>>>(device_a, device_b, device_c,
                                               elements);
  }
  check_cuda(cudaGetLastError(), "vector_add timed launch");
  check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
  check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

  float elapsed_ms = 0.0f;
  check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
             "cudaEventElapsedTime");
  check_cuda(cudaMemcpy(host_c.data(), device_c, bytes, cudaMemcpyDeviceToHost),
             "cudaMemcpy(device_c)");

  constexpr float tolerance = 1e-5f;
  std::size_t mismatches = 0;
  float max_error = 0.0f;
  for (std::size_t i = 0; i < elements; ++i) {
    const float expected = host_a[i] + host_b[i];
    const float error = std::abs(host_c[i] - expected);
    max_error = std::max(max_error, error);
    if (error > tolerance) {
      ++mismatches;
    }
  }

  const double average_ms = elapsed_ms / timed_iterations;
  const double bytes_per_iteration =
      static_cast<double>(elements) * 3.0 * sizeof(float);
  const double bandwidth_gb_per_second =
      bytes_per_iteration / (average_ms * 1.0e6);
  std::cout << std::fixed << std::setprecision(3)
            << "device=" << properties.name << '\n'
            << "compute_capability=" << properties.major << '.'
            << properties.minor << '\n'
            << "elements=" << elements << '\n'
            << "blocks=" << blocks << '\n'
            << "threads_per_block=" << threads_per_block << '\n'
            << "warmup_iterations=" << warmup_iterations << '\n'
            << "timed_iterations=" << timed_iterations << '\n'
            << "total_kernel_time_ms=" << elapsed_ms << '\n'
            << "average_kernel_time_ms=" << average_ms << '\n'
            << "effective_bandwidth_gb_per_second=" << bandwidth_gb_per_second
            << '\n'
            << "max_error=" << max_error << '\n'
            << "validation=" << (mismatches == 0 ? "PASS" : "FAIL") << '\n';

  check_cuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
  check_cuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
  check_cuda(cudaFree(device_a), "cudaFree(device_a)");
  check_cuda(cudaFree(device_b), "cudaFree(device_b)");
  check_cuda(cudaFree(device_c), "cudaFree(device_c)");
  return mismatches == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
