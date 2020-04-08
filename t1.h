#pragma once

static void BM_hipMalloc_B(benchmark::State& state) {
  BENCHMARK_GPU_DECLARE();
  for (auto _ : state) {
    BENCHMARK_GPU_BEGIN();
    int* d_a;
    hipMalloc(&d_a, sizeof(int));
    hipFree(d_a);
    BENCHMARK_GPU_END();
  }
  BENCHMARK_GPU_CLEANUP();
}
// Register the function as a benchmark
GPUBENCHMARK(BM_hipMalloc_B);

// Define another benchmark
static void BM_LaunchKernel_B(benchmark::State& state) {
  BENCHMARK_GPU_DECLARE();
  for (auto _ : state) {
    BENCHMARK_GPU_BEGIN();
    int* d_a;
    hipMalloc(&d_a, sizeof(int));
    hipLaunchKernelGGL(add, 1, 1, 0, 0, d_a);
    hipFree(d_a);
    BENCHMARK_GPU_END();
  }
  BENCHMARK_GPU_CLEANUP();
}
GPUBENCHMARK(BM_LaunchKernel_B);
