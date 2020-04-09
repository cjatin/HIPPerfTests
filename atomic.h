#pragma once

template<typename T>
__global__ void atomicAddKernel(T *a) {
    atomicAdd(a, 2.0);
}

static void BM_atomic_add_float(benchmark::State& state) {
  BENCHMARK_GPU_INIT();
  float data[32];
  for (int i = 0; i < 32; i++) {
    data[i] = 1.21f * i;
  }
  float *d_a;
  hipMalloc(&d_a, sizeof(float) * 32);
  for (auto _ : state) {
    hipMemcpy(d_a, data, sizeof(float) * 32, hipMemcpyHostToDevice);
    BENCHMARK_GPU_START();
    hipLaunchKernelGGL(atomicAddKernel<float>, 1, 32, 0, 0, d_a);
    BENCHMARK_GPU_STOP();
    benchmark::DoNotOptimize(d_a);
  }
  hipFree(d_a);
}

GPUBENCHMARK(BM_atomic_add_float);

static void BM_atomic_add_double(benchmark::State& state) {
  BENCHMARK_GPU_INIT();
  double data[32];
  for (int i = 0; i < 32; i++) {
    data[i] = 1.21 * i;
  }
  double *d_a;
  hipMalloc(&d_a, sizeof(double) * 32);
  for (auto _ : state) {
    hipMemcpy(d_a, data, sizeof(double), hipMemcpyHostToDevice);
    BENCHMARK_GPU_START();
    hipLaunchKernelGGL(atomicAddKernel<double>, 1, 1, 0, 0, d_a);
    BENCHMARK_GPU_STOP();
    benchmark::DoNotOptimize(d_a);
  }
  hipFree(d_a);
}
GPUBENCHMARK(BM_atomic_add_double);
