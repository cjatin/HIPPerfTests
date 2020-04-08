#pragma once

template<typename T>
__global__ void atomicAddKernel(T *a) {
    atomicAdd(a, 2.0);
}

static void BM_atomic_add_float(benchmark::State& state) {
  for (auto _ : state) {
    BENCHMARK_GPU_INIT();
    float* d_a;
    hipMalloc(&d_a, sizeof(float));
    float tf = 3.0f;
    hipMemcpy(d_a, &tf, sizeof(float), hipMemcpyHostToDevice);
    BENCHMARK_GPU_START();
    hipLaunchKernelGGL(atomicAddKernel<float>, 1, 1, 0, 0, d_a);
    BENCHMARK_GPU_STOP();
    benchmark::DoNotOptimize(d_a);
    hipFree(d_a);
  }
}

GPUBENCHMARK(BM_atomic_add_float);

static void BM_atomic_add_double(benchmark::State& state) {
  for (auto _ : state) {
    BENCHMARK_GPU_INIT();
    double* d_a;
    hipMalloc(&d_a, sizeof(double));
    double tf = 3.0f;
    hipMemcpy(d_a, &tf, sizeof(double), hipMemcpyHostToDevice);
    BENCHMARK_GPU_START();
    hipLaunchKernelGGL(atomicAddKernel<double>, 1, 1, 0, 0, d_a);
    BENCHMARK_GPU_STOP();
    benchmark::DoNotOptimize(d_a);
    hipFree(d_a);
  }
}
GPUBENCHMARK(BM_atomic_add_double);
