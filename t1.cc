#include <iostream>
#include <hip/hip_runtime.h>

template <typename T>
__global__ void add(T* a, T* b, T* c) {
  *a = *b + *c;
}

template <typename T>
int call(T a, T b, void (*f)(T*, T*, T*)) {
  T c;
  T*d_a, *d_b, *d_c;
  hipMalloc(&d_a, sizeof(T));
  hipMalloc(&d_b, sizeof(T));
  hipMalloc(&d_c, sizeof(T));
  hipMemcpy(d_b, &a, sizeof(T), hipMemcpyHostToDevice);
  hipMemcpy(d_c, &b, sizeof(T), hipMemcpyHostToDevice);
  hipLaunchKernelGGL(f, 1, 1, 0, 0, d_a, d_b, d_c);

  hipMemcpy(&c, d_a,  sizeof(T), hipMemcpyDeviceToHost);
  return c;
}

int main() {
    int a = 10, b = 20;
    auto ff = add<int>;
    auto F = call<int>(a,b, add<int>);
    std::cout << F << std::endl;
}
