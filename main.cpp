#include <CL/sycl.hpp>
#include <iostream>

using namespace sycl;

const uint N = 4;
const uint B = 4;

void condense(queue &q, const float *mat, float *det) {
  float *mat_ = (float *)malloc(sizeof(float) * N * N);
  float *tmp = (float *)malloc(sizeof(float) * N * N);
  float *arr = (float *)malloc(sizeof(float) * (N - 2));

  memcpy(mat_, mat, sizeof(float) * N * N);
  memset(tmp, 0, sizeof(float) * N * N);
  memset(arr, 0, sizeof(float) * (N - 2));

  buffer<float, 2> b_mat{mat, range<2>{N, N}};
  buffer<float, 2> b_tmp{tmp, range<2>{N, N}};
  buffer<float, 1> b_arr{mat, range<1>{N - 2}};

  uint n = N;
  for (uint i = 0; i < N - 2; i++) {
    q.submit([&](handler &h) {
      accessor<float, 2, access::mode::read, access::target::global_buffer>
          a_mat{b_mat, h};
      accessor<float, 2, access::mode::write, access::target::global_buffer>
          a_tmp{b_tmp, h};

      h.parallel_for(nd_range<2>{range<2>{N, N}, range<2>{1, B}},
                     [=](nd_item<2> it) {});
    });

    auto evt = q.submit([&](handler &h) {
      accessor<float, 2, access::mode::write, access::target::global_buffer>
          a_mat{b_mat, h};
      accessor<float, 2, access::mode::read, access::target::global_buffer>
          a_tmp{b_tmp, h};

      h.copy(a_tmp, a_mat);
    });
    evt.wait();
  }
}

int main() { return 0; }
