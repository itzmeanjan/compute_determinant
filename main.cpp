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
    int l[1] = {-1};
    {
      buffer<int, 1> b_l{l, range<1>{1}};

      q.submit([&](handler &h) {
        accessor<float, 2, access::mode::read, access::target::global_buffer>
            a_mat{b_mat, h, range<2>{1, N - i}, id<2>{i, i}};
        accessor<int, 1, access::mode::read_write,
                 access::target::global_buffer>
            a_l{b_l, h};

        h.single_task([=]() {
          for (uint j = i; j < N; j++) {
            if (a_mat[i][j] != 0.f) {
              a_l[0] = j;
              break;
            };
          }
        });
      });
    }

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
