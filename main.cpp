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

  buffer<float, 2> b_mat{mat_, range<2>{N, N}};
  buffer<float, 2> b_tmp{tmp, range<2>{N, N}};
  buffer<float, 1> b_arr{mat, range<1>{N - 2}};

  for (uint i = 0; i < N - 2; i++) {
    int l[1] = {-1};
    {
      buffer<int, 1> b_l{l, range<1>{1}};

      q.submit([&](handler &h) {
        accessor<float, 2, access::mode::read, access::target::global_buffer>
            a_mat{b_mat, h, range<2>{1, N - i}, id<2>{i, i}};
        accessor<int, 1, access::mode::write, access::target::global_buffer>
            a_l{b_l, h};

        h.single_task([=]() {
          for (uint j = 0; j < N - i; j++) {
            if (a_mat[0][j] != 0.f) {
              a_l[0] = j;
              break;
            };
          }
        });
      });
    }

    if (l[0] == -1) {
      *det = 0;
      return;
    }

    uint l_idx = l[0];

    q.submit([&](handler &h) {
      accessor<float, 2, access::mode::read, access::target::global_buffer>
          a_mat{b_mat, h, range<2>{N - i, N - i}, id<2>{i, i}};
      accessor<float, 2, access::mode::write, access::target::global_buffer>
          a_tmp{b_tmp, h, range<2>{N - (i + 1), N - (i + 1)},
                id<2>{i + 1, i + 1}};
      accessor<float, 1, access::mode::write, access::target::global_buffer>
          a_arr{b_arr, h, range<1>{1}, id<1>{i}};

      h.parallel_for(nd_range<2>{range<2>{N - (i + 1), N - (i + 1)},
                                 range<2>{1, N - (i + 1) > B ? B : 2}},
                     [=](nd_item<2> it) {
                       const size_t r = it.get_global_id(0);
                       const size_t c = it.get_global_id(1);

                       if (c >= l_idx) {
                         a_tmp[r][c] = a_mat[0][l_idx] * a_mat[r + 1][c + 1] -
                                       a_mat[0][c + 1] * a_mat[r + 1][l_idx];
                       } else {
                         a_tmp[r][c] = (0 - a_mat[r + 1][c] * a_mat[0][l_idx]);
                       }

                       if (r == 0 && c == 0) {
                         a_arr[0] =
                             sycl::pow(a_mat[0][l_idx], (float)((N - i) - 2));
                       }
                     });
    });

    auto evt = q.submit([&](handler &h) {
      accessor<float, 2, access::mode::write, access::target::global_buffer>
          a_mat{b_mat, h, range<2>{N - (i + 1), N - (i + 1)},
                id<2>{i + 1, i + 1}};
      accessor<float, 2, access::mode::read, access::target::global_buffer>
          a_tmp{b_tmp, h, range<2>{N - (i + 1), N - (i + 1)},
                id<2>{i + 1, i + 1}};

      h.copy(a_tmp, a_mat);
    });
    evt.wait();
  }

  host_accessor<float, 2, access::mode::read> h_mat{b_mat, range<2>{2, 2},
                                                    id<2>{N - 2, N - 2}};
  float lst_det = h_mat[0][0] * h_mat[1][1] - h_mat[0][1] * h_mat[1][0];

  host_accessor<float, 1, access::mode::read> h_arr{b_arr};
  float mult = 1.f;

  for (uint i = 0; i < N - 2; i++) {
    mult *= h_arr[i];
  }

  *det = lst_det / mult;
}

void example_matrix(float *const mat) {
  mat[0 * N + 0] = 1;
  mat[0 * N + 1] = 2;
  mat[0 * N + 2] = -1;
  mat[0 * N + 3] = 3;

  mat[1 * N + 0] = 2;
  mat[1 * N + 1] = 1;
  mat[1 * N + 2] = -2;
  mat[1 * N + 3] = 3;

  mat[2 * N + 0] = 3;
  mat[2 * N + 1] = 1;
  mat[2 * N + 2] = 2;
  mat[2 * N + 3] = 1;

  mat[3 * N + 0] = 1;
  mat[3 * N + 1] = -1;
  mat[3 * N + 2] = 0;
  mat[3 * N + 3] = 2;
}

void show(const float *mat) {
  for (uint i = 0; i < N; i++) {
    for (uint j = 0; j < N; j++) {
      std::cout << mat[i * N + j] << "\t";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

int main() {
  device d{default_selector{}};
  queue q{d};
  std::cout << "running on " << d.get_info<info::device::name>() << "\n"
            << std::endl;

  float *mat = (float *)malloc(sizeof(float) * N * N);
  float det = 0;
  example_matrix(mat);

  condense(q, mat, &det);
  std::cout << "determinant " << det << std::endl;

  std::free(mat);

  return 0;
}
