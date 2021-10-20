#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <random>

using namespace sycl;

const uint N = 1 << 8;
const uint B = 1 << 5;

int64_t condense(queue &q, const double *mat, double *det) {
  double *mat_ = (double *)malloc(sizeof(double) * N * N);
  double *tmp = (double *)malloc(sizeof(double) * N * N);
  double *arr = (double *)malloc(sizeof(double) * (N - 2));

  memcpy(mat_, mat, sizeof(double) * N * N);
  memset(tmp, 0, sizeof(double) * N * N);
  memset(arr, 0, sizeof(double) * (N - 2));

  buffer<double, 2> b_mat{mat_, range<2>{N, N}};
  buffer<double, 2> b_tmp{tmp, range<2>{N, N}};
  buffer<double, 1> b_arr{arr, range<1>{N - 2}};

  std::chrono::_V2::steady_clock::time_point start =
      std::chrono::steady_clock::now();

  for (uint i = 0; i < N - 2; i++) {
    int l[1] = {-1};
    {
      buffer<int, 1> b_l{l, range<1>{1}};

      q.submit([&](handler &h) {
        accessor<double, 2, access::mode::read, access::target::global_buffer>
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
      std::chrono::_V2::steady_clock::time_point end =
          std::chrono::steady_clock::now();
      return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
    }

    uint l_idx = l[0];
    double pivot = 0.f;
    {
      host_accessor<double, 2, access::mode::read> h_mat{b_mat, range<2>{1, 1},
                                                         id<2>{i, i + l_idx}};
      pivot = h_mat[0][0];
    }

    q.submit([&](handler &h) {
      accessor<double, 2, access::mode::read_write,
               access::target::global_buffer>
          a_mat{b_mat, h, range<2>{N - i, 1}, id<2>{i, i + l_idx}};
      accessor<double, 1, access::mode::write, access::target::global_buffer>
          a_arr{b_arr, h, range<1>{1}, id<1>{i}, noinit};

      h.parallel_for(nd_range<1>{range<1>{N - i}, range<1>{N - i > B ? B : 2}},
                     [=](nd_item<1> it) {
                       const size_t r = it.get_global_id(0);

                       a_mat[r][0] /= pivot;

                       if (r == 0) {
                         a_arr[0] = pivot;
                       }
                     });
    });

    q.submit([&](handler &h) {
      accessor<double, 2, access::mode::read, access::target::global_buffer>
          a_mat{b_mat, h, range<2>{N - i, N - i}, id<2>{i, i}};
      accessor<double, 2, access::mode::write, access::target::global_buffer>
          a_tmp{b_tmp, h, range<2>{N - (i + 1), N - (i + 1)},
                id<2>{i + 1, i + 1}, noinit};
      accessor<double, 1, access::mode::read_write, access::target::local>
          a_lds{range<1>{1}, h};

      h.parallel_for(nd_range<2>{range<2>{N - (i + 1), N - (i + 1)},
                                 range<2>{1, N - (i + 1) > B ? B : 2}},
                     [=](nd_item<2> it) {
                       ONEAPI::sub_group sg = it.get_sub_group();
                       if (ONEAPI::leader(sg)) {
                         a_lds[0] = a_mat[0][l_idx];
                       }

                       sg.barrier();

                       const size_t r = it.get_global_id(0);
                       const size_t c = it.get_global_id(1);

                       if (c >= l_idx) {
                         a_tmp[r][c] = a_lds[0] * a_mat[r + 1][c + 1] -
                                       a_mat[0][c + 1] * a_mat[r + 1][l_idx];
                       } else {
                         a_tmp[r][c] = -1.f * a_mat[r + 1][c] * a_lds[0];
                       }
                     });
    });

    q.submit([&](handler &h) {
      accessor<double, 2, access::mode::write, access::target::global_buffer>
          a_mat{b_mat, h, range<2>{N - (i + 1), N - (i + 1)},
                id<2>{i + 1, i + 1}};
      accessor<double, 2, access::mode::read, access::target::global_buffer>
          a_tmp{b_tmp, h, range<2>{N - (i + 1), N - (i + 1)},
                id<2>{i + 1, i + 1}};

      h.copy(a_tmp, a_mat);
    });
  }

  host_accessor<double, 2, access::mode::read> h_mat{b_mat, range<2>{2, 2},
                                                     id<2>{N - 2, N - 2}};
  double lst_det = h_mat[0][0] * h_mat[1][1] - h_mat[0][1] * h_mat[1][0];

  host_accessor<double, 1, access::mode::read> h_arr{b_arr};
  double mult = 1.f;

  for (uint i = 0; i < N - 2; i++) {
    mult *= h_arr[i];
  }

  *det = lst_det * mult;

  std::chrono::_V2::steady_clock::time_point end =
      std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
      .count();
}

void random_matrix(double *const mat) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(-1.f, 1.f);

  for (uint i = 0; i < N; i++) {
    for (uint j = 0; j < N; j++) {
      mat[i * N + j] = dis(gen);
    }
  }
}

void hilbert_matrix(double *const mat) {
  for (uint i = 0; i < N; i++) {
    for (uint j = 0; j < N; j++) {
      mat[i * N + j] = 1.f / (double)(i + j + 1);
    }
  }
}

void show(const double *mat) {
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

  double det = 0;
  double *mat = (double *)malloc(sizeof(double) * N * N);

  random_matrix(mat);

  int64_t ts = condense(q, mat, &det);
  std::cout << "computed determinant " << det << ", in " << ts << " ms"
            << std::endl;

  std::free(mat);

  return 0;
}
