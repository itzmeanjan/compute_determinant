# compute_determinant
Parallel Matrix Determinant Computation, in SYCL DPC++

## Background

I implement Pivot Condensation-based Square Matrix Determinant Computation method in SIMT model, using SYCL DPC++.

Sequential implementation is [here](https://gist.github.com/itzmeanjan/c7f4dca2374484c388393f19e06c33ea)

Original Paper which I followed is [here](https://doi.org/10.1088/1742-6596%2F341%2F1%2F012031).

## Usage

For running this implementation, make sure you've Intel oneAPI toolkits installed. I found [it](https://software.intel.com/content/www/us/en/develop/documentation/installation-guide-for-intel-oneapi-toolkits-linux/top/installation/install-using-package-managers/apt.html#apt) helpful.

---

- Compile with

```bash
make

# or clang++ -fsycl main.cpp -o run
```

- Run binary, while offloading certain computations to default accelerator

```bash
./run
```
