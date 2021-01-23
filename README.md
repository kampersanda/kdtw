# kdtw: Regularized Dynamic Time Warping Kernel
This provides a (yet another) C++11 implementation of the [Regularized Dynamic Time Warping Kernel (KDTW)](https://people.irisa.fr/Pierre-Francois.Marteau/REDK/KDTW/KDTW.html) and its Python binding.

KDTW is also known as the Exponentiated Recursive Edit Distance Kernel (REDK) based on DTW, described in the paper: P-F. Marteau and S. Gibet, [On Recursive Edit Distance Kernels with Application to Time Series Classification](https://arxiv.org/abs/1005.5141), IEEE Transactions on Neural Networks and Learning Systems 26(6): 1121–1133, 2015.

## How to install

### C++11

Just add the header file `kdtw.hpp` to your own project.

### Python

You can build the Python module `pykdtw` from `pykdtw.cpp` through [pybind11](https://github.com/pybind/pybind11), as the following command.

```shell
$ python -m pip install .
```

If necessary, set the include path for pybind11 at line 19 of `setup.py`.

## Sample usage

### C++11

`sample.cpp` provides a sample usage. You can use `CMakeLists.txt` to compile it.

```c++
#include <iostream>
#include <random>
#include <vector>

#include "kdtw.hpp"

int main() {
    const int64_t P = 20;  // Length of one sequence
    const int64_t Q = 20;  // Length of the other sequence

    // Two random sequences
    std::vector<double> A(P), B(Q);
    {
        std::mt19937 engine(7);
        std::uniform_real_distribution<double> dist(0, 1);
        std::for_each(A.begin(), A.end(), [&](double& e) { e = dist(engine); });
        std::for_each(B.begin(), B.end(), [&](double& e) { e = dist(engine); });
    }

    // Make the P \times Q distance matrix
    // You can use dist_mat.resize(P,Q) not to reconstruct the matrix
    MatrixF64 dist_mat(P, Q);
    {
        for (int64_t i = 0; i < P; i++) {
            for (int64_t j = 0; j < Q; j++) {
                dist_mat(i, j) = std::abs(A[i] - B[j]);  // Lp norm
            }
        }
    }

    // Parameters for KDTW
    const double C = 3;  // non negative constant (typically 3)
    const double NU[] = {0.01, 0.1, 1};  // stiffness parameters (ν')

    for (double nu : NU) {
        // Compute the kernel value (in log)
        std::cout << "K = " << std::exp(log_KDTW(dist_mat, nu, C))
                  << "\t(ν' = " << nu << ")" << std::endl;
    }

    return 0;
}
```

### Python

`sample.py` provides a sample usage.

```python
#!/usr/bin/env python3

import math
import random
import pykdtw

random.seed(7)  # set seed

P = 20  # Length of one sequence
Q = 20  # Length of the other sequence

# Random sequences
A = [random.random() for _ in range(P)]
B = [random.random() for _ in range(Q)]

# Make the P \times Q distance matrix
# You can use dist_mat.resize(P,Q) not to reconstruct the matrix
dist_mat = pykdtw.MatrixF64(P, Q)
for i, a in enumerate(A):
    for j, b in enumerate(B):
        dist_mat.set(i, j, abs(a - b))  # Lp norm

# Parameters for KDTW
C = 3  # non negative constant (typically 3)
NU = [0.01, 0.1, 1]  # stiffness parameter (ν')

for nu in NU:
    # Compute the kernel value (in log)
    print(f"K = {math.exp(pykdtw.log_KDTW(dist_mat, nu, C)):g}\t(ν' = {nu})")
```

