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
