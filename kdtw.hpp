/**
 * C++11 implementation of Regularized Dynamic Time Warping Kernel (KDTW),
 * described in the paper: P-F. Marteau and S. Gibet, On recursive edit distance
 * kernels with application to time series classification, IEEE Transactions on
 * Neural Networks and Learning Systems 26(6): 1121–1133, 2015.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <exception>
#include <vector>

// Row-major matrix to store distances
template <class T>
class Matrix {
  private:
    int64_t m_nrows = 0;
    int64_t m_ncols = 0;
    std::vector<T> m_mat;

  public:
    Matrix() = default;

    Matrix(int64_t _nrows, int64_t _ncols)
        : m_nrows(_nrows), m_ncols(_ncols), m_mat(_nrows * _ncols) {}

    void resize(int64_t _nrows, int64_t _ncols) {
        m_nrows = _nrows;
        m_ncols = _ncols;
        m_mat.resize(_nrows * _ncols);
    }

    const T& operator()(int64_t i, int64_t j) const {
        return m_mat[i * m_ncols + j];
    }
    T& operator()(int64_t i, int64_t j) {
        return m_mat[i * m_ncols + j];
    }

    const T& get(int64_t i, int64_t j) const {
        return m_mat[i * m_ncols + j];
    }
    void set(int64_t i, int64_t j, const T& v) {
        m_mat[i * m_ncols + j] = v;
    }

    int64_t nrows() const {
        return m_nrows;
    }
    int64_t ncols() const {
        return m_ncols;
    }
};

using MatrixF64 = Matrix<double>;

// Approximate LogSumExp implementation
double logsumexp(double x, double y) {
    static constexpr double MINUS_LOG_EPSILON = 50.0;
    const double vmin = std::min(x, y);
    const double vmax = std::max(x, y);
    if (vmax > vmin + MINUS_LOG_EPSILON) {
        return vmax;
    } else {
        return vmax + std::log(std::exp(vmin - vmax) + 1.0);
    }
}
double logsumexp(double x, double y, double z) {
    return logsumexp(logsumexp(x, y), z);
}

// 1st Exponentiated Recursive Accumulation of f-cost Products (in log)
double log_RAfP1(const MatrixF64& dist_mat, const double nu, const double c) {
    const int64_t nrows = dist_mat.nrows();
    const int64_t ncols = dist_mat.ncols();

    if ((nrows <= 0) or (ncols <= 0)) {
        throw std::runtime_error("Error: input matrix is empty");
    }

    const double log_c = std::log(c);
    auto get = [&](int64_t i, int64_t j) -> double {
        return -nu * dist_mat(i - 1, j - 1) - log_c;
    };
    std::vector<double> prev_row(ncols + 1), curr_row(ncols + 1);

    // 1st row (The initial value 0.0 means C = 1.0, i.e., log(C) = 0.0)
    curr_row[0] = 0.0;
    for (int64_t j = 1; j <= ncols; j++) {
        curr_row[j] = curr_row[j - 1] + get(1, j);
    }

    // Next rows
    for (int64_t i = 1; i <= nrows; i++) {
        std::swap(prev_row, curr_row);
        curr_row[0] = prev_row[0] + get(i, 1);
        for (int64_t j = 1; j <= ncols; j++) {
            curr_row[j] = get(i, j) + logsumexp(prev_row[j],  //
                                                prev_row[j - 1],  //
                                                curr_row[j - 1]);
        }
    }
    return curr_row[ncols];
}

// 2st Exponentiated Recursive Accumulation of f-cost Products (in log)
double log_RAfP2(const MatrixF64& dist_mat, const double nu, const double c) {
    const int64_t nrows = dist_mat.nrows();
    const int64_t ncols = dist_mat.ncols();

    if ((nrows <= 0) or (ncols <= 0) or (nrows != ncols)) {
        throw std::runtime_error("Error: input matrix is empty");
    }

    const double log_c = std::log(c);
    auto get = [&](int64_t i, int64_t j) -> double {
        return -nu * dist_mat(i - 1, j - 1) - log_c;
    };
    std::vector<double> prev_row(ncols + 1), curr_row(ncols + 1);

    // 1st row
    curr_row[0] =
        0.0;  // The initial value 0.0 means C = 1.0, i.e., log(C) = 0.0
    for (int64_t j = 1; j <= ncols; j++) {
        curr_row[j] = curr_row[j - 1] + get(j, j);
    }

    // Next rows
    for (int64_t i = 1; i <= nrows; i++) {
        std::swap(prev_row, curr_row);
        curr_row[0] = prev_row[0] + get(i, i);
        for (int64_t j = 1; j <= ncols; j++) {
            if (i == j) {
                curr_row[j] = logsumexp(prev_row[j - 1] + get(i, j),
                                        prev_row[j] + get(i, i),
                                        curr_row[j - 1] + get(j, j));
            } else {
                curr_row[j] = logsumexp(prev_row[j] + get(i, i),
                                        curr_row[j - 1] + get(j, j));
            }
        }
    }
    return curr_row[ncols];
}

// Exponentiated REDK based on DTW (in log), i.e., returns log(K(A,B))
double log_KDTW(const MatrixF64& dist_mat, const double nu, const double c) {
    const int64_t nrows = dist_mat.nrows();
    const int64_t ncols = dist_mat.ncols();
    if ((nrows <= 0) or (ncols <= 0)) {
        throw std::runtime_error("Error: input matrix is empty");
    } else if (nrows != ncols) {  // Then, RAfP2() is always 0
        return log_RAfP1(dist_mat, nu, c);
    } else {
        return logsumexp(log_RAfP1(dist_mat, nu, c),
                         log_RAfP2(dist_mat, nu, c));
    }
}
