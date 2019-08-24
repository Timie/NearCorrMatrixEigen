#ifndef NCORR_COMMON_H
#define NCORR_COMMON_H

#include <Eigen/Core>

namespace ncorr
{

/**
  * Tells whether the provided matrix is positive semi-definite.
  */
template <typename DerivedA>
bool
isPositiveSemiDefinite(Eigen::MatrixBase<DerivedA> &m)
{
    // Matrix must be absolutelly symmetric
    if(m.transpose() != m)
    {
        return false;
    }

    auto ldltStatus = m.ldlt().info();
    if((ldltStatus == Eigen::NumericalIssue) ||
       (ldltStatus == Eigen::InvalidInput))
    {
        return false;
    }

    return true;
}

/**
 * This method checks the symmetric elements off the main diagonal of the matrix.
 * If the elements are not symmetric, they are made symmetric by averaging them.
 */
template <typename DerivedA>
void
makeSymmetric(Eigen::MatrixBase<DerivedA> &m)
{
    for(Eigen::Index rowIdx = 1;
        rowIdx < m.rows();
        ++rowIdx)
    {

        for(Eigen::Index colIdx = 0;
            colIdx < rowIdx;
            ++colIdx)
        {
            if(m(rowIdx,colIdx) != m(colIdx, rowIdx))
            {
                m(rowIdx,colIdx) = (m(rowIdx,colIdx) + m(colIdx, rowIdx)) * 0.5;
                m(colIdx, rowIdx) = m(rowIdx,colIdx);
            }
        }
    }
}

template <typename DerivedA>
Eigen::Matrix<typename Eigen::MatrixBase<DerivedA>::Scalar,
              Eigen::MatrixBase<DerivedA>::RowsAtCompileTime,
              Eigen::MatrixBase<DerivedA>::ColsAtCompileTime>
getSymmetric(const Eigen::MatrixBase<DerivedA> &m)
{
    Eigen::Matrix<typename Eigen::MatrixBase<DerivedA>::Scalar,
                  Eigen::MatrixBase<DerivedA>::RowsAtCompileTime,
                  Eigen::MatrixBase<DerivedA>::ColsAtCompileTime> result = m;
    makeSymmetric(result);

    return result;
}

/**
  * Changes the covariance matrix to correlation matrix, and returns the standard deviations
  */

template <typename DerivedA>
Eigen::Matrix<typename Eigen::MatrixBase<DerivedA>::Scalar,
              Eigen::MatrixBase<DerivedA>::RowsAtCompileTime,
              1>
covarianceToCorrelation(Eigen::MatrixBase<DerivedA> &m)
{
    assert(m.cols() == m.rows());

    Eigen::Matrix<typename Eigen::MatrixBase<DerivedA>::Scalar,
                  Eigen::MatrixBase<DerivedA>::RowsAtCompileTime,
                  1> standardDevs = m.diagonal().array().sqrt();

    Eigen::Matrix<typename Eigen::MatrixBase<DerivedA>::Scalar,
                  Eigen::MatrixBase<DerivedA>::RowsAtCompileTime,
                  1> invStandardDevs = standardDevs.array().inverse();
    m = invStandardDevs.diagonal() * m * invStandardDevs.diagonal();

    return standardDevs;
}

template <typename DerivedA, typename DerivedB>
Eigen::Matrix<typename Eigen::MatrixBase<DerivedA>::Scalar,
              Eigen::MatrixBase<DerivedA>::RowsAtCompileTime,
              Eigen::MatrixBase<DerivedA>::ColsAtCompileTime>
correlationToCovariance(Eigen::MatrixBase<DerivedA> &corrM,
                        Eigen::MatrixBase<DerivedB> &standardDevs)
{
    assert(corrM.cols() == corrM.rows());
    assert(((standardDevs.cols() == corrM.cols()) || (standardDevs.rows() == corrM.rows())) &&
           (standardDevs.size() == corrM.cols()));

    return standardDevs.diagonal() * corrM * standardDevs.diagonal();
}

// for compile-time sized matrices, it returns constant value.
template <typename DerivedT>
typename std::enable_if<!((Eigen::MatrixBase<DerivedT>::RowsAtCompileTime == Eigen::Dynamic) ||
                        (Eigen::MatrixBase<DerivedT>::ColsAtCompileTime == Eigen::Dynamic)),
                       Eigen::Matrix<typename Eigen::MatrixBase<DerivedT>::Scalar,
                                     Eigen::MatrixBase<DerivedT>::RowsAtCompileTime,
                                     Eigen::MatrixBase<DerivedT>::ColsAtCompileTime>>::type
getConstantOrEmpty(typename Eigen::MatrixBase<DerivedT>::Scalar value)
{
    return Eigen::Matrix<typename Eigen::MatrixBase<DerivedT>::Scalar,
                         Eigen::MatrixBase<DerivedT>::RowsAtCompileTime,
                         Eigen::MatrixBase<DerivedT>::ColsAtCompileTime>::Constant(value);
}

// for dynamically sized matrices, it returns empty matrix
template <typename DerivedT>
typename std::enable_if<((Eigen::MatrixBase<DerivedT>::RowsAtCompileTime == Eigen::Dynamic) ||
                         (Eigen::MatrixBase<DerivedT>::ColsAtCompileTime == Eigen::Dynamic)),
                        Eigen::Matrix<typename Eigen::MatrixBase<DerivedT>::Scalar,
                                      Eigen::MatrixBase<DerivedT>::RowsAtCompileTime,
                                      Eigen::MatrixBase<DerivedT>::ColsAtCompileTime>>::type
getConstantOrEmpty(typename Eigen::MatrixBase<DerivedT>::Scalar /*value*/) // value is ignored.
{
    return {};
}


template <typename T>
matlabEps(T v)
{
   return std::nextafter(std::abs(v), std::abs(v) + 1) - std::abs(v);
}

}

#endif // NCORR_COMMON_H
