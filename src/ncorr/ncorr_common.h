#ifndef NCORR_COMMON_H
#define NCORR_COMMON_H

#include <Eigen/Core>

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


#endif // NCORR_COMMON_H
