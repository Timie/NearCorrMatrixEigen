#ifndef NCORR_HIGHAM_H
#define NCORR_HIGHAM_H

#include <Eigen/Core>
#include <Eigen/Eigenvalues> // for Eigen::SelfAdjointEigenSolver to be used in finding eigenvalues/vectors

#include "ncorr/ncorr_common.h"

namespace ncorr {

    // Helper function
    template <typename DerivedA>
    Eigen::Matrix<typename Eigen::MatrixBase<DerivedA>::Scalar,
                  Eigen::MatrixBase<DerivedA>::RowsAtCompileTime,
                  Eigen::MatrixBase<DerivedA>::ColsAtCompileTime>
    proj_spd(const Eigen::MatrixBase<DerivedA> &A);


    // TODO: Add proj_spd_eigs version (which is in original matlab implementation)
    /**
      * This implementation is based upon Python implementation of the nearest correlation algorithm
      * by Mike Croucher, available at https://github.com/mikecroucher/nearest_correlation,
      * which is in turn based on Matlab implementation by Nick Higham available at
      * http://nickhigham.wordpress.com/2013/02/13/the-nearest-correlation-matrix/.
      *
      * The Python implementation uses BSD-3-Clause licence.
      *
      * Parameters are the following:
      * A - non-empty symmetric square matrix which is "almost" correlation. The matrix should be floating
      * point type (float, double, long double). No checks are done on the contents of the matrix, so it
      * is the caller's responsibility to provide correct matrix.
      *
      * tol - minimum change between successive iterations of the algorithm to keep the iterations running.
      * If not provided, or NaN is provided, it is calculated as the scalar epsilon * dimensionality of the
      * matrix. Only NaN and positive values are allowed.
      *
      * reachedMaxIterations - optional pointer to bool value through which the information on whether the
      * algorithm finished by reaching maximum number of iterations (true) or by achieving desired tolerance
      * (false) is reported. If not provided, or nullptr provided, no reporting is done. If non-nullptr
      * pointer is provided, the pointer must point to valid bool object.
      *
      * weights - column vector of weights of the diagonal elements of the matrix A. It must have the same
      * length as columns or rows of the matrix A. If not provided, uniform weights (= 1), are used.
      *
      *
      */
    template <typename DerivedA,
              typename DerivedW = Eigen::Matrix<typename Eigen::MatrixBase<DerivedA>::Scalar,
                                                Eigen::MatrixBase<DerivedA>::RowsAtCompileTime,
                                                1>>
    Eigen::Matrix<typename Eigen::MatrixBase<DerivedA>::Scalar,
                  Eigen::MatrixBase<DerivedA>::RowsAtCompileTime,
                  Eigen::MatrixBase<DerivedA>::ColsAtCompileTime>
    findNearestCorrelationMatrix_Higham(const Eigen::MatrixBase<DerivedA> &A,
                                        typename Eigen::MatrixBase<DerivedA>::Scalar tol = std::numeric_limits<typename Eigen::MatrixBase<DerivedA>::Scalar>::quiet_NaN(),
                                        const Eigen::Index max_iterations = 100,
                                        bool *reachedMaxIterations = nullptr,
                                        const Eigen::MatrixBase<DerivedW> &weights = getConstantOrEmpty<DerivedW>(1))
    {
        typedef Eigen::Matrix<typename Eigen::MatrixBase<DerivedA>::Scalar,
                Eigen::MatrixBase<DerivedA>::RowsAtCompileTime,
                Eigen::MatrixBase<DerivedA>::ColsAtCompileTime> MatrixType; // type of the result
        typedef Eigen::Matrix<typename Eigen::MatrixBase<DerivedA>::Scalar,
                              Eigen::MatrixBase<DerivedA>::RowsAtCompileTime,
                              1> VectorType;
        typedef typename Eigen::MatrixBase<DerivedA>::Scalar ScalarType;

        ScalarType eps = Eigen::NumTraits<ScalarType>::epsilon();

        // set default tolerance if not provided
        if(std::isnan(tol))
        {
            tol = eps * A.cols();
        }

        // set default weights if dynamic size is provided and does not have correct size
        VectorType effectiveWeights = weights; //
        if(effectiveWeights.rows() != A.rows()) // assuming that they are dynamically sized
        {
            effectiveWeights.resize(A.rows(),
                                    1);
            effectiveWeights.array() = ScalarType(1);
        }

        MatrixType X = A;
        MatrixType Y = A;

        MatrixType ds;
        ds.resize(A.rows(),
                  A.cols()); // we must call resize to make sure the correct size for dynamically sized matrices.
        ds.setZero();

        ScalarType rel_diffY = std::numeric_limits<ScalarType>::infinity();
        ScalarType rel_diffX = std::numeric_limits<ScalarType>::infinity();
        ScalarType rel_diffXY = std::numeric_limits<ScalarType>::infinity();

        MatrixType Whalf = (effectiveWeights * effectiveWeights.transpose()).array().sqrt();

        Eigen::Index iteration = 0;

        while(std::max(rel_diffX, std::max(rel_diffY, rel_diffXY)) > tol)
        {
            ++iteration;
//            std::cout << "Iteration: " << iteration << std::endl;
            if(iteration > max_iterations)
            {
                if(reachedMaxIterations != nullptr)
                {
                    *reachedMaxIterations = true;
                }

                return X;
            }

            MatrixType Xold = X;
//            std::cout << "Xold\n"
//                      << Xold << std::endl;
            MatrixType R = X - ds;
//            std::cout << "R\n"
//                      << R << std::endl;
            MatrixType R_wtd = Whalf.array() * R.array();
//            std::cout << "R_wtd\n"
//                      << R_wtd << std::endl;

            X = proj_spd(R_wtd);

//            std::cout << "X (proj)\n"
//                      << X << std::endl;

            X = X.array() / Whalf.array();
//            std::cout << "X (/WHalf)\n"
//                      << X << std::endl;
            ds = X - R;
//            std::cout << "ds\n"
//                      << ds << std::endl;

            MatrixType Yold = Y;

//            std::cout << "Yold\n"
//                      << Yold << std::endl;
            Y = X;
            Y.diagonal().array() = ScalarType(1);

//            std::cout << "Y\n"
//                      << Y << std::endl;

            ScalarType normY = Y.norm();
            rel_diffX = (X-Xold).norm() / (X.norm());
            rel_diffY = (Y-Yold).norm() / (Y.norm());
            rel_diffXY = (Y - X).norm() / normY;

            X = Y;

        }

        return X;

    }

    template <typename DerivedA>
    Eigen::Matrix<typename Eigen::MatrixBase<DerivedA>::Scalar,
                  Eigen::MatrixBase<DerivedA>::RowsAtCompileTime,
                  Eigen::MatrixBase<DerivedA>::ColsAtCompileTime>
    proj_spd(const Eigen::MatrixBase<DerivedA> &A)
    {
//        std::cout << "A:\n"
//                  << A << std::endl;

        Eigen::Matrix<typename Eigen::MatrixBase<DerivedA>::Scalar,
                      Eigen::MatrixBase<DerivedA>::RowsAtCompileTime,
                      Eigen::MatrixBase<DerivedA>::ColsAtCompileTime> A_result;


        if(A == A.transpose()) // A is symmetric
        {
            Eigen::SelfAdjointEigenSolver<DerivedA> solver(A); // here we assume that A is symmetric
            A_result = (solver.eigenvectors().real().array().rowwise() * solver.eigenvalues().real().array().max(0).transpose()).matrix() * // multiply each vector with its eigen value. If the eigenvalue is negative, use 0 instead.
                        solver.eigenvectors().real().transpose();


//            std::cout << "ScaledEigenVectors:\n"
//                      << (solver.eigenvectors().real().array().rowwise() * solver.eigenvalues().real().array().max(0).transpose()).matrix() << std::endl;

//            std::cout << "solver.eigenvalues()\n"
//                      << solver.eigenvalues()
//                      << "\nsolver.eigenvectors()\n"
//                      << solver.eigenvectors() << std::endl;
        }
        else
        {

            // A is not symmetric
            Eigen::EigenSolver<DerivedA> solver(A);
            A_result = (solver.eigenvectors().real().array().rowwise() * solver.eigenvalues().real().array().max(0).transpose()).matrix() * // multiply each vector with its eigen value. If the eigenvalue is negative, use 0 instead.
                        solver.eigenvectors().real().transpose();

//            std::cout << "ScaledEigenVectors:\n"
//                      << (solver.eigenvectors().real().array().rowwise() * solver.eigenvalues().real().array().max(0).transpose()).matrix() << std::endl;
//            std::cout << "solver.eigenvalues()\n"
//                      << solver.eigenvalues()
//                      << "\nsolver.eigenvectors()\n"
//                      << solver.eigenvectors() << std::endl;
        }



//        std::cout << "A:\n"
//                  << A << std::endl
//                  << A_result << std::endl;

        makeSymmetric(A_result);

        return A_result;
    }
}

#endif // NCORR_HIGHAM_H
