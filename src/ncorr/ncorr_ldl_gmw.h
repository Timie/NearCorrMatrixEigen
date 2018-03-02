#ifndef NCORR_LDL_GMW_H
#define NCORR_LDL_GMW_H

#include <Eigen/Core>


namespace ncorr
{
    /**
     * This method implements algorithm for finding nearest correlation matrix based on modified
     * LDL Cholesky decomposition for non (positive definite) matrices. The implementation is a
     * rewrite of implmentation by Brian Borchers (Dept. of Mathematics, New Mexico Tech), and
     * Michael Zibulevksy, originally written in 1994 - 2002 for Matlab, and is based upon book
     * "Practical Optimization" by Gill, Murray and Wright, p. 111.
     *
     * In case of non-square input matrix, the method throws exception in debug mode. In
     * release mode, it does not check and assumes squareness (thus, undef behaviour).
     * In case of empty matrix, an empty matrix is returned. In case of algorithm failure
     * (e.g., the optimisation fails), it should (not guaranteed) return NaN matrix elements.
     */
    template <typename DerivedA>
    Eigen::Matrix<typename Eigen::MatrixBase<DerivedA>::Scalar,
                  Eigen::MatrixBase<DerivedA>::RowsAtCompileTime,
                  Eigen::MatrixBase<DerivedA>::ColsAtCompileTime>
    findNearestCorrelationMatrix_LDL_GMW(const Eigen::MatrixBase<DerivedA> &m);



    // *** IMPLEMENTATION **************************************************************




    /** Given a symmetric matrix G, find a matrix E of "small" norm and c
      *  L, and D such that  G+E is Positive Definite, and
      *
      *      G+E = L*D*L'
      *
      *  Also, calculate a direction pneg, such that if G is not PD, then
      *
      *     pneg'*G*pneg < 0
      *
      * Note that if G is PD, then the routine will return pneg=NaNs.
      *
      * The method returns E. The method calculates and returns L, D,
      * and pneg, via its parameter pointers (if provided). The method does not do any checks
      * on its input arguments, and it is responsibility of the caller to provide valid
      * pointers and correct sizes of the matrices. Otherwise, you may face undef behaviour.
      *
      * G must be square. L and D must be square and the same size as G, and must reference
      * separated data, pneg must be 1-dimensional column vector with size of G. You are
      * advised to use double precision
      * matrices.
     */
    template <typename DerivedG,
              typename DerivedL,
              typename DerivedD,
              typename DerivedP>
    Eigen::Matrix<typename Eigen::MatrixBase<DerivedG>::Scalar,
                  Eigen::MatrixBase<DerivedG>::RowsAtCompileTime,
                  Eigen::MatrixBase<DerivedG>::ColsAtCompileTime>
    mcholmz1(const Eigen::MatrixBase<DerivedG> &G,
             Eigen::MatrixBase<DerivedL> &L,
             Eigen::MatrixBase<DerivedD> &D,
             Eigen::MatrixBase<DerivedP> *pneg = nullptr)
    {
        typedef Eigen::MatrixBase<DerivedG>::Scalar GScalar;
        typedef Eigen::Matrix<typename Eigen::MatrixBase<DerivedG>::Scalar,
                              Eigen::MatrixBase<DerivedG>::RowsAtCompileTime,
                              Eigen::MatrixBase<DerivedG>::ColsAtCompileTime> EType;

        typedef Eigen::Matrix<typename Eigen::MatrixBase<DerivedG>::Scalar,
                              Eigen::MatrixBase<DerivedG>::RowsAtCompileTime,
                              1> EVecType;

        const Eigen::Index n = G.cols(); // num of dimensions
        GScalar gamma = G.diagonal().maxCoeff(); // maximum element at diagonal
        GScalar zi = (G - G.diagonal().asDiagonal()).maxCoeff(); // maximum element of the diagonal (max element of G with zeroed diagonal).
        GScalar nu = std::max<GScalar>(1, std::sqrt(n*n - 1));
        GScalar beta2 = std::max<GScalar>(std::max<GScalar>(gamma,
                                                            zi / nu),
                                          Eigen::NumTraits<GScalar>::dummy_precision());

        EType C = G.diagonal().asDiagonal(); // TODO: Optimise this away - the same thing was used before.
        EType E;
        E.resize(G.rows(), G.cols());
        E.setZero();
        L.setZero();
        D.setZero();

        for(Eigen::Index j = 0;
            j < n;
            ++j)
        {
            // calculate j-th row of L
            if(j > 0)
            {
//                for(Eigen::Index bb = 0;
//                    bb < j;
//                    ++bb)
//                {
//                    L(j, bb) = C(j,bb) / D(bb, bb);
//                }
                L.row(j).colRange(0,j) = C.row(j).colRange(0,j).asArray() / D.block(0,0, j,j).diagonal().asArray();
            }

            // update j-th column of C
            if(j >= 1)
            {
                if((j+1) < n)
                {
                    // C(ee, j)               = G(ee,j)                   -
                    C.col(j).rowRange(j+1, n) = G.col(j).rowRange(j+1, n) -
                                            //   (L(j,     bb)              C(ee,bb)                         '           )'
                                                ((L.row(j).colRange(0,j) * (C.rowRange(j+1, n).colRange(0,j).transpose())).transpose());
                }
            }
            else
            {
                if((j+1) < n)
                {
//                    for(Eigen::Index ee = j + 1;
//                        ee < n;
//                        ++ee)
//                    {
//                        C(ee, j) = G(ee,j);
//                    }
                    C.col(j).rowRange(j+1, n) = G.col(j).rowRange(j+1, n);
                }
            }

            // calculate theta
            GScalar theta;
            if((j+1) == n)
            {
                theta = 0;
            }
            else
            {
                theta = C.col(j).rowRange(j+1, n).cwiseAbs().maxCoeff();
            }

            // update D
            D(j,j) = std::max<GScalar>(Eigen::NumTraits<GScalar>::epsilon(),
                                       std::max<GScalar>(std::abs(C(j,j)),
                                                         theta * theta / beta2));

            // update E
            E(j,j) = D(j,j) - C(j,j);

            // update C again
            for(Eigen::Index i = j+1;
                i < n;
                ++i)
            {
                C(i,i) = C(i,i) - C(i,j) * C(i,j) / D(j,j);
            }
        }

        // Put 1's on the diagonal of L
        L.diagonal().array() = GScalar(1);


        // if needed, find a descent direction
        if(pneg != nullptr)
        {
            Eigen::Index col = 0; // minDiagIdx
            GScalar m = C(0,0); //minDiagValue
            for(Eigen::Index i = 1;
                i < n;
                ++i)
            {
                if(m > C(i,i))
                {
                    m = C(i,i);
                    col = i;
                }
            }

            if(m < GScalar(0))
            {
                *pneg = std::numeric_limits<GScalar>::quiet_NaN();
            }
            else
            {
                EVecType rhs;
                rhs.resize(n,1);
                rhs.setZero;
                rhs(col) = GScalar(1);
                *pneg = L.colPivHouseholderQr().solve(rhs); // here, using colPivHouseholderQr instead of generic matlab's solve.
            }
        }

        return E;

    }


    template <typename DerivedA>
    Eigen::Matrix<typename Eigen::MatrixBase<DerivedA>::Scalar,
                  Eigen::MatrixBase<DerivedA>::RowsAtCompileTime,
                  Eigen::MatrixBase<DerivedA>::ColsAtCompileTime>
    findNearestCorrelationMatrix_LDL_GMW(const Eigen::MatrixBase<DerivedA> &m)
    {
        assert(m.cols() == m.rows()); // input matrix must be squared


        typedef Eigen::Matrix<typename Eigen::MatrixBase<DerivedA>::Scalar,
                              Eigen::MatrixBase<DerivedA>::RowsAtCompileTime,
                              Eigen::MatrixBase<DerivedA>::ColsAtCompileTime> EType;

        if(!m.allFinite())
        {
            return EType::Constant(std::numeric_limits<typename Eigen::MatrixBase<DerivedA>::Scalar>::quiet_NaN());
        }

        if((m.cols() == 0) ||
           (m.rows() == 0))
        {
            return {}; // empty matrix
        }




        EType L, D; // temporaries - used inside of mcholmz1.

        auto E = mcholmz1(m,
                          L,
                          D);
        return m + E;
    }



}


#endif // NCORR_LDL_GMW_H
