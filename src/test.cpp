#include <iostream>

#include <Eigen/Core>

#include <ncorr/ncorr_ldl_gmw.h>
#include <ncorr/ncorr_higham.h>

// fwd declaration from test_data (to shaddow from compile time optimisation)

Eigen::Matrix3d getMatrix3DIdentity();

using namespace std;

int main()
{
    // TODO: Implement something like "unit tests".

    Eigen::Matrix3d m = getMatrix3DIdentity();

    Eigen::Matrix3d mFixed = ncorr::findNearestCorrelationMatrix_LDL_GMW(m);
    Eigen::Matrix3d mFixed2 = ncorr::findNearestCorrelationMatrix_Higham(m);

    std::cout << "Found nearest corelation matrix:\n"
              << mFixed << std::endl;
    std::cout << "Found nearest corelation matrix2:\n"
              << mFixed2 << std::endl;

    return 0;
}
