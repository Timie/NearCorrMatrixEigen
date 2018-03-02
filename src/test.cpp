#include <iostream>

#include <Eigen/Core>

#include <ncorr/ncorr_ldl_gmw.h>

using namespace std;

int main()
{
    Eigen::Matrix3d m;
    m << 1,0,0,
         0,1,0,
         0,0,1;

    Eigen::Matrix3d mFixed = ncorr::findNearestCorrelationMatrix_LDL_GMW(m);

    std::cout << "Found nearest corelation matrix:\n"
              << mFixed << std::endl;

    return 0;
}
