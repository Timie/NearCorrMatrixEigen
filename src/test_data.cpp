#include <Eigen/Core>

Eigen::Matrix3d getMatrix3DIdentity()
{
    Eigen::Matrix3d m;
    m << 1,0,0,
         0,1,0,
         0,0,1;

    return m;
}
