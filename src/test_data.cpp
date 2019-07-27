#include <Eigen/Core>

// These are taken from https://github.com/mikecroucher/nearest_correlation

Eigen::Matrix3d getMatrix3DIdentity()
{
    Eigen::Matrix3d m;
    m << 1,0,0,
         0,1,0,
         0,0,1;

    return m;
}


Eigen::Matrix4d getNAGETestData()
{
    Eigen::Matrix4d m;
    m << 2, -1, 0, 0,
         -1, 2, -1, 0,
         0, -1, 2, -1,
         0, 0, -1, 2;
    return m;
}

Eigen::Matrix4d getNAGEExpectedResult()
{
    Eigen::Matrix4d m;

    m << 1.        , -0.8084125 ,  0.1915875 ,  0.10677505,
         -0.8084125 ,  1.        , -0.65623269,  0.1915875 ,
         0.1915875 , -0.65623269,  1.        , -0.8084125 ,
         0.10677505,  0.1915875 , -0.8084125 ,  1.0;

    return m;
}


Eigen::Matrix3d getHighamExample2002TestData()
{
    Eigen::Matrix3d m;
    m << 1,1,0,
         1,1,1,
         0,1,1;

    return m;
}


Eigen::Matrix3d getHighamExample2002ExpectedResult()
{
    Eigen::Matrix3d m;
    m << 1.        ,  0.76068985,  0.15729811,
         0.76068985,  1.        ,  0.76068985,
         0.15729811,  0.76068985,  1.        ;

    return m;

}


