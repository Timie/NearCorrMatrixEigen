#include <iostream>

#include <Eigen/Core>

#include <ncorr/ncorr_ldl_gmw.h>
#include <ncorr/ncorr_higham.h>

// fwd declaration from test_data (to shaddow from compile time optimisation)
Eigen::Matrix3d getMatrix3DIdentity();
Eigen::Matrix4d getNAGETestData();
Eigen::Matrix4d getNAGEExpectedResult();
Eigen::Matrix3d getHighamExample2002TestData();
Eigen::Matrix3d getHighamExample2002ExpectedResult();


template<typename DerivedA,
         typename DerivedB>
bool
isInTolerance_T(const Eigen::MatrixBase<DerivedA> &A,
                const Eigen::MatrixBase<DerivedB> &B)
{
    return ((A.array() - B.array()).abs().maxCoeff() <= std::sqrt(Eigen::NumTraits<typename DerivedA::Scalar>::dummy_precision()));

}

bool
isInTolerance(const Eigen::MatrixXd &A,
                const Eigen::MatrixXd &B)
{
    return isInTolerance_T(A, B);
}

std::string getPassedString(bool value)
{
    if(value)
    {
        return "passed";
    }
    else
    {
        return "FAILED!!!";
    }
}

std::string getCorrectString(bool value)
{
    if(value)
    {
        return "OK";
    }
    else
    {
        return "INCORRECT!!!";
    }
}

template<typename DerivedA,
         typename DerivedB>
bool check_T(const std::string &inputName,
           const Eigen::MatrixBase<DerivedA> &testInput,
                           const Eigen::MatrixBase<DerivedB> &expectedResult)
{
    Eigen::MatrixXd testInputX = testInput; // to test dynamically sized arrays

    std::cout << inputName << " check: Input:\n"
              << testInput
              << "\nExpected Result:\n"
              << expectedResult
              << std::endl;

    auto resLDL = ncorr::findNearestCovarianceMatrix_LDL_GMW(testInput);
    auto resLDLX = ncorr::findNearestCovarianceMatrix_LDL_GMW(testInputX);
    bool resLDLVerdict = isInTolerance(resLDL,
                                       expectedResult);
    bool resLDLXCorrect = !(resLDL.array() != resLDLX.array()).any();
    std::cout << "Algorithm: LDL GMW. Result:\n"
              << resLDL
              << "\nResult (dynamic):\n"
              << resLDLX
              << "\nDiff:"
              << (resLDL - expectedResult).norm()
              << "\nResulution:"
              << getPassedString(resLDLVerdict)
              << "\nDynamic Equal:"
              << getCorrectString(resLDLXCorrect)
              << std::endl;

    auto resHigham = ncorr::findNearestCorrelationMatrix_Higham(testInput, 0);
    auto resHighamX = ncorr::findNearestCorrelationMatrix_Higham(testInputX, 0);
    bool resHighamVerdict = isInTolerance(resHigham,
                                          expectedResult);
    bool resHighamXCorrect = !(resHigham.array() != resHighamX.array()).any();
    std::cout << "Algorithm: Higham. Result:\n"
              << resHigham
              << "\nResult (dynamic):\n"
              << resHighamX
              << "\nDiff: "
              << (resHigham - expectedResult).norm()
              << "\nResulution: "
              << getPassedString(resHighamVerdict)
              << "\nDynamic Equal: "
              << getCorrectString(resHighamXCorrect)
              << std::endl;

    return resLDLVerdict &&
           resLDLXCorrect &&
           resHighamVerdict &&
           resHighamXCorrect;
}


bool check(const std::string &inputName,
           const Eigen::MatrixXd &testInput,
           const Eigen::MatrixXd &expectedResult)
{
    return check_T(inputName, testInput, expectedResult);
}

bool checkIdentity()
{
    return check("Identity",
                 getMatrix3DIdentity(),
                 getMatrix3DIdentity());
}

bool checkNAGE()
{
    return check("NAGE",
                 getNAGETestData(),
                 getNAGEExpectedResult());
}

bool checkHighamExample()
{
    return check("Higham Example",
                 getHighamExample2002TestData(),
                 getHighamExample2002ExpectedResult());
}


int main(int argc, const char *argv[])
{

    bool identityOK = checkIdentity();
    bool nageOK =     checkNAGE();
    bool highamOK =   checkHighamExample();

    if(identityOK && nageOK && highamOK)
    {
        std::cout << "All passed... :-)" << std::endl;
        return 0;
    }
    else
    {
        std::cerr << "SOME OF THE TESTS DID NOT FINISHED SUCCESSFULLY!!!" << std::endl;
        return -1;
    }
}
