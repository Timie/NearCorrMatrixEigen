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
bool checkHighamCovToCorr_T(const std::string &inputName,
           const Eigen::MatrixBase<DerivedA> &testInput,
                           const Eigen::MatrixBase<DerivedB> &expectedResult)
{
    std::cout << inputName << " checkHighamCovToCorr_T: Input:\n"
              << testInput
              << "\nExpected Result:\n"
              << expectedResult
              << std::endl;

    Eigen::MatrixXd testInputX = testInput; // to test dynamically sized arrays

    // Convert to correlation matrix, then find nearest correlation matrix, and then convert back to covariance.

    auto correlationMat = testInput;
    auto standardDevsVec = ncorr::covarianceToCorrelation(correlationMat);

    auto correlationMatX = testInputX;
    auto standardDevsVecX = ncorr::covarianceToCorrelation(correlationMatX);

    auto resHigham = ncorr::findNearestCorrelationMatrix_Higham(correlationMat, 0);
    resHigham = ncorr::correlationToCovariance(resHigham, standardDevsVec);
    auto resHighamX = ncorr::findNearestCorrelationMatrix_Higham(correlationMatX, 0);
    resHighamX = ncorr::correlationToCovariance(resHighamX, standardDevsVecX);
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

    return resHighamVerdict &&
           resHighamXCorrect;
}


bool checkHighamCovToCorr(const std::string &inputName,
           const Eigen::MatrixXd &testInput,
           const Eigen::MatrixXd &expectedResult)
{
    return checkHighamCovToCorr_T(inputName, testInput, expectedResult);
}

template<typename DerivedA,
         typename DerivedB>
bool checkLDLCovariance_T(const std::string &inputName,
           const Eigen::MatrixBase<DerivedA> &testInput,
                           const Eigen::MatrixBase<DerivedB> &expectedResult)
{
    std::cout << inputName << " checkLDLCovariance_T: Input:\n"
              << testInput
              << "\nExpected Result:\n"
              << expectedResult
              << std::endl;

    Eigen::MatrixXd testInputX = testInput; // to test dynamically sized arrays

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

    return resLDLXCorrect && resLDLVerdict;
}

template<typename DerivedA>
bool checkLDLCovariance_T(const std::string &inputName,
           const Eigen::MatrixBase<DerivedA> &testInput)
{

    std::cout << inputName << " checkLDLCovariance_T: Input:\n"
              << testInput
              << "\nNo Expected Result."
              << std::endl;

    Eigen::MatrixXd testInputX = testInput; // to test dynamically sized arrays

    auto resLDL = ncorr::findNearestCovarianceMatrix_LDL_GMW(testInput);
    auto resLDLX = ncorr::findNearestCovarianceMatrix_LDL_GMW(testInputX);
    bool resLDLVerdict = ncorr::isPositiveSemiDefinite(resLDL);
    bool resLDLXCorrect = !(resLDL.array() != resLDLX.array()).any();
    std::cout << "Algorithm: LDL GMW. Result:\n"
              << resLDL
              << "\nResult (dynamic):\n"
              << resLDLX
              << "\nResulution:"
              << getPassedString(resLDLVerdict)
              << "\nDynamic Equal:"
              << getCorrectString(resLDLXCorrect)
              << std::endl;

    return resLDLXCorrect && resLDLVerdict;
}

bool checkLDLCovariance(const std::string &inputName,
           const Eigen::MatrixXd &testInput,
           const Eigen::MatrixXd &expectedResult)
{
    return checkLDLCovariance_T(inputName, testInput, expectedResult);
}


bool checkLDLCovariance(const std::string &inputName,
           const Eigen::MatrixXd &testInput)
{
    return checkLDLCovariance_T(inputName, testInput);
}

template<typename DerivedA>
bool checkHighamCovariance_T(const std::string &inputName,
           const Eigen::MatrixBase<DerivedA> &testInput)
{

    std::cout << inputName << " checkHighamCovariance_T: Input:\n"
              << testInput
              << "\nNo Expected Result."
              << std::endl;

    Eigen::MatrixXd testInputX = testInput; // to test dynamically sized arrays

    auto resLDL = ncorr::findNearestCovarianceMatrix_Higham(testInput);
    auto resLDLX = ncorr::findNearestCovarianceMatrix_Higham(testInputX);
    bool resLDLVerdict = ncorr::isPositiveSemiDefinite(resLDL);
    bool resLDLXCorrect = !(resLDL.array() != resLDLX.array()).any();
    std::cout << "Algorithm: Higham Covariance. Result:\n"
              << resLDL
              << "\nResult (dynamic):\n"
              << resLDLX
              << "\nResulution:"
              << getPassedString(resLDLVerdict)
              << "\nDynamic Equal:"
              << getCorrectString(resLDLXCorrect)
              << std::endl;

    return resLDLXCorrect && resLDLVerdict;
}

bool checkHighamCovariance(const std::string &inputName,
           const Eigen::MatrixXd &testInput)
{
    return checkHighamCovariance_T(inputName, testInput);
}


template<typename DerivedA,
         typename DerivedB>
bool checkHighamCorrelation_T(const std::string &inputName,
           const Eigen::MatrixBase<DerivedA> &testInput,
                           const Eigen::MatrixBase<DerivedB> &expectedResult)
{
    std::cout << inputName << " checkHighamCorrelation_T: Input:\n"
              << testInput
              << "\nExpected Result:\n"
              << expectedResult
              << std::endl;

    Eigen::MatrixXd testInputX = testInput; // to test dynamically sized arrays

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

    return resHighamVerdict &&
           resHighamXCorrect;
}


bool checkHighamCorrelation(const std::string &inputName,
           const Eigen::MatrixXd &testInput,
           const Eigen::MatrixXd &expectedResult)
{
    return checkHighamCorrelation_T(inputName, testInput, expectedResult);
}

int main(int argc, const char *argv[])
{
    bool allOK =
            checkHighamCorrelation("Identity (Higham Correlation)",
                                   getMatrix3DIdentity(),
                                   getMatrix3DIdentity()) &&
            checkHighamCorrelation("Higham (Higham Correlation)",
                                   getHighamExample2002TestData(),
                                   getHighamExample2002ExpectedResult()) &&
            checkHighamCorrelation("NAGE (Higham Correlation)",
                                   getNAGETestData(),
                                   getNAGEExpectedResult()) &&
            checkLDLCovariance("Identity (LDL Covariance)",
                               getMatrix3DIdentity(),
                               getMatrix3DIdentity()) &&
            checkLDLCovariance("Higham (LDL Covariance)",
                               getHighamExample2002TestData()) &&
            checkLDLCovariance("NAGE (LDL Covariance)",
                               getNAGETestData()) &&
            checkHighamCovariance("Identity (Higham Covariance)",
                               getMatrix3DIdentity()) &&
            checkHighamCovariance("Higham (Higham Covariance)",
                               getHighamExample2002TestData()) &&
            checkHighamCovariance("NAGE (Higham Covariance)",
                               getNAGETestData());
    // TODO: Test LDL properly!!!


    if(allOK)
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
