TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt


QMAKE_CXXFLAGS += /bigobj #big template mess causes big objects
QMAKE_CXXFLAGS -= /MP1
QMAKE_CXXFLAGS -= /maxcpucount1
QMAKE_CXXFLAGS += /Zm2000

# Change this path to your instance of the Eigen library:
INCLUDEPATH += "C:/Eigen/SOURCE"


SOURCES += \
    test.cpp

HEADERS += \
    ncorr/ncorr_ldl_gmw.h
