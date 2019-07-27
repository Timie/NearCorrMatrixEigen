TEMPLATE = app
CONFIG += c++11
CONFIG -= app_bundle console
CONFIG -= qt


#QMAKE_CXXFLAGS += /bigobj #big template mess causes big objects
#QMAKE_CXXFLAGS -= /MP1
#QMAKE_CXXFLAGS -= /maxcpucount1
#QMAKE_CXXFLAGS += /Zm2000

# QMAKE_CXXFLAGS += -Wa,-mbig-obj

# Change this path to your instance of the Eigen library:
INCLUDEPATH += "C:/Eigen/SOURCE"


SOURCES += \
    test.cpp \
    test_data.cpp

HEADERS += \
    ncorr/ncorr_ldl_gmw.h \
    ncorr/ncorr_higham.h \
    ncorr/ncorr_common.h
