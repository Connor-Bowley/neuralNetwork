cmake_minimum_required(VERSION 2.8)
project (NeuralNetwork_BOINC_app)

add_executable (ConvNetVideoDriverCL_BOINC ConvNetVideoDriverCL_BOINC.cpp)
target_link_libraries (ConvNetVideoDriverCL_BOINC ${OpenCV_LIBS} ${OpenCL_LIBRARIES} ${BOINC_APP_LIBRARIES} ConvNetCL)

add_executable (ConvNetFullImageDriverCL_BOINC ConvNetFullImageDriverCL_BOINC.cpp)
target_link_libraries (ConvNetFullImageDriverCL_BOINC ${OpenCV_LIBS} ${OpenCL_LIBRARIES} ${BOINC_APP_LIBRARIES} ConvNetCL)

add_executable (ConvNetSeamRun_BOINC ConvNetSeamRun_BOINC.cpp)
target_link_libraries (ConvNetSeamRun_BOINC ${OpenCL_LIBRARIES} ${OpenCV_LIBS} ${BOINC_APP_LIBRARIES} ConvNetCL ConvNetSeam ConvNetEvent)
