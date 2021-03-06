cmake_minimum_required (VERSION 2.8)
project (NeuralNetwork_BASIC)

# make the ConvNetCL library
if (BUILD_DYNAMIC) 
    add_library (ConvNetCL SHARED ConvNetCommon.cpp ConvNetCL.cpp)
    #add_library (ConvNet SHARED ConvNet.cpp ConvNetCommon.cpp)
    add_library (ConvNetEvent SHARED ConvNetEvent.cpp ConvNetCommon.cpp)
    add_library (ConvNetSeam SHARED ConvNetSeam.cpp ConvNetCommon.cpp)
    add_library (Seamcarver SHARED Seamcarver.cpp)
    add_library (IDX_lib SHARED IDX.cpp)
    add_library (MSILocations SHARED MSILocations.cpp ConvNetCommon.cpp)
else ()
    add_library (ConvNetCL STATIC ConvNetCommon.cpp ConvNetCL.cpp)
    #add_library (ConvNet STATIC ConvNet.cpp ConvNetCommon.cpp)
    add_library (ConvNetEvent STATIC ConvNetEvent.cpp ConvNetCommon.cpp)
    add_library (ConvNetSeam STATIC ConvNetSeam.cpp ConvNetCommon.cpp)
    add_library (Seamcarver STATIC Seamcarver.cpp)
    add_library (IDX_lib STATIC IDX.cpp)
    add_library (MSILocations STATIC MSILocations.cpp ConvNetCommon.cpp)
endif ()

######
###### MAKE EXECUTABLES
######

add_executable (BlobCounter BlobCounter.cpp)
target_link_libraries(BlobCounter ${OpenCV_LIBS})

add_executable (CombineIDX_Thesis CombineIDX_Thesis.cpp)
target_link_libraries(CombineIDX_Thesis ${OpenCV_LIBS} IDX_lib)

add_executable (ConvNetTrainerCL_idx ConvNetTrainerCL_idx.cpp)
target_link_libraries(ConvNetTrainerCL_idx ${OpenCL_LIBRARIES} ConvNetCL ${OpenCV_LIBS})

# TODO: pthread
add_executable (ConvNetFullImageDriverParallelCL ConvNetFullImageDriverParallelCL.cpp)
target_link_libraries(ConvNetFullImageDriverParallelCL ${OpenCV_LIBS} ${OpenCL_LIBRARIES} ConvNetCL)

# TODO: pthread
#add_executable (ConvNetVideoDriverParallelCL ConvNetVideoDriverParallelCL.cpp)
#target_link_libraries(ConvNetVideoDriverParallelCL ${OpenCV_LIBS} ${OpenCL_LIBRARIES} ConvNetCL)

# TODO: pthread
#add_executable (ConvNetVideoDriverParallelCL_multiNet ConvNetVideoDriverParallelCL_multiNet.cpp)
#target_link_libraries(ConvNetVideoDriverParallelCL_multiNet ${OpenCV_LIBS} ${OpenCL_LIBRARIES} ConvNetCL)

add_executable(CNNtoObserver_comparator CNNtoObserver_comparator.cpp)
target_link_libraries(CNNtoObserver_comparator ConvNetEvent ${OpenCV_LIBS} MSILocations)

add_executable(DataVisualizer DataVisualizer.cpp)
target_link_libraries(DataVisualizer IDX_lib ${OpenCV_LIBS})

add_executable (kernelBuilder kernelBuilder.cpp)
target_link_libraries(kernelBuilder ${OpenCL_LIBRARIES})

# add_executable (MNIST_test MNIST_test.cpp)
# target_link_libraries (MNIST_test ConvNetCL ${OpenCL_LIBRARIES} ${OpenCV_LIBS})

add_executable (Overlay Overlay.cpp)
target_link_libraries(Overlay ConvNetCL ${OpenCL_LIBRARIES} ${OpenCV_LIBS})

add_executable(SeeUserObs SeeUserObs.cpp)
target_link_libraries(SeeUserObs ${OpenCV_LIBS} ConvNetCL MSILocations)

add_executable(SmoteIDX SmoteIDX.cpp)
target_link_libraries(SmoteIDX ${OpenCV_LIBS})

IF(MYSQL_FOUND)
	add_executable(ConvNetSeam_ToIDX ConvNetSeam_ToIDX.cpp)
	target_link_libraries(ConvNetSeam_ToIDX ${OpenCV_LIBS} ${OpenCL_LIBRARIES} ConvNetEvent Seamcarver ${MYSQL_LIBRARIES})
ELSE()
	MESSAGE("Not building ConvNetSeam_ToIDX due to MySQL not found")
ENDIF()
