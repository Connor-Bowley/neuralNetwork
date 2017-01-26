cmake_minimum_required (VERSION 2.8)
project (NeuralNetwork)

if (${CMAKE_BINARY_DIR} EQUAL ${CMAKE_SOURCE_DIR})
    message (FATAL_ERROR "Building in source is not allowed. Create a build
    directory.")
endif()

# all our options
option(BUILD_BOINC "builds the BOINC app" ON)
option(BUILD_DYNAMIC "build dynamic libraries instead of static" OFF)

if (APPLE)
    set(CMAKE_OSX_ARCHITECTURES "x86_64")
    add_definitions("-arch x86_64")
endif ()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules")

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

# check for the CXX11 or CXX0x compiler flags
include (CheckCXXCompilerFlag)
if (WIN32)
    set(COMPILER_SUPPORTS_CXX11 1)
else()
    CHECK_CXX_COMPILER_FLAG(-std=c++11 COMPILER_SUPPORTS_CXX11)
    CHECK_CXX_COMPILER_FLAG(-std=c++0x COMPILER_SUPPORTS_CXX0x)
endif()

# require CXX11
if (COMPILER_SUPPORTS_CXX11)
    if (NOT WIN32)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    endif ()
elseif (COMPILER_SUPPORTS_CXX0x)
    if (NOT WIN32)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    endif ()
else()
    message (FATAL_ERROR "C++11 support required. Please update your compiler.")
endif()

if (BUILD_DYNAMIC)
    set (OpenCV_STATIC OFF)
else ()
    set (OpenCV_STATIC ON)
endif ()

# find all our packages
find_package( OpenCV )
IF (NOT OpenCV_FOUND)
    message (FATAL_ERROR "Unable to find OpenCV.")
ENDIF ()
include_directories(${OpenCV_INCLUDE_DIRS})

find_package( OpenCL )
IF(NOT OpenCL_FOUND)
    message (FATAL_ERROR "Unable to find OpenCL.")
ENDIF ()

include_directories(${OpenCL_INCLUDE_DIRS})
message(“OpenCL include LIBS: ${OpenCL_LIBRARIES})

# add our includes directory
include_directories (${CMAKE_SOURCE_DIR}/include)
if (WIN32)
    include_directories(${CMAKE_SOURCE_DIR}/win_include)
    if (BUILD_DYNAMIC)
        set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
        set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
    else ()
        set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
        set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
    endif ()
endif ()

# compile the main source
add_subdirectory (src)

# compile BOINC
if (${BUILD_BOINC})
    add_subdirectory(boinc)
endif()
