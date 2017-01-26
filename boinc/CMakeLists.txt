cmake_minimum_required(VERSION 2.8)
project (NeuralNetwork_BOINC)

#set variables
set (BOINC_BUILD_DYNAMIC ${BUILD_DYNAMIC})
set (BOINC_DIR $ENV{BOINC_SOURCE} ${CMAKE_SOURCE_DIR}/externals/boinc)

#find packages and libraries
find_package (BOINC)
if (BOINC_APP_FOUND OR BOINC_SERVER_FOUND)
	add_definitions(-D_BOINC_)
    include_directories(${BOINC_INCLUDE_DIRS})
endif ()

#set include directories
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/include )

if (BOINC_APP_FOUND)
    add_subdirectory (app)
endif()

if (BOINC_SERVER_FOUND)
    add_subdirectory (server)
endif()
