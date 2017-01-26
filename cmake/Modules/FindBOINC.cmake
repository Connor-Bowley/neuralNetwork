# - Find BOINC 
# Find the native BOINC includes and libraries
#
#  BOINC_INCLUDE_DIR        - where to find boinc.h, etc.
#  BOINC_SERVER_FOUND       - true if libraries required for compiling boinc server code are found
#  BOINC_SERVER_LIBRARIES   - all the libraries required for compiling boinc server code
#  BOINC_APP_FOUND          - true if libraries required for compiling boinc apps are found 
#  BOINC_APP_LIBRARIES      - all the libraries required for compiling boinc apps

if (BOINC_INCLUDE_DIRS)
    # Already in cache, be silent
    set (BOINC_FIND_QUIETLY TRUE)
endif ()

FIND_PATH (BOINC_INCLUDE_DIR boinc/boinc_api.h api/boinc_api.h
    PATHS ${BOINC_DIR}
)

if (BOINC_INCLUDE_DIR)
    if (EXISTS "${BOINC_INCLUDE_DIR}/api/boinc_api.h")
        set (BOINC_INCLUDE_DIRS ${BOINC_INCLUDE_DIR} ${BOINC_INCLUDE_DIR}/api ${BOINC_INCLUDE_DIR}/lib)
    else ()
        set (BOINC_INCLUDE_DIRS ${BOINC_INCLUDE_DIR}/boinc)
    endif ()
else ()
    message (FATAL_ERROR "BOINC not found.")
endif ()

message (STATUS "BOINC include directories: ${BOINC_INCLUDE_DIRS}")

if (BOINC_BUILD_DYNAMIC)
    set (BOINC_API_NAME boinc_api boincapi boincapi_staticcrt)
    set (BOINC_CRYPT_NAME boinc_crypt boinccrypt boinccrypt_staticcrt)
    set (BOINC_LIB_NAME boinc boinc_staticcrt)
    set (BOINC_OPENCL_NAME boinc_opencl boincopencl boincopencl_staticcrt)
    set (BOINC_SCHED_NAME boinc_sched boincsched boincsched_staticcrt)
else ()
    if (WIN32)
        set (BOINC_API_NAME "libboincapi_staticcrt.lib")
        set (BOINC_CRYPT_NAME "libboinccrypt_staticcrt.lib")
        set (BOINC_LIB_NAME "libboinc_staticcrt.lib")
        set (BOINC_OPENCL_NAME "libboincopencl_staticcrt.lib")
        set (BOINC_SCHED_NAME "libboincsched_staticcrt.lib")
    else ()
        set (BOINC_API_NAME "libboinc_api.a")
        set (BOINC_CRYPT_NAME "libboinc_crypt.a")
        set (BOINC_LIB_NAME "libboinc.a")
        set (BOINC_OPENCL_NAME "libboinc_opencl.a")
        set (BOINC_SCHED_NAME "libboinc_sched.a")
    endif ()
endif ()

if (WIN32)
    if (CMAKE_CL_64)
        set (BOINC_DEBUG_DIR ${BOINC_INCLUDE_DIR}/win_build/Build/x64/Debug)
        set (BOINC_RELEASE_DIR ${BOINC_INCLUDE_DIR}/win_build/Build/x64/Release) 
    else ()
        set (BOINC_DEBUG_DIR ${BOINC_INCLUDE_DIR}/win_build/Build/Win32/Debug)
        set (BOINC_RELEASE_DIR ${BOINC_INCLUDE_DIR}/win_build/Build/Win32/Release)
    endif ()
elseif (APPLE)
    set (BOINC_RELEASE_DIR ${BOINC_INCLUDE_DIR}/mac_build/build/Deployment)
    set (BOINC_DEBUG_DIR ${BOINC_INCLUDE_DIR}/mac_build/build/Deployment)
else ()
    set (BOINC_RELEASE_DIR )
    set (BOINC_DEBUG_DIR )
endif()

find_library (BOINC_LIBRARY ${BOINC_LIB_NAME}
    PATHS
        ${BOINC_INCLUDE_DIR}
        ${BOINC_INCLUDE_DIR}/../lib
        ${BOINC_RELEASE_DIR}
    PATH_SUFFIXES lib
)
MESSAGE(STATUS "BOINC library: ${BOINC_LIBRARY}")

find_library (BOINC_API_LIBRARY ${BOINC_API_NAME}
    PATHS
        ${BOINC_INCLUDE_DIR}
        ${BOINC_INCLUDE_DIR}/../lib
        ${BOINC_RELEASE_DIR}
    PATH_SUFFIXES api
)
MESSAGE(STATUS "BOINC api library: ${BOINC_API_LIBRARY}")

FIND_LIBRARY(BOINC_CRYPT_LIBRARY ${BOINC_CRYPT_NAME}
    PATHS
        ${BOINC_INCLUDE_DIR}
        ${BOINC_INCLUDE_DIR}/../lib
        ${BOINC_RELEASE_DIR}
    PATH_SUFFIXES lib
)
MESSAGE(STATUS "BOINC crypt library: ${BOINC_CRYPT_LIBRARY}")

FIND_LIBRARY(BOINC_OPENCL_LIBRARY ${BOINC_OPENCL_NAME}
    PATHS
        ${BOINC_INCLUDE_DIR}
        ${BOINC_INCLUDE_DIR}/../lib
        ${BOINC_RELEASE_DIR}
    PATH_SUFFIXES lib
)
MESSAGE(STATUS "BOINC opencl library: ${BOINC_OPENCL_LIBRARY}")

FIND_LIBRARY(BOINC_SCHED_LIBRARY ${BOINC_SCHED_LIBRARY}
    PATHS
        ${BOINC_INCLUDE_DIR}
        ${BOINC_INCLUDE_DIR}/../lib
        ${BOINC_RELEASE_DIR}
    PATH_SUFFIXES sched
)
MESSAGE(STATUS "BOINC sched library: ${BOINC_SCHED_LIBRARY}")
if (BOINC_CRYPT_LIBRARY)
    set (BOINC_INCLUDE_DIRS ${BOINC_INCLUDE_DIRS} ${BOINC_INCLUDE_DIR}/sched)
endif ()

if (BOINC_INCLUDE_DIRS AND BOINC_LIBRARY AND BOINC_API_LIBRARY AND BOINC_OPENCL_LIBRARY)
    add_definitions( -D_BOINC_APP_ )
    set (BOINC_APP_FOUND TRUE)
    set (BOINC_APP_LIBRARIES ${BOINC_API_LIBRARY} ${BOINC_LIBRARY} ${BOINC_OPENCL_LIBRARY})

    message (STATUS "Found BOINC_APP_LIBRARIES: ${BOINC_APP_LIBRARIES}")
    message (STATUS "BOINC include directory: ${BOINC_INCLUDE_DIR}")
else ()
    SET (BOINC_APP_FOUND FALSE)
    SET (BOINC_APP_LIBRARIES )
endif ()

if (BOINC_INCLUDE_DIRS AND BOINC_LIBRARY AND BOINC_API_LIBRARY AND BOINC_SCHED_LIBRARY AND BOINC_CRYPT_LIBRARY)
    add_definitions( -D_BOINC_SERVER_ )
    set (BOINC_SERVER_FOUND TRUE)
    set (BOINC_SERVER_LIBRARIES ${BOINC_SCHED_LIBRARY} ${BOINC_LIBRARY} ${BOINC_API_LIBRARY} ${BOINC_CRYPT_LIBRARY})

    message (STATUS "Found BOINC_SERVER_LIBRARIES: ${BOINC_SERVER_LIBRARIES}")
    message (STATUS "BOINC include directory: ${BOINC_INCLUDE_DIR}")
else ()
    set (BOINC_FOUND FALSE)
    set ( BOINC_LIBRARIES )
endif ()

MARK_AS_ADVANCED(
    BOINC_APP_LIBRARIES
    BOINC_SERVER_LIBRARIES
    BOINC_INCLUDE_DIRS
)
