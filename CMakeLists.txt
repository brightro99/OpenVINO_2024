CMAKE_MINIMUM_REQUIRED (VERSION 3.24)

PROJECT(OpenVINO_2024)
SET(TARGET_NAME Sample1)

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
MESSAGE(STATUS "C++ Compiler Version : ${CMAKE_CXX_COMPILER_VERSION}")

# CXX
IF("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EMSDK_FLAGS} -Wall")
    SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${EMSDK_FLAGS} -Wall -g -O0")
ELSEIF(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EMSDK_FLAGS} -Wall")
    SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${EMSDK_FLAGS} -Wall -g -O0")
ENDIF()

LIST(APPEND CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/third_party)

### OPENVINO
MESSAGE(STATUS ">>> [OpenVINO SETTING]")
SET(OpenVINO_DIR "C:/Program Files (x86)/Intel/openvino_2024.4.0/runtime/cmake")
FIND_PACKAGE(OpenVINO REQUIRED Runtime Threading)
IF(OpenVINO_FOUND)
    MESSAGE(STATUS "    Find OpenVINO ${OpenVINO_VERSION_MAJOR}.${OpenVINO_VERSION_MINOR}.${OpenVINO_VERSION_PATCH}")
    MESSAGE(STATUS "    OpenVINO_DIR : " ${OpenVINO_DIR})
    LIST(APPEND INCLUDE_DIRECTORIES "${OpenVINO_DIR}/../include/openvino")
    MESSAGE(STATUS "        INCLUDE    : " ${INCLUDE_DIRECTORIES})
    LIST(APPEND DEPENDENCIES openvino::runtime openvino::threading)
    MESSAGE(STATUS "        LIBRARIES  : " ${DEPENDENCIES})

    SET(TBB_DIR "${OpenVINO_DIR}/../3rdparty/tbb/lib/cmake/TBB")
    FIND_PACKAGE(TBB)
    IF(TBB_FOUND)
            MESSAGE(STATUS "    TBB_DIR : " ${TBB_DIR})
            LIST(APPEND DEPENDENCIES TBB::tbb)
            MESSAGE(STATUS "        LIBRARIES  : " ${DEPENDENCIES})
    ENDIF()
ENDIF()

### OPENCV
MESSAGE(STATUS ">>> [OpenCV SETTING]")
SET(OpenCV_DIR "${OpenVINO_DIR}/../3rdparty/opencv-4.10.0")
FIND_PACKAGE(OpenCV REQUIRED core imgproc imgcodecs highgui videoio)

IF(OpenCV_FOUND)
    MESSAGE(STATUS "    Find OpenCV ${OpenCV_VERSION}")
    MESSAGE(STATUS "    OpenCV_DIR : " ${OpenCV_DIR})
    LIST(APPEND INCLUDE_DIRECTORIES ${OpenCV_INCLUDE_DIRS})
    MESSAGE(STATUS "        INCLUDE    : " ${OpenCV_INCLUDE_DIRS})
    LIST(APPEND DEPENDENCIES ${OpenCV_LIBRARIES})
    MESSAGE(STATUS "        LIBRARIES  : " ${OpenCV_LIBRARIES})
ELSE()
    MESSAGE(FATAL_ERROR "OPENCV is disabled or not found")
ENDIF()


FILE (GLOB SRCS 
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
)
FILE (GLOB HDRS 
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
)

ADD_EXECUTABLE(${TARGET_NAME} ${SRCS} ${HDRS})

TARGET_INCLUDE_DIRECTORIES(${TARGET_NAME} PRIVATE ${INCLUDE_DIRECTORIES})
TARGET_LINK_LIBRARIES(${TARGET_NAME} PRIVATE ${DEPENDENCIES})

message(STATUS "OUT : ${CMAKE_BINARY_DIR}")
if (WIN32)
    # Debug
    ## OpenVINO
    configure_file("${OpenVINO_DIR}/../bin/intel64/Debug/openvino_auto_batch_plugind.dll"  "${CMAKE_BINARY_DIR}/Debug/openvino_auto_batch_plugind.dll" COPYONLY)
    configure_file("${OpenVINO_DIR}/../bin/intel64/Debug/openvino_auto_plugind.dll"  "${CMAKE_BINARY_DIR}/Debug/openvino_auto_plugind.dll" COPYONLY)
    configure_file("${OpenVINO_DIR}/../bin/intel64/Debug/openvino_cd.dll"  "${CMAKE_BINARY_DIR}/Debug/openvino_cd.dll" COPYONLY)
    configure_file("${OpenVINO_DIR}/../bin/intel64/Debug/openvino_intel_cpu_plugind.dll"  "${CMAKE_BINARY_DIR}/Debug/openvino_intel_cpu_plugind.dll" COPYONLY)
    configure_file("${OpenVINO_DIR}/../bin/intel64/Debug/openvino_intel_gpu_plugind.dll"  "${CMAKE_BINARY_DIR}/Debug/openvino_intel_gpu_plugind.dll" COPYONLY)
    configure_file("${OpenVINO_DIR}/../bin/intel64/Debug/openvino_intel_npu_plugind.dll"  "${CMAKE_BINARY_DIR}/Debug/openvino_intel_npu_plugind.dll" COPYONLY)
    configure_file("${OpenVINO_DIR}/../bin/intel64/Debug/openvinod.dll"  "${CMAKE_BINARY_DIR}/Debug/openvinod.dll" COPYONLY)
    ## TBB
    configure_file("${OpenVINO_DIR}/../3rdparty/tbb/bin/tbb12_debug.dll" "${CMAKE_BINARY_DIR}/Debug/tbb12_debug.dll" COPYONLY)
    configure_file("${OpenVINO_DIR}/../3rdparty/tbb/bin/tbb12.dll" "${CMAKE_BINARY_DIR}/Debug/tbb12.dll" COPYONLY)
    configure_file("${OpenVINO_DIR}/../3rdparty/tbb/bin/tbbbind_2_5_debug.dll" "${CMAKE_BINARY_DIR}/Debug/tbbbind_2_5_debug.dll" COPYONLY)
    configure_file("${OpenVINO_DIR}/../3rdparty/tbb/bin/tbbbind_2_5.dll" "${CMAKE_BINARY_DIR}/Debug/tbbbind_2_5.dll" COPYONLY)
    configure_file("${OpenVINO_DIR}/../3rdparty/tbb/bin/tbbmalloc_debug.dll" "${CMAKE_BINARY_DIR}/Debug/tbbmalloc_debug.dll" COPYONLY)
    configure_file("${OpenVINO_DIR}/../3rdparty/tbb/bin/tbbmalloc_proxy_debug.dll" "${CMAKE_BINARY_DIR}/Debug/tbbmalloc_proxy_debug.dll" COPYONLY)
    configure_file("${OpenVINO_DIR}/../3rdparty/tbb/bin/tbbmalloc_proxy.dll" "${CMAKE_BINARY_DIR}/Debug/tbbmalloc_proxy.dll" COPYONLY)
    configure_file("${OpenVINO_DIR}/../3rdparty/tbb/bin/tbbmalloc.dll" "${CMAKE_BINARY_DIR}/Debug/tbbmalloc.dll" COPYONLY)
    ## OpenCV
    configure_file("${OpenVINO_DIR}/../3rdparty/opencv-4.10.0/x64/vc16/bin/opencv_videoio_ffmpeg4100_64.dll"     "${CMAKE_BINARY_DIR}/Debug/opencv_videoio_ffmpeg4100_64.dll" COPYONLY)
    configure_file("${OpenVINO_DIR}/../3rdparty/opencv-4.10.0/x64/vc16/bin/opencv_videoio_msmf4100_64d.dll"     "${CMAKE_BINARY_DIR}/Debug/opencv_videoio_msmf4100_64d.dll" COPYONLY)
    configure_file("${OpenVINO_DIR}/../3rdparty/opencv-4.10.0/x64/vc16/bin/opencv_world4100d.dll"     "${CMAKE_BINARY_DIR}/Debug/opencv_world4100d.dll" COPYONLY)

# opencv_world4100.dll
# opencv_videoio_msmf4100_64.dll
# opencv_videoio_ffmpeg4100_64.dll
endif()