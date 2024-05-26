cmake_minimum_required(VERSION 3.27)
project(echovisionRktoolkit2)

include(CMakePrintHelpers)

set(CMAKE_CXX_STANDARD 14)


find_package(OpenCV REQUIRED)

#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -fexceptions")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

set(EXTRA_LIBS ${OpenCV_LIBS} /usr/local/lib/librknnrt.so)


include_directories(include ${OpenCV_INCLUDE_DIRS})

#rga
include_directories(${RKNN_API_PATH}/include)
set(RGA_PATH ${CMAKE_SOURCE_DIR}/include/3rdparty/rga/RK3588)
set(RGA_LIB ${RGA_PATH}/lib/Linux/aarch64/librga.so)
include_directories(${RGA_PATH}/include)

add_executable(echovisionRktoolkit2
        main.cpp
        include/globaltypes.h
        include/kalman.hpp
        include/socketsender.h
        src/socketsender.cpp
        src/cqueue.hpp
        src/file_manager.cpp
        include/file_manager.h
        include/postprocess.h
        include/preprocess.h
        src/postprocess.cc
        src/preprocess.cc
        include/MotDetect.hpp
        src/rk_common.cpp
)


cmake_print_variables(RGA_PATH)

target_link_libraries(echovisionRktoolkit2
        ${EXTRA_LIBS}
        ${OpenCV_LIBS}
        ${RGA_LIB}
)
