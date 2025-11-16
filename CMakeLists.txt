cmake_minimum_required(VERSION 3.18)
project(cv-tracker LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

# include directories
include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)

# sources
set(SRC_CPP
    src/main.cpp
    src/cpu_match.cpp
    src/video_io.cpp
)

set(SRC_CUDA
    src/ssd_kernel.cu
)

# create target
add_executable(tracker ${SRC_CPP} ${SRC_CUDA})

# ensure CUDA compiler uses NVCC
set_target_properties(tracker PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE OFF
)

# link libs
# target_link_libraries(tracker PRIVATE ${OpenCV_LIBS} )
target_link_libraries(tracker PRIVATE
    opencv_core
    opencv_imgproc
    opencv_highgui
    opencv_videoio
    opencv_imgcodecs
)


# compile flags (optional)
if (MSVC)
    # windows-specific options if needed
else()
    target_compile_options(tracker PRIVATE -O3)
    if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        # enable fast math for CUDA
        target_compile_options(tracker PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-use_fast_math -O3>)
    endif()
endif()
