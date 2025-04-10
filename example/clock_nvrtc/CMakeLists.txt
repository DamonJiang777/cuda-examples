cmake_minimum_required(VERSION 3.20)

project(clock_nvrtc LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 80 86 87 89)

set(CMAKE_CUDA_FLAGS "${CMAKA_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CUDA_FLAGS "{CMAKE_CUDA_FLAGS} -G")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

include_directories(../../common)

add_executable(clock_nvrtc clock_nvrtc.cpp)

set_target_properties(clock PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)

find_library(NVRTC_LIB nvrtc PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 REQUIRED)
find_library(CUDART_LIB cudart PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 REQUIRED)
target_link_libraries(clock_nvrtc PRIVATE ${NVRTC_LIB} ${CUART_LIB} cuda)

# copy clock_kernel.cu to the output directory
add_custom_command(
  TARGET clock_nvrtc
  POST_BUILD
  COMMAND
    ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/clock_kernel.cu ${CMAKE_CURRENT_BINARY_DIR})
