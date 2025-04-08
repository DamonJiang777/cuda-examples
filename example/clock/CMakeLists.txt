cmake_minimum_required(VERSION 3.20)

project(clock LANGUAGES C CXX CUDA)

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

add_executable(clock clock.cu)

set_target_properties(clock PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
