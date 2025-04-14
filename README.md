CUDA Examples
# How to build
```Bash
mkdir build
# copy config/config.cmake to build folder
# update CUDA_TOOLKIT_ROOT_DIR variable
# eg. set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12")
cd build && cmake .. && make -j32
```
