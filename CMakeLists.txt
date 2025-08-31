cmake_minimum_required(VERSION 3.12)
project(pro CXX)

# 设置 C++ 标准，避免重复设置
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 设置构建类型
set(CMAKE_BUILD_TYPE Debug)

# 设置可执行文件输出路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/workspace)

# 如果你是不同显卡，请设置为显卡对应的号码参考这里：https://developer.nvidia.com/zh-cn/cuda-gpus#compute
set(CUDA_GEN_CODE "-gencode=arch=compute_75,code=sm_75")

# 设置依赖库路径
set(OpenCV_DIR   "/root/host_container_mapping_dir/container1/hard_decode_trt/opencv-build")
set(CUDA_DIR     "/usr/local/cuda-12.9")
set(NVDEC_DIR    "/root/host_container_mapping_dir/container1/hard_decode_trt/Video_Codec_SDK_13.0.19")
set(FFMPEG_DIR   "/root/host_container_mapping_dir/container1/hard_decode_trt/ffmpeg-build")
# set(PROTOBUF_DIR "/root/host_container_mapping_dir/container1/hard_decode_trt/protobuf")

# 包含头文件目录
include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${OpenCV_DIR}/include/opencv4
    ${CUDA_DIR}/include
    # ${PROTOBUF_DIR}/include
    ${NVDEC_DIR}/Interface
    ${FFMPEG_DIR}/include
)

# 链接库目录
link_directories(
    # ${PROTOBUF_DIR}/lib
    ${CUDA_DIR}/lib64
    ${NVDEC_DIR}/Lib/linux/stubs/x86_64
    ${FFMPEG_DIR}/lib
    ${OpenCV_DIR}/lib
)

# 设置编译选项，移除重复的 C++ 标准设置
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -Wfatal-errors -pthread -w -g")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fPIC -g -w ${CUDA_GEN_CODE}")

# 查找源文件
file(GLOB_RECURSE cpp_srcs ${PROJECT_SOURCE_DIR}/src/*.cpp)

# 创建可执行文件
add_executable(pro ${cpp_srcs})

# 链接库
target_link_libraries(pro 
    cuda cublas cudart
    nvcuvid nvidia-encode
    # protobuf 
    pthread
    avcodec avformat swresample swscale avutil
    opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs
)

# 添加自定义目标
add_custom_target(
    hard_decode
    DEPENDS pro
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./pro hard_decode
)
