#!/bin/bash
set -o errexit
g++ \
    -std=c++17 \
    -Ofast -ggdb3 \
    -o main \
    -I include \
    -I /usr/local/cuda-10.2/targets/aarch64-linux/include \
    sources/App.cpp \
    sources/NvVideoDecoder.cpp \
    sources/NvElement.cpp \
    sources/NvElementProfiler.cpp \
    sources/NvBuffer.cpp \
    sources/NvV4l2Element.cpp \
    sources/NvV4l2ElementPlane.cpp \
    sources/NvVideoConverter.cpp \
    sources/NvLogging.cpp \
    -pthread \
    -lv4l2 \
    -lEGL \
    -lGLESv2 \
    -L /usr/lib/aarch64-linux-gnu/tegra \
    -ltegrav4l2 \
    -lnvbuf_utils \
    -L /usr/local/cuda-10.2/targets/aarch64-linux/lib \
    -lcuda
./main video_1920x1080_25fps_x3.h264
