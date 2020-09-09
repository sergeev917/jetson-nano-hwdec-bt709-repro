/*
 * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "NvUtils.h"
#include "NvVideoDecoder.h"
#include <cudaEGL.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "EGL/egl.h"
#include "EGL/eglext.h"
#include <errno.h>
#include <fcntl.h>
#include <fstream>
#include <sstream>
#include <chrono>
#include <memory>
#include <iomanip>
#include <iostream>
#include <linux/videodev2.h>
#include <malloc.h>
#include <nvbuf_utils.h>
#include <pthread.h>
#include <string.h>
#include <cassert>
#include <unistd.h>
#include "nvbuf_utils.h"

#define CHUNK_SIZE 4000000
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX_BUFFERS 32

using namespace std;

inline void
cuda_check_impl(
    const CUresult rv,
    const char * file,
    const unsigned line,
    const char * func
)
{
    if (rv != CUDA_SUCCESS) {
        ostringstream fmt;
        fmt << file << ":" << line << ": " << func;
        fmt << ": cuda driver api: error " << int(rv);
        const char * description = nullptr;
        cuGetErrorString(rv, &description);
        if (description != nullptr) {
            fmt << ": " << description;
        }
        fmt << '\n';
        cerr << move(fmt).str();
        assert(false);
    }
}

#define cuda_check(rv) cuda_check_impl(rv, __FILE__, __LINE__, __PRETTY_FUNCTION__)

typedef struct {
    CUcontext cuda_ctx;
    NvVideoDecoder *dec;
    EGLDisplay egl_display;
    ifstream in_file;
    uint32_t video_height;
    uint32_t video_width;
    uint32_t display_height;
    uint32_t display_width;
    pthread_t dec_capture_loop;
    bool got_error;
    bool got_eos;
    int dst_dma_fd;
    int extra_cap_plane_buffer;
} context_t;

static void
read_decoder_input_chunk(ifstream & stream, NvBuffer * buffer)
{
    streamsize bytes_to_read = MIN(CHUNK_SIZE, buffer->planes[0].length);
    stream.read((char *)buffer->planes[0].data, bytes_to_read);
    buffer->planes[0].bytesused = stream.gcount();
}

static void
abort(context_t * ctx)
{
    ctx->got_error = true;
    ctx->dec->abort();
}

static void
query_and_set_capture(context_t * ctx)
{
    NvVideoDecoder * dec = ctx->dec;
    struct v4l2_format format;
    struct v4l2_crop crop;
    int32_t min_dec_capture_buffers;
    int ret = 0;

    ret = dec->capture_plane.getFormat(format); assert(ret >= 0);
    ret = dec->capture_plane.getCrop(crop); assert(ret >= 0);

    cout << "Video Resolution: " << crop.c.width << "x" << crop.c.height << endl;
    ctx->display_height = crop.c.height;
    ctx->display_width = crop.c.width;
    if (ctx->dst_dma_fd != -1) {
        NvBufferDestroy(ctx->dst_dma_fd);
        ctx->dst_dma_fd = -1;
    }
    NvBufferCreateParams input_params = {0};
    input_params.payloadType = NvBufferPayload_SurfArray;
    input_params.width = crop.c.width;
    input_params.height = crop.c.height;
    input_params.layout = NvBufferLayout_Pitch;
    input_params.colorFormat = NvBufferColorFormat_NV12_709; // NOTE: NvBufferColorFormat_NV12 works
    input_params.nvbuf_tag = NvBufferTag_VIDEO_CONVERT;
    ret = NvBufferCreateEx(&ctx->dst_dma_fd, &input_params); assert(ret != -1);

    dec->capture_plane.deinitPlane();
    ret = dec->setCapturePlaneFormat(format.fmt.pix_mp.pixelformat, format.fmt.pix_mp.width, format.fmt.pix_mp.height); assert(ret >= 0);
    ctx->video_height = format.fmt.pix_mp.height;
    ctx->video_width = format.fmt.pix_mp.width;
    ret = dec->getMinimumCapturePlaneBuffers(min_dec_capture_buffers); assert(ret >= 0);
    ret = dec->capture_plane.setupPlane(V4L2_MEMORY_MMAP, min_dec_capture_buffers + ctx->extra_cap_plane_buffer, false, false); assert(ret >= 0);
    ret = dec->capture_plane.setStreamStatus(true); assert(ret >= 0);

    for (uint32_t i = 0; i < dec->capture_plane.getNumBuffers(); i++) {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];
        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, sizeof(planes));
        v4l2_buf.index = i;
        v4l2_buf.m.planes = planes;
        v4l2_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        v4l2_buf.memory = V4L2_MEMORY_MMAP;
        ret = dec->capture_plane.qBuffer(v4l2_buf, NULL); assert(ret >= 0);
    }
    cout << "Query and set capture successful" << endl;
}

static void *
dec_capture_loop_fcn(void * arg)
{
    context_t * ctx = (context_t *)arg;
    NvVideoDecoder * dec = ctx->dec;
    struct v4l2_event ev;
    int ret;

    cuCtxPushCurrent(ctx->cuda_ctx);

    cout << "Starting decoder capture loop thread" << endl;
    do {
        ret = dec->dqEvent(ev, 50000);
        if (ret < 0) {
            if (errno == EAGAIN) {
                cerr << "Timed out waiting for first V4L2_EVENT_RESOLUTION_CHANGE\n";
            } else {
                cerr << "Error in dequeueing decoder event\n";
            }
            abort(ctx);
            break;
        }
    } while ((ev.type != V4L2_EVENT_RESOLUTION_CHANGE) && !ctx->got_error);

    if (!ctx->got_error) {
        query_and_set_capture(ctx);
    }

    while (!(ctx->got_error || dec->isInError() || ctx->got_eos)) {
        ret = dec->dqEvent(ev, false);
        if (ret == 0) {
            switch (ev.type) {
            case V4L2_EVENT_RESOLUTION_CHANGE:
                query_and_set_capture(ctx);
                continue;
            }
        }

        while (1) {
            struct v4l2_buffer v4l2_buf;
            struct v4l2_plane planes[MAX_PLANES];
            memset(&v4l2_buf, 0, sizeof(v4l2_buf));
            memset(planes, 0, sizeof(planes));
            v4l2_buf.m.planes = planes;

            NvBuffer * dec_buffer;
            if (dec->capture_plane.dqBuffer(v4l2_buf, &dec_buffer, NULL, 0)) {
                if (errno == EAGAIN) {
                    usleep(1000);
                } else {
                    abort(ctx);
                    cerr << "Error while calling dequeue at capture plane" << endl;
                }
                break;
            }

            NvBufferTransformParams transform_params;
            memset(&transform_params, 0, sizeof(transform_params));
            transform_params.transform_flag = NVBUFFER_TRANSFORM_FILTER;
            transform_params.transform_flip = NvBufferTransform_None;
            transform_params.transform_filter = NvBufferTransform_Filter_Smart;
            transform_params.src_rect.top = 0;
            transform_params.src_rect.left = 0;
            transform_params.src_rect.width = ctx->display_width;
            transform_params.src_rect.height = ctx->display_height;
            transform_params.dst_rect.top = 0;
            transform_params.dst_rect.left = 0;
            transform_params.dst_rect.width = ctx->display_width;
            transform_params.dst_rect.height = ctx->display_height;
            ret = NvBufferTransform(dec_buffer->planes[0].fd, ctx->dst_dma_fd, &transform_params); assert(ret != -1);

            EGLImageKHR egl_image = NvEGLImageFromFd(ctx->egl_display, ctx->dst_dma_fd); assert(egl_image != nullptr);
            CUresult status;
            CUeglFrame eglFrame;
            CUgraphicsResource pResource = nullptr;
            status = cuGraphicsEGLRegisterImage(&pResource, egl_image, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE); cuda_check(status);
            status = cuGraphicsResourceGetMappedEglFrame(&eglFrame, pResource, 0, 0); cuda_check(status);
            assert(eglFrame.frameType == CU_EGL_FRAME_TYPE_PITCH);
            // NOTE: process eglFrame data in CUDA kernels, etc
            status = cuGraphicsUnregisterResource(pResource); cuda_check(status);
            NvDestroyEGLImage(ctx->egl_display, egl_image);

            if (dec->capture_plane.qBuffer(v4l2_buf, NULL) < 0) {
                abort(ctx);
                cerr << "Error while queueing buffer at decoder capture plane" << endl;
                break;
            }
        }
    }
    cuCtxPopCurrent(nullptr);
    cout << "Exiting decoder capture loop thread" << endl;
    return NULL;
}

int
main(int argc, char * argv[])
{
    if (argc != 2) {
        cerr << "Provide path to a video file\n";
        return 1;
    }
    context_t ctx;
    ctx.got_eos = false;
    ctx.got_error = false;
    ctx.dst_dma_fd = -1;
    ctx.extra_cap_plane_buffer = 1;
    ctx.video_height = 0;
    ctx.video_width = 0;
    ctx.display_height = 0;
    ctx.display_width = 0;

    CUdevice dev;
    { auto rv = cuInit(0); cuda_check(rv); }
    { auto rv = cuDeviceGet(&dev, 0); cuda_check(rv); }
    { auto rv = cuCtxCreate(&ctx.cuda_ctx, CU_CTX_SCHED_BLOCKING_SYNC, dev); cuda_check(rv); }

    ctx.egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (ctx.egl_display == EGL_NO_DISPLAY) {
        cerr << "Error while get EGL display connection\n";
        return 1;
    }
    if (!eglInitialize(ctx.egl_display, nullptr, nullptr)) {
        cerr << "Erro while initialize EGL display connection\n";
        return 1;
    }

    int ret = 0;
    int error = 0;
    bool eos = false;

    ctx.dec = NvVideoDecoder::createVideoDecoder("dec0"); assert(ctx.dec);
    ctx.in_file.open(argv[1]); assert(ctx.in_file.is_open());
    ret = ctx.dec->subscribeEvent(V4L2_EVENT_RESOLUTION_CHANGE, 0, 0); assert(ret >= 0);
    ret = ctx.dec->setOutputPlaneFormat(V4L2_PIX_FMT_H264, CHUNK_SIZE); assert(ret >= 0);
    ret = ctx.dec->setFrameInputMode(1); assert(ret >= 0);
    ret = ctx.dec->output_plane.setupPlane(V4L2_MEMORY_MMAP, 10, true, false); assert(ret >= 0);
    ret = ctx.dec->output_plane.setStreamStatus(true); assert(ret >= 0);

    for (unsigned i = 0; i < ctx.dec->output_plane.getNumBuffers(); ++i) {
        NvBuffer * buffer = ctx.dec->output_plane.getNthBuffer(i);
        read_decoder_input_chunk(ctx.in_file, buffer);

        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];
        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, sizeof(planes));
        v4l2_buf.index = i;
        v4l2_buf.m.planes = planes;
        v4l2_buf.m.planes[0].bytesused = buffer->planes[0].bytesused;

        ret = ctx.dec->output_plane.qBuffer(v4l2_buf, NULL);
        if (ret < 0) {
            cerr << "Error Qing buffer at output plane" << endl;
            abort(&ctx);
            break;
        }
        if (v4l2_buf.m.planes[0].bytesused == 0) {
            eos = true;
            cout << "Input file read complete" << endl;
            break;
        }
        if (eos || ctx.got_error || ctx.dec->isInError()) {
            break;
        }
    }

    pthread_create(&ctx.dec_capture_loop, NULL, dec_capture_loop_fcn, &ctx);

    while (!eos && !ctx.got_error && !ctx.dec->isInError()) {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];
        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, sizeof(planes));
        v4l2_buf.m.planes = &planes[0];
        NvBuffer * buffer;
        ret = ctx.dec->output_plane.dqBuffer(v4l2_buf, &buffer, NULL, -1);
        if (ret < 0) {
            cerr << "Error DQing buffer at output plane" << endl;
            abort(&ctx);
            break;
        }
        read_decoder_input_chunk(ctx.in_file, buffer);
        v4l2_buf.m.planes[0].bytesused = buffer->planes[0].bytesused;
        ret = ctx.dec->output_plane.qBuffer(v4l2_buf, NULL);
        if (ret < 0) {
            cerr << "Error Qing buffer at output plane" << endl;
            abort(&ctx);
            break;
        }
        if (v4l2_buf.m.planes[0].bytesused == 0) {
            eos = true;
            cout << "Input file read complete" << endl;
            break;
        }
    }

    while (ctx.dec->output_plane.getNumQueuedBuffers() > 0 && !ctx.got_error && !ctx.dec->isInError()) {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];
        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, sizeof(planes));
        v4l2_buf.m.planes = &planes[0];

        ret = ctx.dec->output_plane.dqBuffer(v4l2_buf, NULL, NULL, -1);
        if (ret < 0) {
            cerr << "Error DQing buffer at output plane" << endl;
            abort(&ctx);
            break;
        }
    }

    ctx.got_eos = true;
    if (ctx.dec_capture_loop) {
        pthread_join(ctx.dec_capture_loop, NULL);
    }
    if (ctx.dec && ctx.dec->isInError()) {
        cerr << "Decoder is in error" << endl;
        error = 1;
    }
    if (ctx.got_error) {
        error = 1;
    }
    delete ctx.dec;
    if (ctx.dst_dma_fd != -1) {
        NvBufferDestroy(ctx.dst_dma_fd);
        ctx.dst_dma_fd = -1;
    }
    if (ctx.egl_display && !eglTerminate(ctx.egl_display)) {
        cerr << "Error while terminate EGL display connection\n";
    }
    cuCtxPopCurrent(nullptr);
    cuCtxDestroy(ctx.cuda_ctx);
    if (error != 0) {
        cout << "App run failed" << endl;
    } else {
        cout << "App run was successful" << endl;
    }
    return 0;
}
