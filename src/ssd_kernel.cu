// src/ssd_kernel.cu
#include "ssd_match.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " "    \
                 << cudaGetErrorString(err) << std::endl;                     \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

static unsigned char* d_tpl = nullptr;
static int tplW = 0, tplH = 0;
static int frameW_cached = 0, frameH_cached = 0;
static float* d_scores = nullptr;
static unsigned char* d_frame = nullptr;
static int totalCandidatesCached = 0;

// kernel: each thread computes SSD for one candidate location
__global__
void ssd_match_kernel(const unsigned char* frame, int frameW, int frameH,
                      const unsigned char* tpl, int tplW, int tplH,
                      float* scores)
{
    int candidatesW = frameW - tplW + 1;
    int candidatesH = frameH - tplH + 1;
    int total = candidatesW * candidatesH;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int cx = idx % candidatesW;
    int cy = idx / candidatesW;

    unsigned int ssd = 0u;
    int baseFrame = cy * frameW + cx;
    for (int ty = 0; ty < tplH; ++ty) {
        int frow = baseFrame + ty * frameW;
        int trow = ty * tplW;
        for (int tx = 0; tx < tplW; ++tx) {
            int fidx = frow + tx;
            int tidx = trow + tx;
            int diff = int(frame[fidx]) - int(tpl[tidx]);
            ssd += (unsigned int)(diff * diff);
        }
    }
    scores[idx] = float(ssd);
}

bool gpu_ssd_init(const cv::Mat& tplGray)
{
    if (tplGray.empty() || tplGray.type() != CV_8U) {
        std::cerr << "gpu_ssd_init: template must be non-empty CV_8U" << std::endl;
        return false;
    }

    tplW = tplGray.cols; tplH = tplGray.rows;

    // allocate device template
    size_t tplBytes = size_t(tplW) * size_t(tplH);
    CUDA_CHECK(cudaMalloc((void**)&d_tpl, tplBytes));
    CUDA_CHECK(cudaMemcpy(d_tpl, tplGray.data, tplBytes, cudaMemcpyHostToDevice));

    // reset cached frame dims
    frameW_cached = frameH_cached = 0;
    totalCandidatesCached = 0;
    d_scores = nullptr;
    d_frame = nullptr;

    return true;
}

void gpu_ssd_release()
{
    if (d_tpl) { cudaFree(d_tpl); d_tpl = nullptr; }
    if (d_frame) { cudaFree(d_frame); d_frame = nullptr; }
    if (d_scores) { cudaFree(d_scores); d_scores = nullptr; }
    tplW = tplH = 0;
    frameW_cached = frameH_cached = 0;
    totalCandidatesCached = 0;
}

std::pair<cv::Point, float> gpu_ssd_match(const cv::Mat& frameGray)
{
    if (d_tpl == nullptr) {
        std::cerr << "gpu_ssd_match: template not initialized (call gpu_ssd_init first)" << std::endl;
        return {cv::Point(0,0), 0.0f};
    }
    if (frameGray.empty() || frameGray.type() != CV_8U) {
        std::cerr << "gpu_ssd_match: frame must be CV_8U" << std::endl;
        return {cv::Point(0,0), 0.0f};
    }

    int frameW = frameGray.cols, frameH = frameGray.rows;

    int candidatesW = frameW - tplW + 1;
    int candidatesH = frameH - tplH + 1;
    if (candidatesW <= 0 || candidatesH <= 0) {
        std::cerr << "gpu_ssd_match: template larger than frame" << std::endl;
        return {cv::Point(0,0), 0.0f};
    }

    int totalCandidates = candidatesW * candidatesH;

    // (re)allocate frame and scores buffers if needed
    if (frameW != frameW_cached || frameH != frameH_cached) {
        // free old
        if (d_frame) { cudaFree(d_frame); d_frame = nullptr; }
        if (d_scores) { cudaFree(d_scores); d_scores = nullptr; }

        CUDA_CHECK(cudaMalloc((void**)&d_frame, size_t(frameW) * size_t(frameH)));
        CUDA_CHECK(cudaMalloc((void**)&d_scores, sizeof(float) * size_t(totalCandidates)));

        frameW_cached = frameW; frameH_cached = frameH;
        totalCandidatesCached = totalCandidates;
    }

    // copy frame
    CUDA_CHECK(cudaMemcpy(d_frame, frameGray.data, size_t(frameW)*size_t(frameH), cudaMemcpyHostToDevice));

    // kernel launch
    int threads = 256;
    int blocks = (totalCandidates + threads - 1) / threads;

    cudaEvent_t startEvent, stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));
    CUDA_CHECK(cudaEventRecord(startEvent, 0));

    ssd_match_kernel<<<blocks, threads>>>(d_frame, frameW, frameH, d_tpl, tplW, tplH, d_scores);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stopEvent, 0));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));
    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, startEvent, stopEvent));

    // copy back scores
    std::vector<float> h_scores(totalCandidates);
    CUDA_CHECK(cudaMemcpy(h_scores.data(), d_scores, sizeof(float) * size_t(totalCandidates), cudaMemcpyDeviceToHost));

    // find argmin on host
    int argmin = 0;
    float bestScore = h_scores[0];
    for (int i = 1; i < totalCandidates; ++i) {
        if (h_scores[i] < bestScore) {
            bestScore = h_scores[i];
            argmin = i;
        }
    }
    cv::Point bestPt(argmin % candidatesW, argmin / candidatesW);

    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));
    return {bestPt, kernel_ms};
}
