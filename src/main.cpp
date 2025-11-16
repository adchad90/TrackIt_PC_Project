// src/main.cpp
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <cstring>

#include "ssd_match.h"

using namespace cv;
using namespace std;
using Clock = chrono::high_resolution_clock;

static void print_usage(const char* prog) {
    cout << "Usage: " << prog << " input_video output_video [--template_image tpl.jpg | --template_crop x y w h]\n";
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }
    string inputPath = argv[1];
    string outputPath = argv[2];

    string tplImagePath = "";
    bool tplFromCrop = false;
    int cropX=0, cropY=0, cropW=0, cropH=0;
    for (int i=3;i<argc;i++){
        if (strcmp(argv[i],"--template_image")==0 && i+1<argc){
            tplImagePath = argv[++i];
        } else if (strcmp(argv[i],"--template_crop")==0 && i+4<argc){
            tplFromCrop = true;
            cropX = stoi(argv[++i]); cropY = stoi(argv[++i]);
            cropW = stoi(argv[++i]); cropH = stoi(argv[++i]);
        }
    }

    VideoCapture cap(inputPath);
    if (!cap.isOpened()) {
        cerr << "Failed to open input video: " << inputPath << endl;
        return 1;
    }

    int frameW = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int frameH = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
    double fps_in = cap.get(CAP_PROP_FPS);
    if (fps_in <= 0) fps_in = 30.0;

    VideoWriter writer;
    int codec = VideoWriter::fourcc('a','v','c','1');
    bool openedWriter = writer.open(outputPath, codec, fps_in, Size(frameW, frameH));
    if (!openedWriter) {
        codec = VideoWriter::fourcc('M','J','P','G');
        openedWriter = writer.open(outputPath, codec, fps_in, Size(frameW, frameH));
        if (!openedWriter) {
            cerr << "Failed to open output video for writing: " << outputPath << endl;
            return 1;
        }
    }

    // read first frame
    Mat firstFrame;
    if (!cap.read(firstFrame)) {
        cerr << "Failed to read first frame." << endl;
        return 1;
    }

    Mat tplColor;
    if (!tplImagePath.empty()) {
        tplColor = imread(tplImagePath);
        if (tplColor.empty()) {
            cerr << "Failed to load template image: " << tplImagePath << endl;
            return 1;
        }
    } else if (tplFromCrop) {
        Rect cropRect(cropX, cropY, cropW, cropH);
        cropRect &= Rect(0,0,firstFrame.cols, firstFrame.rows);
        tplColor = firstFrame(cropRect).clone();
    } else {
        int w = max(32, firstFrame.cols/10);
        int h = max(24, firstFrame.rows/12);
        Rect centerRect((firstFrame.cols - w)/2,(firstFrame.rows-h)/2,w,h);
        cout << "No template specified. Using center crop from first frame." << endl;
        tplColor = firstFrame(centerRect).clone();
    }

    Mat tplGray;
    cvtColor(tplColor, tplGray, COLOR_BGR2GRAY);

    // initialize GPU template
    if (!gpu_ssd_init(tplGray)) {
        cerr << "Failed to initialize GPU template." << endl;
        return 1;
    }

    // rewind to first frame to process from start
    cap.set(CAP_PROP_POS_FRAMES, 0);

    long long frameCount = 0;
    double cpu_total_ms = 0.0, gpu_total_ms = 0.0;

    Mat frameColor, frameGray;
    while (cap.read(frameColor)) {
        ++frameCount;
        cvtColor(frameColor, frameGray, COLOR_BGR2GRAY);

        // CPU baseline
        auto t0 = Clock::now();
        Point bestCPU = cpu_ssd_match(frameGray, tplGray);
        auto t1 = Clock::now();
        double cpu_ms = chrono::duration<double, milli>(t1 - t0).count();
        cpu_total_ms += cpu_ms;

        // GPU
        auto gpu_res = gpu_ssd_match(frameGray);
        Point bestGPU = gpu_res.first;
        float kernel_ms = gpu_res.second;
        gpu_total_ms += kernel_ms;

        // draw
        int tW = tplGray.cols, tH = tplGray.rows;
        Rect boxCPU(bestCPU.x, bestCPU.y, tW, tH);
        Rect boxGPU(bestGPU.x, bestGPU.y, tW, tH);
        Mat outFrame = frameColor.clone();
        rectangle(outFrame, boxCPU, Scalar(0,255,0), 2);
        rectangle(outFrame, boxGPU, Scalar(0,0,255), 2);

        double avgCPUfps = (frameCount==0)?0.0:1000.0/(cpu_total_ms/frameCount);
        double avgGPUfps = (frameCount==0)?0.0:1000.0/(gpu_total_ms/frameCount);
        char info[200];
        snprintf(info, sizeof(info), "Frame %lld CPU_fps(avg)=%.2f GPU_fps(avg)=%.2f kernel_ms=%.2f cpu_ms=%.2f",
                 frameCount, avgCPUfps, avgGPUfps, kernel_ms, cpu_ms);
        putText(outFrame, info, Point(10,30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255,255,255), 1);

        writer.write(outFrame);

        if (frameCount % 20 == 0) {
            cout << "Frame " << frameCount << ": CPU_ms=" << cpu_ms << " GPU_kernel_ms=" << kernel_ms
                 << " CPU_best=(" << bestCPU.x << "," << bestCPU.y << ") GPU_best=(" << bestGPU.x << "," << bestGPU.y << ")" << endl;
        }
    }

    cout << "Processed frames: " << frameCount << endl;
    cout << "Average CPU match time (ms): " << (cpu_total_ms / frameCount) << " -> FPS " << (1000.0 / (cpu_total_ms / frameCount)) << endl;
    cout << "Average GPU kernel time (ms): " << (gpu_total_ms / frameCount) << " -> GPU kernel FPS " << (1000.0 / (gpu_total_ms / frameCount)) << endl;

    gpu_ssd_release();
    writer.release();
    cap.release();

    cout << "Saved output to " << outputPath << endl;
    return 0;
}
