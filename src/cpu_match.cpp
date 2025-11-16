// src/cpu_match.cpp
#include "ssd_match.h"
#include <limits>
#include <iostream>

cv::Point cpu_ssd_match(const cv::Mat& frameGray, const cv::Mat& tplGray)
{
    if (frameGray.empty() || tplGray.empty()) {
        std::cerr << "cpu_ssd_match: empty input" << std::endl;
        return cv::Point(0,0);
    }
    if (frameGray.type() != CV_8U || tplGray.type() != CV_8U) {
        std::cerr << "cpu_ssd_match: inputs must be CV_8U" << std::endl;
        return cv::Point(0,0);
    }

    int fW = frameGray.cols, fH = frameGray.rows;
    int tW = tplGray.cols, tH = tplGray.rows;
    int bestX = 0, bestY = 0;
    unsigned long long bestSSD = ULLONG_MAX;

    for (int y = 0; y <= fH - tH; ++y) {
        for (int x = 0; x <= fW - tW; ++x) {
            unsigned long long ssd = 0;
            for (int ty = 0; ty < tH; ++ty) {
                const unsigned char* fptr = frameGray.ptr<unsigned char>(y + ty) + x;
                const unsigned char* tptr = tplGray.ptr<unsigned char>(ty);
                for (int tx = 0; tx < tW; ++tx) {
                    int diff = int(fptr[tx]) - int(tptr[tx]);
                    ssd += (unsigned long long)(diff * diff);
                }
            }
            if (ssd < bestSSD) {
                bestSSD = ssd;
                bestX = x; bestY = y;
            }
        }
    }
    return cv::Point(bestX, bestY);
}
