#ifndef SSD_MATCH_H
#define SSD_MATCH_H

#include <opencv2/opencv.hpp>
#include <utility>

// Initialize GPU template (call once after reading template)
// returns true on success
bool gpu_ssd_init(const cv::Mat& tplGray);

// Release GPU resources
void gpu_ssd_release();

// Run GPU SSD matcher on a single grayscale frame.
// frameGray must be continuous CV_8U mat (grayscale).
// Returns pair: (bestPoint, kernel_time_ms)
std::pair<cv::Point, float> gpu_ssd_match(const cv::Mat& frameGray);

// CPU baseline SSD match (pure CPU)
cv::Point cpu_ssd_match(const cv::Mat& frameGray, const cv::Mat& tplGray);

#endif // SSD_MATCH_H
