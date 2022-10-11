#pragma once
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <utility>
#include "utils.h"


class HybridNetsDetector
{
public:
    explicit HybridNetsDetector(std::nullptr_t) {};
    HybridNetsDetector(const std::string& modelPath,
                 const bool& isGPU,
                 const cv::Size& inputSize);

    Result detect(cv::Mat &image, const float& confThreshold, const float& iouThreshold);

private:
    Ort::Env env{nullptr};
    Ort::SessionOptions sessionOptions{nullptr};
    Ort::Session session{nullptr};

    std::vector<int> preprocessing(cv::Mat &image, float*& blob, std::vector<int64_t>& inputTensorShape);
    Result postprocessing(cv::Mat &mat, const cv::Size& resizedImageShape,
                                          const cv::Size& originalImageShape,
                                          std::vector<Ort::Value>& outputTensors,
                                          const float& confThreshold, const float& iouThreshold, std::vector<int>& pad);

    static void getBestClassInfo(std::vector<float>::iterator it, const int& numClasses,
                                 float& bestConf, int& bestClassId);

    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;
    bool isDynamicInputShape{};
    cv::Size2f inputImageShape;

};