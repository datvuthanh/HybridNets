#include "hybridnets_cpp/detector.h"
#include "hybridnets_cpp/prior_bbox.h"
#include <chrono>
#include <thread>
#include <limits>


static const std::vector<std::string> kLabelListDet{ "Car" };
static const std::vector<std::string> kLabelListSeg{ "Background", "Lane", "Line" };

HybridNetsDetector::HybridNetsDetector(const std::string& modelPath,
                           const bool& isGPU = true,
                           const cv::Size& inputSize = cv::Size(384, 256))
{
    env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ONNX_DETECTION");
    sessionOptions = Ort::SessionOptions();

    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    for (auto i : availableProviders) std::cout << i << std::endl;
    // auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
    // OrtCUDAProviderOptions cudaOption;

    auto tensorrtAvailable = std::find(availableProviders.begin(), availableProviders.end(), "TensorrtExecutionProvider");
    OrtTensorRTProviderOptions trt_options{};
    trt_options.trt_fp16_enable = 1;
    // trt_options.trt_int8_enable = 1;
    // trt_options.trt_int8_use_native_calibration_table = 1;
    trt_options.trt_engine_cache_enable = 1;
    trt_options.trt_engine_cache_path = "./";
    trt_options.trt_max_workspace_size = 1073741824;  // 1GB

    // if (isGPU && (cudaAvailable == availableProviders.end()))
    // {
    //     std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
    //     std::cout << "Inference device: CPU" << std::endl;
    // }
    // else if (isGPU && (cudaAvailable != availableProviders.end()))
    // {
    //     std::cout << "Inference device: GPU" << std::endl;
    //     sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
    // }
    // else
    // {
    //     std::cout << "Inference device: CPU" << std::endl;
    // }

    if (isGPU && (tensorrtAvailable == availableProviders.end()))
    {
        std::cout << "tensorrt is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
        std::cout << "Inference device: CPU" << std::endl;
    }
    else if (isGPU && (tensorrtAvailable != availableProviders.end()))
    {
        std::cout << "Inference device: tensorrt" << std::endl;
        sessionOptions.AppendExecutionProvider_TensorRT(trt_options);

    }
    else
    {
        std::cout << "Inference device: CPU" << std::endl;
    }

#ifdef _WIN32
    std::wstring w_modelPath = utils::charToWstring(modelPath.c_str());
    session = Ort::Session(env, w_modelPath.c_str(), sessionOptions);
#else
    session = Ort::Session(env, modelPath.c_str(), sessionOptions);
#endif

    Ort::AllocatorWithDefaultOptions allocator;

    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    std::vector<int64_t> inputTensorShape = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
    this->isDynamicInputShape = false;
    // checking if width and height are dynamic
    if (inputTensorShape[2] == -1 && inputTensorShape[3] == -1)
    {
        std::cout << "Dynamic input shape" << std::endl;
        this->isDynamicInputShape = true;
    }

    for (auto shape : inputTensorShape)
        std::cout << "Input shape: " << shape << std::endl;

    inputNames.push_back(session.GetInputName(0, allocator));
    for (int i = 0; i < 3; i++) outputNames.push_back(session.GetOutputName(i, allocator));

    std::cout << "Input name: " << inputNames[0] << std::endl;
    for (auto i : outputNames) std::cout << "Output name: " << i << std::endl;

    this->inputImageShape = cv::Size2f(inputSize);
}

std::vector<int> HybridNetsDetector::preprocessing(cv::Mat &image, float*& blob, std::vector<int64_t>& inputTensorShape)
{
    cv::Mat resizedImage, floatImage;
    cv::cvtColor(image, resizedImage, cv::COLOR_BGR2RGB);
    std::vector<int> pad = utils::letterbox(resizedImage, resizedImage, this->inputImageShape,
                     cv::Scalar(114, 114, 114), this->isDynamicInputShape,
                     false, true, 32);

    inputTensorShape[2] = resizedImage.rows;
    inputTensorShape[3] = resizedImage.cols;

    resizedImage.convertTo(floatImage, CV_32FC3, 1 / 255.0);
    blob = new float[floatImage.cols * floatImage.rows * floatImage.channels()];
    cv::Size floatImageSize {floatImage.cols, floatImage.rows};

    // hwc -> chw
    std::vector<cv::Mat> chw(floatImage.channels());
    for (int i = 0; i < floatImage.channels(); ++i)
    {
        chw[i] = cv::Mat(floatImageSize, CV_32FC1, blob + i * floatImageSize.width * floatImageSize.height);
    }
    cv::split(floatImage, chw);
    return pad;
}

Result HybridNetsDetector::postprocessing(cv::Mat &mat, const cv::Size& resizedImageShape,
                                                    const cv::Size& originalImageShape,
                                                    std::vector<Ort::Value>& outputTensors,
                                                    const float& confThreshold, const float& iouThreshold, std::vector<int>& pad)
{
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> classIds;

    auto* reg = outputTensors[0].GetTensorData<float>();
    auto* cls = outputTensors[1].GetTensorData<float>();
    auto* seg = outputTensors[2].GetTensorData<float>();
    // auto* rawOutput = outputTensors[0].GetTensorData<float>();
    // std::cout << outputTensors.size() << std::endl;
    // std::cout << outputTensors[0].GetTensorTypeAndShapeInfo().GetShape().size() << std::endl;

    // segmentation
    const bool IS_NCHW = true;
    cv::Mat mat_seg_max(resizedImageShape.height, resizedImageShape.width, CV_8UC1, cv::Scalar(0));
    cv::Mat mat_road(resizedImageShape.height, resizedImageShape.width, CV_8UC1, cv::Scalar(0));
    cv::Mat mat_lane(resizedImageShape.height, resizedImageShape.width, CV_8UC1, cv::Scalar(0));
#pragma omp parallel for
    for (int32_t y = 0; y < resizedImageShape.height; y++) {
        for (int32_t x = 0; x < resizedImageShape.width; x++) {
            if (IS_NCHW) {
                int32_t class_index_max = 0;
                float class_score_max = -std::numeric_limits<float>::infinity();
                for (int32_t class_index = 0; class_index < static_cast<int32_t>(kLabelListSeg.size()); class_index++) {
                    float score = seg[class_index * resizedImageShape.height * resizedImageShape.width + resizedImageShape.width * y + x];
                    if (score > class_score_max) {
                        class_score_max = score;
                        class_index_max = class_index;
                    }
                }
                mat_seg_max.at<uint8_t>(cv::Point(x, y)) = static_cast<uint8_t>(class_index_max);
                if (class_index_max == 1) mat_road.at<uint8_t>(cv::Point(x, y)) = 255;
                else if (class_index_max == 2) mat_lane.at<uint8_t>(cv::Point(x, y)) = 255;
            } else {
                // const auto& current_iter = output_seg_list.begin() + y * input_tensor_info.GetWidth() * kLabelListSeg.size() + x * kLabelListSeg.size();
                // const auto& max_iter = std::max_element(current_iter, current_iter + kLabelListSeg.size());
                // auto max_c = std::distance(current_iter, max_iter);
                // mat_seg_max.at<uint8_t>(cv::Point(x, y)) = static_cast<uint8_t>(max_c);
            }
        }
    }

    // cv::imwrite("road.jpg", mat_road);
    // cv::imwrite("lane.jpg", mat_lane);



    // Uncomment this for merged road and lane image
    // cv::Mat mat_seg_max_list[] = { mat_seg_max, mat_seg_max, mat_seg_max };
    // cv::merge(mat_seg_max_list, 3, mat_seg_max);
    // // std::cout << mat_seg_max.size() << std::endl;
    // cv::Mat mat_lut = cv::Mat::zeros(256, 1, CV_8UC3);
    // mat_lut.at<cv::Vec3b>(0) = cv::Vec3b(0, 0, 0);
    // mat_lut.at<cv::Vec3b>(1) = cv::Vec3b(0, 255, 0);
    // mat_lut.at<cv::Vec3b>(2) = cv::Vec3b(0, 0, 255);
    // cv::LUT(mat_seg_max, mat_lut, mat_seg_max);
    // // std::cout << originalImageShape << std::endl;
    // mat_seg_max = mat_seg_max(cv::Rect(pad[0]/2, pad[1]/2, resizedImageShape.width - pad[0], resizedImageShape.height - pad[1]));
    // cv::resize(mat_seg_max, mat_seg_max, originalImageShape, 0.0, 0.0, cv::INTER_CUBIC);
    // cv::Mat mat_masked;
    // cv::addWeighted(mat, 0.8, mat_seg_max, 0.5, 0, mat_masked);
    // //cv::add(mat_seg_max * kResultMixRatio, mat * (1.0f - kResultMixRatio), mat_masked);
    // //cv::hconcat(mat, mat_masked, mat);
    // mat = mat_masked;
    // // cv::imwrite("seg.jpg", mat);
    // // cv::imshow("seg", mat);
    // // cv::waitKey(0);
    // // std::cout << "SAVED" << ", PLEASE CLOSE"; 
    // // std::this_thread::sleep_for(std::chrono::milliseconds(5000));



    int crop_x {0}, crop_y {0}, crop_w {1280}, crop_h {720};

    /* Get boundig box */
    /* reference: https://github.dev/datvuthanh/HybridNets/blob/c626bb89beb1b52440bacdbcc90ac60f9814c9a2/utils/utils.py#L615-L616 */
    float scale_w = static_cast<float>(crop_w) / (resizedImageShape.width - pad[0]); // dw
    float scale_h = static_cast<float>(crop_h) / (resizedImageShape.height - pad[1]);  // dh
    // std::cout << "W"  << scale_w << std::endl;
    // std::cout << "H"  << scale_h << std::endl; 
    // float scale_w = 1;
    // float scale_h = 1;

    static const size_t kNumPrior = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape()[1] / kLabelListDet.size();
    std::vector<BoundingBox> bbox_list;
    for (size_t i = 0; i < kNumPrior; i++) {
        size_t class_index = 0;
        float class_score = cls[i];
        size_t prior_index = i * 4;
        if (class_score >= confThreshold) {
            /* Prior Box: [0.0, MODEL_SIZE], y0, x0, y1, x1 */
            const float prior_x0 = PRIOR_BBOX::BBOX[prior_index + 1];
            const float prior_y0 = PRIOR_BBOX::BBOX[prior_index + 0];
            const float prior_x1 = PRIOR_BBOX::BBOX[prior_index + 3];
            const float prior_y1 = PRIOR_BBOX::BBOX[prior_index + 2];
            const float prior_cx = (prior_x0 + prior_x1) / 2.0f;
            const float prior_cy = (prior_y0 + prior_y1) / 2.0f;
            const float prior_w = prior_x1 - prior_x0;
            const float prior_h = prior_y1 - prior_y0;

            /* Detected Box */
            float box_cx = reg[prior_index + 1];
            float box_cy = reg[prior_index + 0];
            float box_w = reg[prior_index + 3];
            float box_h = reg[prior_index + 2];

            /* Adjust box [0.0, 1.0] */
            float cx = PRIOR_BBOX::VARIANCE[1] * box_cx * prior_w + prior_cx;
            float cy = PRIOR_BBOX::VARIANCE[0] * box_cy * prior_h + prior_cy;
            float w = std::exp(box_w * PRIOR_BBOX::VARIANCE[3]) * prior_w;
            float h = std::exp(box_h * PRIOR_BBOX::VARIANCE[2]) * prior_h;

            /* Store the detected box */
            auto bbox = BoundingBox{
                static_cast<int32_t>(class_index),
                kLabelListDet[class_index],
                class_score,
                std::max(0, static_cast<int32_t>((cx - w / 2.0 - pad[0]/2) * scale_w)),
                std::max(0, static_cast<int32_t>((cy - h / 2.0 - pad[1]/2) * scale_h)),
                static_cast<int32_t>(w * scale_w),
                static_cast<int32_t>(h * scale_h)
            };
            BoundingBoxUtils::FixInScreen(bbox, 1280, 720);
            bbox_list.push_back(bbox);
        }
    }

    // /* Adjust bounding box */
    // for (auto& bbox : bbox_list) {
    //     bbox.x += crop_x;  
    //     bbox.y += crop_y;
    // }

    /* NMS */
    std::vector<BoundingBox> bbox_nms_list;
    BoundingBoxUtils::Nms(bbox_list, bbox_nms_list, iouThreshold);

    /*** Draw detection result ***/
    // for (const auto& bbox : bbox_nms_list) {
    //     cv::rectangle(mat, cv::Rect(bbox.x, bbox.y, bbox.w, bbox.h), cv::Scalar(0, 255, 0), 1);
    // }

    // cv::imshow("full", mat);
    // cv::waitKey(1)
    Result result {mat_road, mat_lane, bbox_nms_list};
    return result;
}

Result HybridNetsDetector::detect(cv::Mat &image, const float& confThreshold = 0.4,
                                            const float& iouThreshold = 0.45)
{
    float *blob = nullptr;
    std::vector<int64_t> inputTensorShape {1, 3, -1, -1};
    std::vector<int> pad = this->preprocessing(image, blob, inputTensorShape);

    size_t inputTensorSize = utils::vectorProduct(inputTensorShape);

    std::vector<float> inputTensorValues(blob, blob + inputTensorSize);

    std::vector<Ort::Value> inputTensors;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, inputTensorValues.data(), inputTensorSize,
            inputTensorShape.data(), inputTensorShape.size()
    ));

    std::vector<Ort::Value> outputTensors = this->session.Run(Ort::RunOptions{nullptr},
                                                              inputNames.data(),
                                                              inputTensors.data(),
                                                              1,
                                                              outputNames.data(),
                                                              3);

    cv::Size resizedShape = cv::Size((int)inputTensorShape[3], (int)inputTensorShape[2]);
    Result result = this->postprocessing(image, resizedShape,
                                                         image.size(),
                                                         outputTensors,
                                                         confThreshold, iouThreshold, pad);

    delete[] blob;

    return result;
}
