#include "hybridnets_cpp/utils.h"
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>

float BoundingBoxUtils::CalculateIoU(const BoundingBox& obj0, const BoundingBox& obj1)
{
    int32_t interx0 = (std::max)(obj0.x, obj1.x);
    int32_t intery0 = (std::max)(obj0.y, obj1.y);
    int32_t interx1 = (std::min)(obj0.x + obj0.w, obj1.x + obj1.w);
    int32_t intery1 = (std::min)(obj0.y + obj0.h, obj1.y + obj1.h);
    if (interx1 < interx0 || intery1 < intery0) return 0;

    int32_t area0 = obj0.w * obj0.h;
    int32_t area1 = obj1.w * obj1.h;
    int32_t areaInter = (interx1 - interx0) * (intery1 - intery0);
    int32_t areaSum = area0 + area1 - areaInter;

    return static_cast<float>(areaInter) / areaSum;
}


void BoundingBoxUtils::Nms(std::vector<BoundingBox>& bbox_list, std::vector<BoundingBox>& bbox_nms_list, float threshold_nms_iou, bool check_class_id)
{
    std::sort(bbox_list.begin(), bbox_list.end(), [](BoundingBox const& lhs, BoundingBox const& rhs) {
        //if (lhs.w * lhs.h > rhs.w * rhs.h) return true;
        if (lhs.score > rhs.score) return true;
        return false;
        });

    std::unique_ptr<bool[]> is_merged(new bool[bbox_list.size()]);
    for (size_t i = 0; i < bbox_list.size(); i++) is_merged[i] = false;
    for (size_t index_high_score = 0; index_high_score < bbox_list.size(); index_high_score++) {
        std::vector<BoundingBox> candidates;
        if (is_merged[index_high_score]) continue;
        candidates.push_back(bbox_list[index_high_score]);
        for (size_t index_low_score = index_high_score + 1; index_low_score < bbox_list.size(); index_low_score++) {
            if (is_merged[index_low_score]) continue;
            if (check_class_id && bbox_list[index_high_score].class_id != bbox_list[index_low_score].class_id) continue;
            if (CalculateIoU(bbox_list[index_high_score], bbox_list[index_low_score]) > threshold_nms_iou) {
                candidates.push_back(bbox_list[index_low_score]);
                is_merged[index_low_score] = true;
            }
        }

        bbox_nms_list.push_back(candidates[0]);
    }
}


void BoundingBoxUtils::FixInScreen(BoundingBox& bbox, int32_t width, int32_t height)
{
    bbox.x = (std::max)(0, bbox.x);
    bbox.y = (std::max)(0, bbox.y);
    bbox.w = (std::min)(width - bbox.x, bbox.w);
    bbox.h = (std::min)(width - bbox.y, bbox.h);
}


size_t utils::vectorProduct(const std::vector<int64_t>& vector)
{
    if (vector.empty())
        return 0;

    size_t product = 1;
    for (const auto& element : vector)
        product *= element;

    return product;
}


std::wstring utils::charToWstring(const char* str)
{
    typedef std::codecvt_utf8<wchar_t> convert_type;
    std::wstring_convert<convert_type, wchar_t> converter;

    return converter.from_bytes(str);
}


std::vector<int> utils::letterbox(const cv::Mat& image, cv::Mat& outImage,
                      const cv::Size& newShape = cv::Size(640, 640),
                      const cv::Scalar& color = cv::Scalar(114, 114, 114),
                      bool auto_ = true,
                      bool scaleFill = false,
                      bool scaleUp = true,
                      int stride = 32)
{
    // 1280 720
    // 512 256
    // 0.4 0.355

    // 384 256
    // 0.3 0.355
    // cv::Mat abc;
    // std::cout << 1 << std::endl;
    // cv::resize(image, abc, cv::Size(1280, 720));
    cv::Size shape = image.size();
    // std::cout << shape.width << shape.height << std::endl;
    float r = std::min((float)newShape.height / (float)shape.height,
                       (float)newShape.width / (float)shape.width);
    if (!scaleUp)
        r = std::min(r, 1.0f);

    float ratio[2] {r, r};
    int newUnpad[2] {(int)std::round((float)shape.width * r),
                     (int)std::round((float)shape.height * r)};

    // 455 256

    // 384 216
    auto dw = (float)(newShape.width - newUnpad[0]);
    auto dh = (float)(newShape.height - newUnpad[1]);
    // 57
    std::vector<int> pad;
    // std::cout << "dw "  << dw << " dh "<< dh << '\n';
    pad.push_back(int(dw));
    pad.push_back(int(dh));
    // std::cout << pad.at(0) << pad.at(1) << '\n';


    // 40
    auto_ = true;
    if (auto_)
    {
        dw = (float)((int)dw % stride);
        dh = (float)((int)dh % stride);
    }
    else if (scaleFill)
    {
        dw = 0.0f;
        dh = 0.0f;
        newUnpad[0] = newShape.width;
        newUnpad[1] = newShape.height;
        ratio[0] = (float)newShape.width / (float)shape.width;
        ratio[1] = (float)newShape.height / (float)shape.height;
    }

    dw /= 2.0f;
    dh /= 2.0f;

    // 216 -> 256
    dh = 20.0f;

    // 28.5

    // 20
    if (shape.width != newUnpad[0] && shape.height != newUnpad[1])
    {
        cv::resize(image, outImage, cv::Size(newUnpad[0], newUnpad[1]));
    }

    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));
    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
    // cv::imshow("image", outImage);
    // cv::waitKey(0);
    // std::cout << outImage.size() << std::endl;    
    return pad;
}
