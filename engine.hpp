#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <vector>
#include <regex>

struct Detection {
    cv::Rect box;
    int class_id;
    float confidence;
};

class YoloInferencer {
public:
    YoloInferencer(std::wstring& modelPath, const char* logid, const char* provider);
    std::vector<Detection> infer(cv::Mat& frame, float conf_threshold, float iou_threshold);

private:
    std::vector<Ort::Value> preprocess(cv::Mat& frame);
    std::vector<Ort::Value> forward(std::vector<Ort::Value>& inputTensors);
    std::vector<Detection> postprocess(std::vector<Ort::Value>& outputTensors, float conf_threshold, float iou_threshold);

    Ort::Env env_{ nullptr };
    Ort::Session session_{ nullptr };

    std::vector<std::string> inputNames_;
    std::vector<std::string> outputNames_;
    std::vector<const char*> inputNamesCStr_;
    std::vector<const char*> outputNamesCStr_;

    Ort::ModelMetadata model_metadata{ nullptr };
    std::unordered_map<std::string, std::string> metadata;

    std::vector<int> imgsz_;
    int stride_ = -1;
    int nc_ = -1;
    int ch_ = 3;

    std::unordered_map<int, std::string> names_;
    std::vector<int64_t> inputTensorShape_;
    std::string task_;

    std::vector<float> inputTensorValues_;

    cv::Size cvSize_;
    cv::Size rawImgSize_;

    // Helper functions, These were stolen and modified from https://github.com/FourierMourier/yolov8-onnx-cpp
    // Same with pretty much everything else

    std::vector<std::string> parseVectorString(const std::string& input) {
        std::regex number_pattern(R"(\d+)");

        std::vector<std::string> result;
        std::sregex_iterator it(input.begin(), input.end(), number_pattern);
        std::sregex_iterator end;

        while (it != end) {
            result.push_back(it->str());
            ++it;
        }

        return result;
    }

    std::vector<int> convertStringVectorToInts(const std::vector<std::string>& input) {
        std::vector<int> result;

        for (const std::string& str : input) {
            try {
                int value = std::stoi(str);
                result.push_back(value);
            }
            catch (const std::invalid_argument& e) {
                throw std::invalid_argument("Bad argument (cannot cast): value=" + str);
            }
            catch (const std::out_of_range& e) {
                throw std::out_of_range("Value out of range: " + str);
            }
        }

        return result;
    }

    std::unordered_map<int, std::string> parseNames(const std::string& input) {
        std::unordered_map<int, std::string> result;

        std::string cleanedInput = input;
        cleanedInput.erase(std::remove(cleanedInput.begin(), cleanedInput.end(), '{'), cleanedInput.end());
        cleanedInput.erase(std::remove(cleanedInput.begin(), cleanedInput.end(), '}'), cleanedInput.end());

        std::istringstream elementStream(cleanedInput);
        std::string element;
        while (std::getline(elementStream, element, ',')) {
            std::istringstream keyValueStream(element);
            std::string keyStr, value;
            if (std::getline(keyValueStream, keyStr, ':') && std::getline(keyValueStream, value)) {
                int key = std::stoi(keyStr);
                result[key] = value;
            }
        }

        return result;
    }

    int64_t vector_product(const std::vector<int64_t>& vec) {
        int64_t result = 1;
        for (int64_t value : vec) {
            result *= value;
        }
        return result;
    }

    const int& DEFAULT_LETTERBOX_PAD_VALUE = 114;

    cv::Mat letterbox(const cv::Mat& image, const cv::Size& newShape, cv::Scalar_<double> color, bool auto_, bool scaleFill, bool scaleUp, int stride) {

        cv::Mat outimage;

        cv::Size shape = image.size();
        float r = std::min(static_cast<float>(newShape.height) / static_cast<float>(shape.height),
            static_cast<float>(newShape.width) / static_cast<float>(shape.width));
        if (!scaleUp)
            r = std::min(r, 1.0f);

        float ratio[2]{ r, r };
        int newUnpad[2]{ static_cast<int>(std::round(static_cast<float>(shape.width) * r)),
                         static_cast<int>(std::round(static_cast<float>(shape.height) * r)) };

        auto dw = static_cast<float>(newShape.width - newUnpad[0]);
        auto dh = static_cast<float>(newShape.height - newUnpad[1]);

        if (auto_)
        {
            dw = static_cast<float>((static_cast<int>(dw) % stride));
            dh = static_cast<float>((static_cast<int>(dh) % stride));
        }
        else if (scaleFill)
        {
            dw = 0.0f;
            dh = 0.0f;
            newUnpad[0] = newShape.width;
            newUnpad[1] = newShape.height;
            ratio[0] = static_cast<float>(newShape.width) / static_cast<float>(shape.width);
            ratio[1] = static_cast<float>(newShape.height) / static_cast<float>(shape.height);
        }

        dw /= 2.0f;
        dh /= 2.0f;

        //cv::Mat outImage;
        if (shape.width != newUnpad[0] || shape.height != newUnpad[1])
        {
            cv::resize(image, outimage, cv::Size(newUnpad[0], newUnpad[1]));
        }
        else
        {
            outimage = image.clone();
        }

        int top = static_cast<int>(std::round(dh - 0.1f));
        int bottom = static_cast<int>(std::round(dh + 0.1f));
        int left = static_cast<int>(std::round(dw - 0.1f));
        int right = static_cast<int>(std::round(dw + 0.1f));


        if (color == cv::Scalar()) {
            color = cv::Scalar(DEFAULT_LETTERBOX_PAD_VALUE, DEFAULT_LETTERBOX_PAD_VALUE, DEFAULT_LETTERBOX_PAD_VALUE);
        }

        cv::copyMakeBorder(outimage, outimage, top, bottom, left, right, cv::BORDER_CONSTANT, color);

        return outimage;
    }

    std::vector<float> fill_blob(cv::Mat& image, std::vector<int64_t>& inputTensorShape) {

        cv::Mat floatImage;

        int inputChannelsNum = inputTensorShape[1];
        int rtype = CV_32FC3;
        image.convertTo(floatImage, rtype, 1.0f / 255.0);

        std::vector<float> blob(floatImage.cols * floatImage.rows * floatImage.channels());
        cv::Size floatImageSize{ floatImage.cols, floatImage.rows };

        // hwc -> chw
        std::vector<cv::Mat> chw(floatImage.channels());
        for (int i = 0; i < floatImage.channels(); ++i)
        {
            chw[i] = cv::Mat(floatImageSize, CV_32FC1, blob.data() + i * floatImageSize.width * floatImageSize.height);
        }
        cv::split(floatImage, chw);

        return blob;
    }

    void clip_boxes(cv::Rect& box, const cv::Size& shape) {
        box.x = std::max(0, std::min(box.x, shape.width));
        box.y = std::max(0, std::min(box.y, shape.height));
        box.width = std::max(0, std::min(box.width, shape.width - box.x));
        box.height = std::max(0, std::min(box.height, shape.height - box.y));
    }

    void clip_boxes(cv::Rect_<float>& box, const cv::Size& shape) {
        box.x = std::max(0.0f, std::min(box.x, static_cast<float>(shape.width)));
        box.y = std::max(0.0f, std::min(box.y, static_cast<float>(shape.height)));
        box.width = std::max(0.0f, std::min(box.width, static_cast<float>(shape.width - box.x)));
        box.height = std::max(0.0f, std::min(box.height, static_cast<float>(shape.height - box.y)));
    }


    void clip_boxes(std::vector<cv::Rect>& boxes, const cv::Size& shape) {
        for (cv::Rect& box : boxes) {
            clip_boxes(box, shape);
        }
    }

    void clip_boxes(std::vector<cv::Rect_<float>>& boxes, const cv::Size& shape) {
        for (cv::Rect_<float>& box : boxes) {
            clip_boxes(box, shape);
        }
    }

    cv::Rect_<float> scale_boxes(const cv::Size& img1_shape, cv::Rect_<float>& box, const cv::Size& img0_shape,
        std::pair<float, cv::Point2f> ratio_pad = std::make_pair(-1.0f, cv::Point2f(-1.0f, -1.0f)), bool padding = true) {

        float gain, pad_x, pad_y;

        if (ratio_pad.first < 0.0f) {
            gain = std::min(static_cast<float>(img1_shape.height) / static_cast<float>(img0_shape.height),
                static_cast<float>(img1_shape.width) / static_cast<float>(img0_shape.width));
            pad_x = roundf((img1_shape.width - img0_shape.width * gain) / 2.0f - 0.1f);
            pad_y = roundf((img1_shape.height - img0_shape.height * gain) / 2.0f - 0.1f);
        }
        else {
            gain = ratio_pad.first;
            pad_x = ratio_pad.second.x;
            pad_y = ratio_pad.second.y;
        }

        //cv::Rect scaledCoords(box);
        cv::Rect_<float> scaledCoords(box);

        if (padding) {
            scaledCoords.x -= pad_x;
            scaledCoords.y -= pad_y;
        }

        scaledCoords.x /= gain;
        scaledCoords.y /= gain;
        scaledCoords.width /= gain;
        scaledCoords.height /= gain;

        // Clip the box to the bounds of the image
        clip_boxes(scaledCoords, img0_shape);

        return scaledCoords;
    }
};