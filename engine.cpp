#include "engine.hpp"

YoloInferencer::YoloInferencer(std::wstring& modelPath, const char* logid, const char* provider)
    : env_(ORT_LOGGING_LEVEL_WARNING, logid) {

    // Set session options
    Ort::SessionOptions sessionOptions;
    if (strcmp(provider, "CUDA") == 0) {
        OrtCUDAProviderOptions cudaOption;
        sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
    }
    session_ = Ort::Session(env_, modelPath.c_str(), sessionOptions);

    // Aquire input names
    std::vector<Ort::AllocatedStringPtr> inputNodeNameAllocatedStrings;
    Ort::AllocatorWithDefaultOptions input_names_allocator;
    auto inputNodesNum = session_.GetInputCount();
    for (int i = 0; i < inputNodesNum; i++) {
        auto input_name = session_.GetInputNameAllocated(i, input_names_allocator);
        inputNodeNameAllocatedStrings.push_back(std::move(input_name));
        inputNames_.push_back(inputNodeNameAllocatedStrings.back().get());
    }

    // Convert input names to cstr
    for (const std::string& name : inputNames_) {
        inputNamesCStr_.push_back(name.c_str());
    }

    // Aquire output names
    std::vector<Ort::AllocatedStringPtr> outputNodeNameAllocatedStrings;
    Ort::AllocatorWithDefaultOptions output_names_allocator;
    auto outputNodesNum = session_.GetOutputCount();
    for (int i = 0; i < outputNodesNum; i++)
    {
        auto output_name = session_.GetOutputNameAllocated(i, output_names_allocator);
        outputNodeNameAllocatedStrings.push_back(std::move(output_name));
        outputNames_.push_back(outputNodeNameAllocatedStrings.back().get());
    }

    // Convert output names to cstr
    for (const std::string& name : outputNames_) {
        outputNamesCStr_.push_back(name.c_str());
    }

    // Aquire model metadata
    model_metadata = session_.GetModelMetadata();

    Ort::AllocatorWithDefaultOptions metadata_allocator;
    std::vector<Ort::AllocatedStringPtr> metadataAllocatedKeys = model_metadata.GetCustomMetadataMapKeysAllocated(metadata_allocator);
    std::vector<std::string> metadata_keys;
    metadata_keys.reserve(metadataAllocatedKeys.size());

    for (const Ort::AllocatedStringPtr& allocatedString : metadataAllocatedKeys) {
        metadata_keys.emplace_back(allocatedString.get());
    }

    // Parse metadata
    for (const std::string& key : metadata_keys) {
        Ort::AllocatedStringPtr metadata_value = model_metadata.LookupCustomMetadataMapAllocated(key.c_str(), metadata_allocator);
        if (metadata_value != nullptr) {
            auto raw_metadata_value = metadata_value.get();
            metadata[key] = std::string(raw_metadata_value);
        }
    }

    // Find the input size of the model
    auto imgsz_item = metadata.find("imgsz");
    if (imgsz_item != metadata.end()) {
        // parse it and convert to int iterable
        std::vector<int> imgsz = convertStringVectorToInts(parseVectorString(imgsz_item->second));
        if (imgsz_.empty()) {
            imgsz_ = imgsz;
        }
    }
    else {
        std::cerr << "Warning: Cannot get imgsz value from metadata" << std::endl;
    }

    // For yolo this is normally 32 but get it anyway
    auto stride_item = metadata.find("stride");
    if (stride_item != metadata.end()) {
        // parse it and convert to int iterable
        int stride = std::stoi(stride_item->second);
        if (stride_ == -1) {
            stride_ = stride;
        }
    }
    else {
        std::cerr << "Warning: Cannot get stride value from metadata" << std::endl;
    }

    // For the names of the classes
    auto names_item = metadata.find("names");
    if (names_item != metadata.end()) {
        // parse it and convert to int iterable
        std::unordered_map<int, std::string> names = parseNames(names_item->second);
        std::cout << "***Names from metadata***" << std::endl;
        for (const auto& pair : names) {
            std::cout << "Key: " << pair.first << ", Value: " << pair.second << std::endl;
        }
        // set it here:
        if (names_.empty()) {
            names_ = names;
        }
    }
    else {
        std::cerr << "Warning: Cannot get names value from metadata" << std::endl;
    }

    // Determine the task (We want detect)
    auto task_item = metadata.find("task");
    if (task_item != metadata.end()) {
        std::string task = std::string(task_item->second);

        if (task_.empty()) {
            task_ = task;
        }
    }
    else {
        std::cerr << "Warning: Cannot get task value from metadata" << std::endl;
    }

    // Aquire number of classes
    if (nc_ == -1 && names_.size() > 0) {
        nc_ = names_.size();
    }
    else {
        std::cerr << "Warning: Cannot get nc value from metadata (probably names wasn't set)" << std::endl;
    }

    // Setup the desired input shape
    if (!imgsz_.empty() && inputTensorShape_.empty())
    {
        inputTensorShape_ = { 1, ch_, imgsz_[0], imgsz_[1] };
    }

    // Setup the CV resizer
    if (!imgsz_.empty())
    {
        cvSize_ = cv::Size(imgsz_[1], imgsz_[0]);
    }
}

// Destructor for the class
YoloInferencer::~YoloInferencer() {
    // The Ort::Session and other Ort:: objects will automatically release resources upon destruction
    // due to their RAII design.
}

// This function does the preprocessing of the image and returns the tensor
std::vector<Ort::Value> YoloInferencer::preprocess(cv::Mat& frame) {

    // This isn't actually used until the postprocess function
    // So if u multithread in the future, this will have to move
    rawImgSize_ = frame.size();

    cv::Mat coloured_frame;
    cv::cvtColor(frame, coloured_frame, cv::COLOR_BGR2RGB);  // Convert to the RGB color space (I think this is correct but dont quote me)

    const bool auto_ = false;
    const bool scalefill_ = false;
    cv::Mat letterbox_image = letterbox(coloured_frame, cvSize_, cv::Scalar(), auto_, scalefill_, true, stride_);

    std::vector<float> blob = fill_blob(letterbox_image, inputTensorShape_);

    int64_t inputTensorSize = vector_product(inputTensorShape_);

    inputTensorValues_.resize(inputTensorSize); // Use a member variable to keep it in scope
    std::copy(blob.begin(), blob.begin() + inputTensorSize, inputTensorValues_.begin());

    std::vector<Ort::Value> inputTensors;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues_.data(), inputTensorSize, inputTensorShape_.data(), inputTensorShape_.size()));

    return inputTensors;
}

// This function does the forward pass and returns the tensor
std::vector<Ort::Value> YoloInferencer::forward(std::vector<Ort::Value>& inputTensors) {
    return session_.Run(Ort::RunOptions{ nullptr }, inputNamesCStr_.data(), inputTensors.data(), inputNamesCStr_.size(), outputNamesCStr_.data(), outputNamesCStr_.size());
}

// This function does the postprocessing of the output and returns the detections
std::vector<Detection> YoloInferencer::postprocess(std::vector<Ort::Value>& outputTensors, float conf_threshold, float iou_threshold) {

    // ngl have this shit is voodoo but it works thanks again @FourierMourier from https://github.com/FourierMourier/yolov8-onnx-cpp

    float* data = outputTensors[0].GetTensorMutableData<float>();
    std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

    cv::Mat output0 = cv::Mat(cv::Size((int)outputShape[2], (int)outputShape[1]), CV_32F, data).t();  // [bs, features, preds_num]=>[bs, preds_num, features]

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    int data_width = nc_ + 4;
    int rows = output0.rows;
    float* pdata = (float*)output0.data;

    for (int i = 0; i < rows; ++i) {
        cv::Mat scores(1, nc_, CV_32FC1, pdata + 4);
        double max_conf;
        cv::Point class_id;

        minMaxLoc(scores, nullptr, &max_conf, nullptr, &class_id);

        if (max_conf > conf_threshold) {
            class_ids.push_back(class_id.x);
            confidences.push_back((float)max_conf);

            float out_w = pdata[2];
            float out_h = pdata[3];
            float out_left = std::max((pdata[0] - 0.5f * out_w), 0.0f);
            float out_top = std::max((pdata[1] - 0.5f * out_h), 0.0f);

            cv::Rect_<float> bbox = cv::Rect_<float>(out_left, out_top, (out_w + 0.5), (out_h + 0.5));
            cv::Rect_<float> scaled_bbox = scale_boxes(cvSize_, bbox, rawImgSize_);

            boxes.push_back(scaled_bbox);
        }

        pdata += data_width;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, iou_threshold, nms_result);

    std::vector<Detection> detections;
    for (int idx : nms_result) {
        boxes[idx] &= cv::Rect(0, 0, rawImgSize_.width, rawImgSize_.height);

        Detection detection;
        detection.class_id = class_ids[idx];
        detection.confidence = confidences[idx];
        detection.box = boxes[idx];

        detections.push_back(detection);
    }

    return detections;
}

// This function does the whole inference, and is publicly accessible, acting as a main
std::vector<Detection> YoloInferencer::infer(cv::Mat& frame, float conf_threshold, float iou_threshold) {

    std::vector<Ort::Value> inputTensors = preprocess(frame);

    std::vector<Ort::Value> outputTensors = forward(inputTensors);

    std::vector<Detection> detections = postprocess(outputTensors, conf_threshold, iou_threshold);

    return detections;
}