#include "RektNet.hpp"
#include <algorithm>

RektNet::RektNet(std::string filename) {
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        rektnet = torch::jit::load(filename);    
        std::cout << "RektNet model loaded\n";
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the RektNet model\n";
    }


    // Convert the PyTorch model to a TensorRT engine
    trtorch::core::conversion::ConvertGraphParams params;
    params.op_precision = torch::kFloat;
    params.use_onnx = false;
    params.device = torch::kCUDA;
    params.max_batch_size = 1;
    nvinfer1::ICudaEngine* engine = trtorch::core::conversion::ConvertGraph(module, params);

    // Serialize the engine and save it to a file
    nvinfer1::IHostMemory* serialized_engine = engine->serialize();
    std::ofstream engine_file(filename, std::ios::binary);
    engine_file.write(static_cast<const char*>(serialized_engine->data()), serialized_engine->size());
}

// input frame is the cropped image
std::vector<std::pair<int,int>> RektNet::detect(cv::Mat coneImage, int w, int h){

    // pre-process the image
    cv::resize(coneImage,coneImage,cv::Size(80,80));
    cv::cvtColor(coneImage, coneImage , cv::COLOR_RGB2BGR );//opencv uses bgr
    coneImage.convertTo(coneImage, CV_32FC3, 1.0 / 255.0);

    // convert to libtorch input tensor
    auto tensor = torch::from_blob(coneImage.data, { coneImage.rows, coneImage.cols, 3 }, torch::kFloat32);
    tensor = tensor.permute({ (2),(0),(1) });
    tensor.unsqueeze_(0); //add batch dim
    std::vector<torch::jit::IValue> input = std::vector<torch::jit::IValue>{tensor};
    
    // forward
    auto output = rektnet.forward(input).toTuple();
    auto points = output->elements()[1].toTensor();

    // extract coordinates
    std::vector<std::pair<int,int>> keypoints;
    for(int i=0; i < 7 ; i++){
        auto cord = points[0][i];
        double x = cord[0].item<double>();
        double y = cord[1].item<double>();
        keypoints.emplace_back(std::pair<int,int>(x*w, y*h));
    }

    // sort keypoints
    std::sort(keypoints.begin(), keypoints.end(), [](const std::pair<int,int> &a, const std::pair<int,int> &b) {return a.first < b.first;});

    return keypoints;
}