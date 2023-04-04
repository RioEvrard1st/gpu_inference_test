#pragma once
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include "torch2trt.hpp"
#include "cuda_runtime_api.h"
#include "NvInfer.h"
#include <string>

class RektNet{
    public:
        RektNet(std::string filename);
        std::vector<std::pair<int,int>> detect(cv::Mat coneImage, int w, int h);

    private:
        torch::jit::script::Module rektnet;
};
