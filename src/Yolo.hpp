#pragma once 
#include <torch/script.h>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>

struct Net_config
{
	float confThreshold; // Confidence threshold
	float nmsThreshold;  // Non-maximum suppression threshold
	std::string modelpath;
	int inpHeight;
	int inpWidth;
};

struct Yolo_detection{
	cv::Rect box;
	int classid;
	float confidence;
};

class YOLOV7
{
public:
	YOLOV7(Net_config config);
	std::vector<cv::Rect> detect(cv::Mat frame);
private:
	std::vector<std::string> class_names;
    torch::jit::script::Module yolo;
	int inpWidth;
	int inpHeight;
	int num_class;

	float confThreshold;
	float nmsThreshold;
};