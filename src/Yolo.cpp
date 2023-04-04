#include "Yolo.hpp"

YOLOV7::YOLOV7(Net_config config)
{
	this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;

    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        yolo = torch::jit::load(config.modelpath);    
        std::cout << "YOLO model loaded\n";
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the YOLO model\n";
    }
	
    class_names.push_back("blue");
    class_names.push_back("large orange");
    class_names.push_back("small orange");
    class_names.push_back("other orange");
    class_names.push_back("yellow");
	this->num_class = class_names.size();
	this->inpHeight = config.inpHeight;
	this->inpWidth = config.inpWidth;
}

std::vector<cv::Rect> YOLOV7::detect(cv::Mat frame)
{
    // save size relations of initial frame
    float ratioh = (float) frame.rows / this->inpHeight;
    float ratiow = (float) frame.cols / this->inpWidth;

	// preprocess image
    cv::resize(frame,frame,cv::Size(640,640)); // resize
    cv::cvtColor(frame ,frame, cv::COLOR_RGB2BGR ); //opencv uses bgr
    frame.convertTo(frame, CV_32FC3, 1.0 / 255.0); // normalize

    // define as libtorch input tensor
    auto tensor = torch::from_blob(frame.data, { frame.rows, frame.cols, 3 }, torch::kFloat32);
    tensor = tensor.permute({ (2),(0),(1) });
    tensor.unsqueeze_(0); // add batch dim (an inplace operation just like in pytorch)
    std::vector<torch::jit::IValue> input = std::vector<torch::jit::IValue>{tensor};
    
    // forward
    auto output = yolo.forward(input).toTuple();
    at::Tensor preds = output->elements()[0].toTensor();

    // filter boxes
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;
	std::vector<int> classIds;
    const int proposals = 25200; // default for 640x640 images
    for(int i = 1; i <= proposals; i++){
        at::Tensor pred = preds.slice(1,i-1,i);
        float box_score = pred.index({0,0,4}).item<float>();
        if(box_score > this->confThreshold){ // larger than confidence threshold
            at::Tensor scores = pred.slice(2,5,10);
            float max = torch::max(scores).item<float>();
            at::Tensor max_index = (scores == max).nonzero();
            max *= box_score;
               
            if(max > this->confThreshold) { // conf threshold
                const int class_id = max_index.index({0,2}).item<int>();
                float cx = pred.index({0,0,0}).item<float>() * ratiow;
                float cy = pred.index({0,0,1}).item<float>() * ratioh;
                float w = pred.index({0,0,2}).item<float>() * ratiow;
                float h = pred.index({0,0,3}).item<float>() * ratioh;

                int left = int(cx - 0.5*w);
                int top = int(cy - 0.5*h);
                cv::Rect b = cv::Rect(left, top, (int)(w), (int)(h));
                confidences.push_back((float)max);
                boxes.push_back(b);
                classIds.push_back(class_id);
            } 
        }
    }
	std::vector<int> indices;	
    std::vector<cv::Rect> filtered_boxes;
	cv::dnn::NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
	{
		filtered_boxes.push_back(boxes[indices[i]]);
	}

    // filter 8 largest boxes not fallen over
    std::sort(filtered_boxes.begin(), filtered_boxes.end(), [](const cv::Rect& r1, const cv::Rect& r2) {return r1.height > r2.height;});

    std::vector<cv::Rect> sorted_boxes;
    int numBoxesAdded = 0;
    for (const auto& box : filtered_boxes) {
        if (box.width <= box.height && numBoxesAdded < 8) {
            sorted_boxes.push_back(box);
            numBoxesAdded++;
        }
    }

	return sorted_boxes;
}
