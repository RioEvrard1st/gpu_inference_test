#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include "Yolo.hpp"
#include "RektNet.hpp"
#include "Handler3D.hpp"
#include <chrono>
#include <string>

// define YOLO
static std::string yolo_path = "/home/rio/thesis_ws/src/inference_package/networks/weights_yolo.torchscript";
static int inputH = 640;
static int inputW = 640;
static Net_config YOLOV7_nets = { 0.8, 0.6, yolo_path, inputH, inputW};
static YOLOV7 yolo(YOLOV7_nets);

// define RektNet
static std::string rektnet_path = "/home/rio/thesis_ws/src/inference_package/networks/weights_rektnet.pt";
static RektNet rektnet{rektnet_path};

// define 3D handler (PnP)
static std::string calibration_path{"/home/rio/thesis_ws/src/inference_package/camera_parameters/"};
static Handler3D handler3d(calibration_path+"calibration.txt");

static int counter=1;

// Draw the predicted bounding box (for visualization purposes only)
void drawBoxPred(float conf, int left, int top, int right, int bottom, cv::Mat& frame, int classid)   
{
	//Draw a rectangle displaying the bounding box
	rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255), 2);

	//Display the label at the top of the bounding box
    std::string label = cv::format("%.2f", conf);
    int baseLine;
	cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = cv::max(top, labelSize.height);
	//rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
	putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 255), 1);
}
// draw the predicted keypoints
void drawKptPred( cv::Mat& frame, std::vector<std::pair<int, int>> kps, int left, int top)
{
	for(std::pair<int, int> p: kps){
		cv::circle(frame,cv::Point2d(p.first+left, p.second+top),2, cv::Scalar(255, 0, 0), -1);
	}
}

void imageCallback(const sensor_msgs::ImageConstPtr & msg){
    try{
        // convert the msg to cv mat object
        cv::Mat img = cv_bridge::toCvShare(msg,"rgb8")->image;
        //cv::Mat img3 = img;

        // yolo inference
        //auto t1 = std::chrono::high_resolution_clock::now();
        std::vector<cv::Rect> boxes =  yolo.detect(img);   
        //auto t2 = std::chrono::high_resolution_clock::now();
        //std::chrono::duration<double, std::milli> ms_double = t2 - t1;
        //std::cout << ms_double.count() << "ms for yolo inference\n";

        // rektnet inference
        for(cv::Rect b : boxes){
            
            // crop the image
	        cv::Mat coneImage = img(b);

            // detect keypoints of the given bounding box
            //t1 = std::chrono::high_resolution_clock::now();
            std::vector<std::pair<int,int>> kps = rektnet.detect(coneImage, b.width, b.height);
            //t2 = std::chrono::high_resolution_clock::now();
            //std::chrono::duration<double, std::milli> ms_double = t2 - t1;
            //std::cout << ms_double.count() << "ms for rektnet inference\n";

            //t1 = std::chrono::high_resolution_clock::now();
            handler3d.solvePnP(kps, b.x, b.y);
            //t2 = std::chrono::high_resolution_clock::now();
            //std::chrono::duration<double, std::milli> ms_double = t2 - t1;
            //std::cout << ms_double.count() << "ms for PnP calculation\n";
            
            // draw bounding boxes and keypoints
            drawBoxPred(1, b.x, b.y, b.width+b.x, b.height+b.y, img, 2);
            drawKptPred(img, kps, b.x, b.y);
        }
        cv::imwrite("outputs/box_kpt_pred/detection"+std::to_string(counter++)+".jpeg",img);
    }
    catch(cv_bridge::Exception& e) {
        ROS_ERROR("Could not convert from '%s' to 'rgb8'.", msg->encoding.c_str());
    }
}

int main(int argc, char ** argv){
    ros::init(argc, argv, "inference_node");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber sub = it.subscribe("left/image_raw", 1, imageCallback);
    ros::spin();
    return 0;
}