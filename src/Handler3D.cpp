#include "Handler3D.hpp"
#include <fstream>
#include <ros/ros.h>

cv::Mat parse_line(std::string line ){
    std::string temp = line.substr(line.find("[") + 1, line.find("]") - line.find("[") - 1);
    std::stringstream ss(temp);
    double value;
    std::vector<double> values;
    while (ss >> value)
    {
        values.push_back(value);
        if (ss.peek() == ',') ss.ignore();
    }
    // Create a cv::Mat object from the vector of values
    cv::Mat mat(1,values.size(), CV_64FC1);
    for(int i=0; i < values.size();i++){
        mat.at<double>(0,i) = values[i];
    }
    return mat;
}

void Handler3D::readParamsText(std::string calibrationdata){
    try{
        std::fstream data;
        data.open(calibrationdata);
        std::string l;
        if(data.is_open()){
            while(getline(data, l)){
                if(l == "Left:"){
                    getline(data, l);
                    left_D = parse_line(l);
                    getline(data, l);
                    left_K = parse_line(l);
                    getline(data, l);
                    left_R = parse_line(l);
                    getline(data, l);
                    left_P = parse_line(l);
                }else if(l == "Right:"){
                    getline(data, l);
                    right_D = parse_line(l);
                    getline(data, l);
                    right_K = parse_line(l);
                    getline(data, l);
                    right_R = parse_line(l);
                    getline(data, l);
                    right_P = parse_line(l);
                    getline(data, l);
                    translation_relative = parse_line(l);
                    getline(data, l);
                    rotation_relative = parse_line(l);
                }else{
                    //std::cout << "other line" << "\n";
                }
            } 
            data.close();
        }
    }catch (cv::Exception &e){
		ROS_INFO_STREAM("No camera paramters found" << e.what());
	}
    //printParams();
}

void Handler3D::printParams(){
    std::cout << "D_left: " << left_D << std::endl;
    std::cout << "K_left: " << left_K << std::endl;
    std::cout << "R_left: " << left_R << std::endl;
    std::cout << "P_left: " << left_P << std::endl;
    std::cout << "D_right: " << right_D << std::endl;
    std::cout << "K_right: " << right_K << std::endl;
    std::cout << "R_right: " << right_R << std::endl;
    std::cout << "P_right: " << right_P << std::endl;
    std::cout << "translation relative: " << translation_relative << std::endl;
    std::cout << "rotation relative: " << rotation_relative << std::endl;
}

Handler3D::Handler3D(std::string filename){
    ROS_INFO_STREAM("Reading camera parameters from: " << filename << "\n");
    readParamsText(filename);
    
    //define 3d points wrt to the base of the cone which is taken as the world frame
    this->modelPoints.push_back(cv::Point3f(0,-10,0));
    this->modelPoints.push_back(cv::Point3f(0,-7,10));
    this->modelPoints.push_back(cv::Point3f(0,-4,20));
    this->modelPoints.push_back(cv::Point3f(0,0,30));
    this->modelPoints.push_back(cv::Point3f(0,4,20));
    this->modelPoints.push_back(cv::Point3f(0,7,10));
    this->modelPoints.push_back(cv::Point3f(0,10,0));
}

void Handler3D::solvePnP(std::vector<std::pair<int,int>> kps, int left, int top)
{
    //corresponding 2d points
    std::vector<cv::Point2f> imagePoints;
    for(std::pair<int,int> & p : kps){
        imagePoints.push_back(cv::Point2f(p.first+left,p.second+top));
    }

    cv::Mat cam_mat = this->left_K.reshape(0,3); 

    cv::Mat rvec, tvec;
    cv::solvePnP(this->modelPoints, imagePoints, cam_mat, this->left_D, rvec, tvec);
    //cv::solvePnPRansac(this->modelPoints, imagePoints, cam_mat, this->left_D, rvec, tvec);

    // Print results
    std::cout << "Rotation vector:" << std::endl << rvec << std::endl;
    std::cout << "Translation vector:" << std::endl << tvec << std::endl;
}