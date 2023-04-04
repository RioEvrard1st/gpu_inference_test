#pragma once
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

class Handler3D{
    
    public:
        Handler3D() = default;
        Handler3D(std::string filename);
        void readParamsText(std::string calibrationfile);
        //void readParamsYaml(std::string file_r, std::string file_l);
        void solvePnP(std::vector<std::pair<int,int>> kps, int left, int top);
        void printParams();

    private:
        std::vector<cv::Point3f> modelPoints;

        cv::Mat left_D;
        cv::Mat left_K;
        cv::Mat left_R;
        cv::Mat left_P;

        cv::Mat right_D;
        cv::Mat right_K;
        cv::Mat right_R;
        cv::Mat right_P;

        cv::Mat rotation_relative; //right to left
        cv::Mat translation_relative;
};