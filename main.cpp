#include <iostream>
#include "opencv2/opencv.hpp"
#include "openvino/openvino.hpp"

int main(int argc, char const *argv[])
{
    /* code */
    std::cout << "Hello World" << std::endl;
    cv::Mat dog = cv::imread("C:/Users/brightro99/Desktop/Workspace/OpenVINO_2024/images/dog.jpg");

    ov::Core core;
    
    return 0;
}
