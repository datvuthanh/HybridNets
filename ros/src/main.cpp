#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ros/package.h>
#include <math.h>
#include <limits>

#include "hybridnets_cpp/utils.h"
#include "hybridnets_cpp/detector.h"
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
// #include <opencv2/videoio.hpp>  // Video write
#include <std_msgs/MultiArrayLayout.h>
#include <std_msgs/MultiArrayDimension.h>
#include <std_msgs/Int16MultiArray.h>
#include <std_msgs/UInt8.h>
#include <sensor_msgs/LaserScan.h>
#include <chrono>
#include <thread>
#define PI 3.14159265

// rosrun image_transport republish compressed in:=/zed2i/zed_node/left_raw/image_raw_color raw out:=/raw_image



image_transport::Publisher pub_road;
image_transport::Publisher pub_lane;
ros::Publisher pub_objects;
ros::Publisher pub_num_objects;
ros::Publisher pub_scan;
HybridNetsDetector detector {nullptr};
const float confThreshold = 0.3;
const float iouThreshold = 0.2;
cv::Mat depth;


void imageCallback(const sensor_msgs::ImageConstPtr& msg){
    try
    {
        cv::Mat img = cv_bridge::toCvShare(msg, "bgr8")->image;
        // cv::resize(img, img, cv::Size(384, 216));
        // cv::imwrite("calib.jpg", img);
        //cv::Size shape = img.size();
        //std::cout << shape.width << shape.height << std::endl;
        cv::resize(img, img, cv::Size(1280, 720));

        // cv::imwrite("original.jpg", img);
        // auto start = std::chrono::steady_clock::now();
        Result result = detector.detect(img, confThreshold, iouThreshold);

        // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        // std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "[ms]" << std::endl;

        // std_msgs::Int16MultiArray arr;
        // arr.data.clear();
        // arr.layout.dim.push_back(std_msgs::MultiArrayDimension());
        // arr.layout.dim[0].label = "nboxes";
        // arr.layout.dim[0].size = result.boxes.size();
        // arr.layout.dim[0].stride = result.boxes.size()*4;
        // arr.layout.dim.push_back(std_msgs::MultiArrayDimension());
        // arr.layout.dim[1].label = "box";
        // arr.layout.dim[1].size = 4;
        // arr.layout.dim[1].stride = 4;
        // arr.layout.data_offset = 0;

        // for (auto r : result.boxes) {
        //     // cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
        //     // cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        //     arr.data.push_back(r.x);
        //     arr.data.push_back(r.y);
        //     arr.data.push_back(r.w);
        //     arr.data.push_back(r.h);
        // }
        // pub_objects.publish(arr);
        std_msgs::UInt8 num_objects;
        num_objects.data =  result.boxes.size();
        // std::cout << num_objects.data << std::endl;
        pub_num_objects.publish(num_objects);
        pub_road.publish(cv_bridge::CvImage(std_msgs::Header(), "mono8", result.road).toImageMsg());
        // pub_lane.publish(cv_bridge::CvImage(std_msgs::Header(), "mono8", result.lane).toImageMsg());

        cv::Mat src = result.road;
        // cv::Mat src = result.lane;
        
        src = src(cv::Rect(0, 20, 384, 216));
        cv::Mat mask = cv::Mat(src.size(), CV_8UC1, cv::Scalar(0));
        cv::Point pt1 {0, 216};   // 10, 216
        cv::Point pt2 {0, 88};   // 135, 108
        cv::Point pt3 {384, 88};  // 249, 108
        cv::Point pt4 {384, 216};  // 374, 21
        std::vector<std::vector<cv::Point>> pts = { { pt1, pt2, pt3, pt4 } };
        cv::fillPoly(mask, pts, cv::Scalar(255));
        // for (auto box: result.boxes){
        //     // std::cout << box.x << " " << box.y << " " << box.w << " " << box.h << std::endl;
        //     mask(cv::Rect(box.x / 3.3, box.y / 3.3, box.w / 3.3, box.h / 3.3)) = 0;
        // }

        cv::Mat src2;
        src.copyTo(src2, mask);
        
        cv::Point2f src_vertices[4];
        // magic numbers, idc bro
        src_vertices[0] = cv::Point(0, 216);   // 0, 216
        src_vertices[1] = cv::Point(0, 88);   // 132, 108
        src_vertices[2] = cv::Point(384, 88);   // 252, 108
        src_vertices[3] = cv::Point(384, 216);   // 384, 216
        cv::Point2f dst_vertices[4];
        dst_vertices[0] = cv::Point(180, 200);
        dst_vertices[1] = cv::Point(0, 0);
        dst_vertices[2] = cv::Point(390, 0);
        dst_vertices[3] = cv::Point(210, 200);
        cv::Mat M = cv::getPerspectiveTransform(src_vertices, dst_vertices);
        cv::Mat dst(200, 390, CV_8UC1);

        // TEST
        // cv::Mat test(200, 390, CV_8UC3);
        // cv::resize(img, img, cv::Size(384, 216));
        // cv::warpPerspective(img, test, M, test.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT);
        // cv::imshow("BEV", test);
        // cv::waitKey(1);

        cv::warpPerspective(src2, dst, M, dst.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT);

        // int erosion_size = 5;
        // cv::Mat element = cv::getStructuringElement( cv::MORPH_RECT,
        //                 cv::Size( 2*erosion_size + 1, 2*erosion_size+1 ),
        //                 cv::Point( erosion_size, erosion_size ) );
        // cv::erode(src2, src2, element);
        // cv::morphologyEx(src2, src2, cv::MORPH_CLOSE, element);
        std::vector<std::vector<cv::Point> > contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours( dst, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE );
        // cv::findContours( src2, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE );

        const int num_angles = 111; // horizontal FOV = 110
        std::array<float, num_angles> dist = {std::numeric_limits<float>::infinity()}; // distances on different angles (simulating a laser scan)
        for (int i=0; i< num_angles; i++)
        {
            dist[i]=std::numeric_limits<float>::infinity();
        }
        //std::array<float> dist(num_angles,std::numeric_limits<float>::infinity());

        float cur_depth = 0;
        int cur_angle = 0;
        // const float img2ego = 110.0/180.0;  // an image coordinate is 180 degree, while real FOV is 110
        const int angle_offset = 55; // convert from [-55, 55] to [0, 110]
        // const int angle_offset = 90; // convert from [-90, 90] to [0, 180]
        // const float x_scale = 1280.0 / 384.0;
        // const float y_scale = 720.0 / 216.0;
        int x = 0;
        int y = 0;

        for (auto contour: contours){
            for (auto point: contour){
                // x = round(point.x * x_scale);
                // y = round(point.y * y_scale);
                x = point.x;
                y = point.y;
                if (y >  190 || y < 20) continue;
                // std::cout<<"IMAGE X" << x<<std::endl;
                // std::cout<<"IMAGE Y" << y<<std::endl;

                float real_x  = (200 - y) / 200.0 * 10 + 0.3;     
                float real_y  = (195 - x) / 390.0 * 19.5;
                // cur_depth = static_cast<float>(depth.at<ushort>(y, x)) / 1000;
                // cur_depth = sqrt(pow(x - 192, 2) + pow(y - vertical_height, 2))/100 * 5/9;  // 5/9 is a constant to match distance between simulator and real life  
                cur_depth = sqrt(pow(real_x, 2) + pow(real_y, 2));
                // std::cout<<"X" << real_x<<std::endl;
                // std::cout<<"Y" << real_y<<std::endl;
                
                // std::cout<<cur_depth<<std::endl;
                cur_angle = std::atan(static_cast<float>(real_y) / static_cast<float>(real_x)) * 180 / PI + angle_offset;
                // std::cout<<"ANGLE" <<cur_angle<<std::endl;
                if (cur_angle < 0 || cur_angle > 110) continue;


                // if (cur_angle == 175) {
                //     std::cout<<"CUR DEPTH:"<<cur_depth<<std::endl;
                // }
                if ((cur_depth < dist[cur_angle])){
                    dist[cur_angle] = cur_depth;
                    // }
                }
            }
        }
        pub_lane.publish(cv_bridge::CvImage(std_msgs::Header(), "mono8", dst).toImageMsg());

        // https://github.com/AbhiRP/Fake-LaserScan/blob/master/src/laserscan_publisher.cpp
        sensor_msgs::LaserScan scan;
        scan.header.stamp = ros::Time::now();
        scan.header.frame_id = "odom";
        scan.angle_min = -55 * (PI /180);
        scan.angle_max = 55 * (PI / 180);
        scan.angle_increment = 1 * (PI / 180);
        scan.time_increment = (1/50) / (111);
        scan.range_min = 0.02;
        scan.range_max = 100.0;
        std::vector<float> v (std::begin(dist), std::end(dist));
        scan.ranges = v;

        // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        // std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "[ms]" << std::endl;
        pub_scan.publish(scan);
        // cv::imshow("contours", drawing);
        // cv::imshow("result", testing);
        // cv::imwrite("testing.jpg", testing);
        // cv::waitKey(1);
        // int count = 0;
        // for (auto& i: dist){
        //     cout<<count << ":" << i <<endl;
        //     count++;
        // }
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}

void depthCallback(const sensor_msgs::ImageConstPtr& msg){
    try
    {
        // std::cout<<::depth.size()<<::depth.type()<<std::endl;
        ::depth = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::TYPE_16UC1)->image;
        // std::cout<<::depth.size()<<::depth.type()<<std::endl;
        // cv::imwrite("depth.jpg", ::depth);

        // std::this_thread::sleep_for(std::chrono::milliseconds(5000));
        // cv::imwrite("depth.png", img);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}

int main(int argc, char* argv[])
{
    ros::init(argc, argv, "hybridnets");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::TransportHints hints("compressed");
    // RAW
    // image_transport::Subscriber sub = it.subscribe("/raw_image", 1, imageCallback);
    // COMPRESSED
    // image_transport::Subscriber sub = it.subscribe("/zed2i/zed_node/left_raw/image_raw_color", 1, imageCallback, ros::VoidPtr(), hints);
    image_transport::Subscriber sub = it.subscribe("/camera", 1, imageCallback, ros::VoidPtr(), hints);
    // image_transport::Subscriber sub2 = it.subscribe("/zed2i/zed_node/depth/depth_registered", 1, depthCallback);
    pub_road = it.advertise("road", 0);
    pub_lane = it.advertise("lane", 0);
    // pub_objects = nh.advertise<std_msgs::Int16MultiArray>("objects", 0);
    pub_num_objects = nh.advertise<std_msgs::UInt8>("num_objects", 0);
    pub_scan = nh.advertise<sensor_msgs::LaserScan>("scan", 0);
    std::string path = ros::package::getPath("hybridnets_cpp");
    std::string modelPath = path + "/hybridnets_dynamic_sim_full11.onnx";
    bool isGPU = true;
    detector = HybridNetsDetector(modelPath, isGPU, cv::Size(384, 256));
    std::cout << "Model was initialized." << std::endl;

    ros::MultiThreadedSpinner spinner(2); // Use 2 threads
    spinner.spin(); // spin() will not return until the node has been shutdown

    return 0;
}
