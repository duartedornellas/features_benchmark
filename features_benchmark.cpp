#include <stdio.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
using namespace cv;
using namespace cv::xfeatures2d;

// Global variable declaration
const Mat img_EuRoC = imread( "Images/EuRoC/1403636579763555584.png", IMREAD_GRAYSCALE );
const Mat img_KITTI = imread( "Images/KITTI/000000.png", IMREAD_GRAYSCALE );
const Mat img_PennCOSYVIO = imread( "Images/PennCOSYVIO/frame_0001.png", IMREAD_GRAYSCALE );
const Mat img_TUMVI = imread( "Images/TUMVI/1520531829251142058.png", IMREAD_GRAYSCALE );

// Function 'mean_time_detectors'
template<class type> float mean_time_detectors(Ptr<type> &detector){
    std::vector<KeyPoint> keypoints_EuRoC,
                          keypoints_KITTI,
                          keypoints_PennCOSYVIO,
                          keypoints_TUMVI;
    auto start = std::chrono::high_resolution_clock::now();
    detector->detect( img_EuRoC, keypoints_EuRoC );
    detector->detect( img_KITTI, keypoints_KITTI );
    detector->detect( img_PennCOSYVIO, keypoints_PennCOSYVIO );
    detector->detect( img_TUMVI, keypoints_TUMVI );
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    int n_keypoints = keypoints_EuRoC.size() +
                      keypoints_KITTI.size() +
                      keypoints_PennCOSYVIO.size() +
                      keypoints_TUMVI.size();
    float mean_runtime = elapsed.count() / n_keypoints;
    std::cout << std::endl
              << "Detector type: " << typeid(type).name() << std::endl
              << "Number of keypoints: " << n_keypoints << '\n'
              << "Elapsed time: " << elapsed.count() << " s\n"
              << "Mean computation time per keypoint: " << mean_runtime << " s\n";
    return mean_runtime;
}

// Function 'mean_time_descriptors'
template<class type> float mean_time_descriptors(Ptr<type> &descriptor){
    Ptr<FastFeatureDetector> detector_FAST = FastFeatureDetector::create();

    std::vector<KeyPoint> keypoints_EuRoC, keypoints_KITTI, keypoints_PennCOSYVIO, keypoints_TUMVI;
    Mat descriptors_EuRoC, descriptors_KITTI, descriptors_PennCOSYVIO, descriptors_TUMVI;

    detector_FAST->detect( img_EuRoC, keypoints_EuRoC );
    detector_FAST->detect( img_KITTI, keypoints_KITTI );
    detector_FAST->detect( img_PennCOSYVIO, keypoints_PennCOSYVIO );
    detector_FAST->detect( img_TUMVI, keypoints_TUMVI );

    auto start = std::chrono::high_resolution_clock::now();
    descriptor->compute( img_EuRoC, keypoints_EuRoC, descriptors_EuRoC );
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    int n_keypoints = keypoints_EuRoC.size() +
                      keypoints_KITTI.size() +
                      keypoints_PennCOSYVIO.size() +
                      keypoints_TUMVI.size();
    float mean_runtime = elapsed.count() / n_keypoints;
    std::cout << std::endl
              << "Descriptor type: " << typeid(type).name() << std::endl
              << "Number of keypoints: " << n_keypoints << '\n'
              << "Elapsed time: " << elapsed.count() << " s\n"
              << "Mean computation time per keypoint: " << mean_runtime << " s\n";
    return mean_runtime;
}

// Function 'main'
int main( int argc, char** argv )
{
    // Check for image loading errors
    if( !img_EuRoC.data || !img_KITTI.data || !img_PennCOSYVIO.data || !img_TUMVI.data )
    { std::cout<< " --(!) Error reading images " << std::endl; return -1; }

    // Initialize feature detectors:
    Ptr<GFTTDetector> detector_GFTT = GFTTDetector::create();
    Ptr<SIFT> detector_SIFT = SIFT::create();
    Ptr<MSER> detector_MSER = MSER::create();
    Ptr<FastFeatureDetector> detector_FAST = FastFeatureDetector::create();
    Ptr<SURF> detector_SURF = SURF::create();
    Ptr<BRISK> detector_BRISK = BRISK::create();
    Ptr<ORB> detector_ORB = ORB::create();

    // Initialize feature descriptors:
    Ptr<SIFT> descriptor_SIFT = SIFT::create();
    Ptr<SURF> descriptor_SURF = SURF::create();
    Ptr<BriefDescriptorExtractor> descriptor_BRIEF = BriefDescriptorExtractor::create();
    Ptr<BRISK> descriptor_BRISK = BRISK::create();
    Ptr<ORB> descriptor_ORB = ORB::create();
    Ptr<FREAK> descriptor_FREAK = FREAK::create();

    // Compute elapsed and mean runtime per keypoint detection (10 iterations)
    std::ofstream results_detectors, results_descriptors;

    results_detectors.open ("results_detectors.txt");
    for(int i=0; i<10; i++){
        results_detectors << mean_time_detectors<GFTTDetector>(detector_GFTT) << '\t'
                          << mean_time_detectors<SIFT>(detector_SIFT) << '\t'
                          << mean_time_detectors<MSER>(detector_MSER) << '\t'
                          << mean_time_detectors<FastFeatureDetector>(detector_FAST) << '\t'
                          << mean_time_detectors<SURF>(detector_SURF) << '\t'
                          << mean_time_detectors<BRISK>(detector_BRISK) << '\t'
                          << mean_time_detectors<ORB>(detector_ORB) << '\n';
    }
    results_detectors.close();

    // Compute elapsed and mean runtime per keypoint description (10 iterations)
    results_descriptors.open ("results_descriptors.txt");
    for(int i=0; i<10; i++){
        results_descriptors << mean_time_descriptors<SIFT>(descriptor_SIFT) << '\t'
                            << mean_time_descriptors<SURF>(descriptor_SURF) << '\t'
                            << mean_time_descriptors<BriefDescriptorExtractor>(descriptor_BRIEF) << '\t'
                            << mean_time_descriptors<BRISK>(descriptor_BRISK) << '\t'
                            << mean_time_descriptors<ORB>(descriptor_ORB) << '\t'
                            << mean_time_descriptors<FREAK>(descriptor_FREAK) << '\n';
    }
    results_descriptors.close();

    return 0;
}
