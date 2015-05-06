//
//  OpticalFlow.h
//  cyyoung
//
//  Created by MATHEW RENY on 4/18/15.
//  Copyright (c) 2015 MATHEW RENY. All rights reserved.
//
//  Warning: Undocumented code below

#ifndef __cyyoung__OpticalFlow__
#define __cyyoung__OpticalFlow__

#include <stdio.h>

#include <opencv2/opencv.hpp>

#define MAX_FEATURES 10000
#define MIN_FEATURE_DISTANCE 0.001

inline float distance(cv::Point_<float> p1, cv::Point_<float> p2)
{
    return sqrtf((p2.x-p1.x)*(p2.x-p1.x) +
                 (p2.y-p1.y)*(p2.y-p1.y));
}

typedef std::vector< std::pair< cv::Point2f , cv::Point2f > > MotionVec;


class OpticalFlow {
private:
    cv::TermCriteria tc;
    cv::Mat prevFrame;


    std::vector< cv::Point2f > prevFeatures, nextFeatures;

public:

    OpticalFlow(cv::Mat &firstFrame, cv::Rect roi);

    MotionVec feed(cv::Mat img);


    cv::Mat drawMotion(MotionVec mv, cv::Mat &image);


};



#endif /* defined(__cyyoung__OpticalFlow__) */


