//
//  OpticalFlow.h
//  cyyoung
//
//  Created by MATHEW RENY on 4/18/15.
//  Copyright (c) 2015 MATHEW RENY. All rights reserved.
//

#ifndef __cyyoung__OpticalFlow__
#define __cyyoung__OpticalFlow__

#include <stdio.h>

#include <opencv2/opencv.hpp>

#define MAX_FEATURES 1000


typedef std::vector< std::pair< cv::Point2f , cv::Point2f > > MotionVec;


class OpticalFlow {
private:
    cv::TermCriteria tc;
    cv::Mat prevFrame;


    std::vector< cv::Point2f > prevFeatures, nextFeatures;

public:

    OpticalFlow(cv::Mat &firstFrame);


    MotionVec feed(cv::Mat &img);


    cv::Mat drawMotion(MotionVec mv, cv::Mat &image);


};



#endif /* defined(__cyyoung__OpticalFlow__) */


