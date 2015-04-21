//
//  PitchComposite.h
//  cyyoung
//
//  Created by MATHEW RENY on 3/29/15.
//  Copyright (c) 2015 MATHEW RENY. All rights reserved.
//

#ifndef __cyyoung__PitchComposite__
#define __cyyoung__PitchComposite__

#include <stdio.h>
#include <opencv2/opencv.hpp>

typedef enum {
    PositionPost = 0,
    PositionTee = 1,
    PositionRelease = 2,
    PositionFollow = 3,
    // Helper positions.
    PositionCount = 4,
    PositionFirst = PositionPost,
    PositionLast = PositionFollow,
} Position;


class PitchComposite {

private:
    int imageWidth, imageHeight, imageChannelType;
    cv::Mat images[PositionCount];
public:
    PitchComposite(const int inputWidth = 1280, const int inputHeight = 720, const int inputChannelType = CV_8UC3);

    void setPosition(cv::Mat image, Position p);
    cv::Mat compose();
};


#endif /* defined(__cyyoung__PitchComposite__) */
