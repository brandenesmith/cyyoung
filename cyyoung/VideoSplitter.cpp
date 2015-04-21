//
//  VideoSplitter.cpp
//  cyyoung
//
//  Created by MATHEW RENY on 4/4/15.
//  Copyright (c) 2015 MATHEW RENY. All rights reserved.
//

#include "VideoSplitter.h"

VideoSplitter::VideoSplitter(cv::string videoFilePath, cv::string outDirPath)
{
    if ('/' != (char)outDirPath[outDirPath.length()-1])
    {
        outDirPath.append("/");
    }

    this->odp = outDirPath;
    this->vfp = videoFilePath;
}

cv::string VideoSplitter::split(FrameNumber start, FrameNumber end)
{
    // Check input for invalid cases.
    if (0 == end)
    {
        throw new std::invalid_argument("The second frame cannot be zero.");
    }
    else if (start >= end)
    {
        throw new std::invalid_argument("The frames are not sorted.");
    }

    cv::VideoCapture vc(this->vfp);

    if (!vc.isOpened())
    {
        printf("Could not open video capture at the path %s.", this->vfp.c_str());
        throw new std::runtime_error("Video does not exist.");
    }

    // Construct the new output path.
    CV_Assert('/' != (char)this->odp[this->odp.length()-1]);
    std::ostringstream path;
    path << this->odp << start << "-" << end << ".mjpg";

    cv::string      p(path.str());
    int             cc4(CV_FOURCC('M','J','P','G'));
    int             fps(30);
    cv::Size        size(720,1280);
    bool            color(true);
    cv::VideoWriter vw(cv::VideoWriter(p,cc4,fps,size,color));

    if (!vw.isOpened())
    {
        printf("Could not open a video writer at the path %s.", p.c_str());
        throw new std::runtime_error("Video write path invalid.");
    }

    for (int i = 0; i < end; i++)
    {
        cv::Mat f;
        vc >> f;

        if (this->displayOutput)
        {
            
        }

        if (i >= start)
        {
            vw << f;
        }
    }

    vw.release();
    vc.release();

    return p;
}