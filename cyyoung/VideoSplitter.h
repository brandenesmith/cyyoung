//
//  VideoSplitter.h
//  cyyoung
//
//  Created by MATHEW RENY on 4/4/15.
//  Copyright (c) 2015 MATHEW RENY. All rights reserved.
//

#ifndef __cyyoung__VideoSplitter__
#define __cyyoung__VideoSplitter__

#include <stdio.h>
#include <opencv2/opencv.hpp>


typedef unsigned int FrameNumber;
typedef float Second;




class VideoSplitter {


private:
    const bool displayOutput = true;


    // Output directory path
    cv::string odp;

    // Video file path
    cv::string vfp;
    
public:
    VideoSplitter(cv::string videoFilePath, cv::string outputDirectoryPath);

    cv::string split(FrameNumber start, FrameNumber end);
};




#endif /* defined(__cyyoung__VideoSplitter__) */
