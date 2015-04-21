//
//  VideoSplitterTests.cpp
//  cyyoung
//
//  Created by MATHEW RENY on 4/4/15.
//  Copyright (c) 2015 MATHEW RENY. All rights reserved.
//

#include "VideoSplitterTests.h"

#include "VideoSplitter.h"


bool test1SplitOn1Video()
{
    srand((int)time(NULL));

//    for (int i = 0; i < vids.size(); i++)
//    {
//        std::ostringstream path;
//        path << vidDirPath << vidNames[i] << "." << vidExt;
//
//        VideoSplitter vs(path.str(), tmpDirPath);
//
//        vs.split(<#FrameNumber start#>, <#FrameNumber end#>)
//
//
//    }


    return false;
}


bool videoEqual(cv::string v1, cv::string v2, cv::Vec2i range)
{
    cv::VideoCapture vc1(v1);
    cv::VideoCapture vc2(v2);

    for (int i = 0; i < range[0]; i++)
    {
        vc1.grab();
    }

    for (int i = 0; i < range[1]-range[0]; i++)
    {
        cv::Mat f1;
        cv::Mat f2;
        cv::Mat diffed;
        vc1 >> f1;
        vc2 >> f2;
        cv::absdiff(f1, f2, diffed);
        if (0 != cv::countNonZero(diffed))
        {
            return false;
        }
    }

    return true;
}
