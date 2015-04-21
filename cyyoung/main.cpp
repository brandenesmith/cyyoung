//
//  main.cpp
//  cyyoung
//
//  Created by MATHEW RENY on 3/21/15.
//  Copyright (c) 2015 MATHEW RENY. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>

#include <dispatch/dispatch.h>
#include <unistd.h>

#include "Segment.h"

#include "CompositeImageTests.h"

#include "OpticalFlow.h"

cv::VideoCapture vc;

int main(int argc, const char * argv[]) {
    //std::string file;
    //std::getline(std::cin, file);

//    std::cout << "Hello, World!\n";
//
//    CompositeImageTests cit;
//
//    if (cit.testAllCases())
//    {
//        printf("Composite image tests passed!\n");
//    }
//    else
//    {
//        printf("Composite image tests failed!\n");
//    }
//
//    cv::waitKey();

//    std::cout << "Hello, World!\n";
//
//    //Video v = Video("/Users/mahi/testvid.MP4");
//
    cv::namedWindow("Video Output", CV_WINDOW_KEEPRATIO);

//    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
//        cv::VideoCapture vc;
//        vc.open("/Users/mahi/testvid.MP4");
//
//        dispatch_block_t dblock = ^{
//            cv::Mat frame;
//            vc >> frame;
//            cv::Mat copy = frame.clone();
//            //dispatch_sync(dispatch_get_main_queue(), ^{
//            cv::imshow("Video Output", copy);
//        };
//
//        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 300 * NSEC_PER_USEC), dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), dblock);
//
//            //std::ostringstream win;
//            //win << "Video" << ++count;
//
//            cv::Mat frame;
//            vc >> frame;
//            cv::Mat copy = frame.clone();
//            //dispatch_sync(dispatch_get_main_queue(), ^{
//                cv::imshow("Video Output", copy);
//            //});
//        }
//    });

    cv::VideoCapture vc;
    vc.open("/Users/mahi/shortsha2.mov");

    OpticalFlow *oflow = NULL;

    while (true)
    {
        //std::ostringstream win;
        //win << "Video" << ++count;

        cv::Mat frame;
        vc >> frame;
        cv::Mat copy = frame.clone();

        if (NULL == oflow)
        {
            oflow = new OpticalFlow(copy);
        }
        else
        {
            oflow->feed(copy);
        }

        //cv::imshow("Video Output", copy);
    }

    // insert code here...
    return 0;
}
