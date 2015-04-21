//
//  OpticalFlow.cpp
//  cyyoung
//
//  Created by MATHEW RENY on 4/18/15.
//  Copyright (c) 2015 MATHEW RENY. All rights reserved.
//

#include "OpticalFlow.h"

void cvt2gray(cv::Mat &img)
{
    switch (img.channels())
    {
        case 1:
            // Already gray.
            return;
        case 3:
            cv::cvtColor(img, img, CV_BGR2GRAY);
            return;
        case 4:
            cv::cvtColor(img, img, CV_BGRA2GRAY);
            return;
        default:
            printf("Input image bad\n");
            exit(1);
            return;
    }
}

OpticalFlow::OpticalFlow(cv::Mat &firstFrame)
{
    cvt2gray(firstFrame);
    this->prevFrame = firstFrame;
    cv::goodFeaturesToTrack(this->prevFrame,
                            this->prevFeatures,
                            MAX_FEATURES,
                            0.01,
                            10);

    this->tc = cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.3);
}

cv::Mat OpticalFlow::drawMotion(MotionVec mv, cv::Mat &image)
{
    // Convert image to HSV so we can see the lines better
    // It is easier to work with colors using HSV
    cv::Mat imageColor;
    cv::cvtColor(image, imageColor, CV_GRAY2BGR);
    cv::cvtColor(imageColor, imageColor, CV_BGR2HSV);

    const double pi = 3.14159265358979323846;
    for (int i = 0; i < mv.size(); i++)
    {
        cv::Point2f from, to;
        from = mv[i].first;
        to = mv[i].second;

        // Get color based from 0 to 2pi
        cv::Point2f toNorm; // normalize |from| to 0,0;
        toNorm = to;
        toNorm.x -= from.x;
        toNorm.y -= from.y;

        double angle;
        if (0 == toNorm.x)
        {
            if (0 > toNorm.y)
            {
                angle = (3.0*pi) / 2.0;
            }
            else
            {
                angle = pi/2.0;
            }
        }
        else
        {
            angle = tan(double(toNorm.y)/double(toNorm.x));
        }

        angle /= 2*pi; // get percentage of full circle.
        //angle += 0.25;

        cv::Scalar colorHSV = {angle*255, 255, 255};

        //cv::Point2f extended = {mv[i].second.x * 5, mv[i].second.y*5};

        cv::line(imageColor, mv[i].first, mv[i].second, colorHSV);
    }

    // Convert it to bgr since that's the format everyone likes to use.
    cv::cvtColor(imageColor, imageColor, CV_HSV2BGR);

    return imageColor;
}

float distance(cv::Point2f p1, cv::Point2f p2)
{
    return sqrtf((p2.x-p1.x)*(p2.x-p1.x) +
                 (p2.y-p1.y)*(p2.y-p1.y));
}


MotionVec OpticalFlow::feed(cv::Mat &img)
{
    MotionVec motion = MotionVec();

    if (1 != img.channels())
    {
        cvt2gray(img);
    }

    // 0 if feature doesn't correspond to points.
    std::vector<uchar> status;

    // Ignoring for now.
    std::vector<float> err;

    this->nextFeatures.clear();
    cv::calcOpticalFlowPyrLK(this->prevFrame,
                             img,
                             this->prevFeatures,
                             this->nextFeatures,
                             status,
                             err);

    //CV_DbgAssert(50 == status.size());
    std::vector<cv::Point2f> replacements;

    std::vector<float> magnitudes;
    for (int i = 0; i < status.size(); i++)
    {
        if (0 != status[i])
        {
            cv::Point2f pp, pn;
            pp = this->prevFeatures[i];
            pn = this->nextFeatures[i];
            //float dist = distance(pp, pn);
            //magnitudes.push_back(dist);
            motion.push_back(std::pair<cv::Point2f, cv::Point2f>(pp, pn));
        }
//        else
//        {
//            this->nextFeatures[i] = replacements[i];
//        }
    }




    // Cleanup.
    this->prevFrame = img;
    //this->prevFeatures = this->nextFeatures;

    cv::goodFeaturesToTrack(this->prevFrame,
                            this->prevFeatures,
                            MAX_FEATURES,
                            0.01,
                            10);



//    for (int i = 0; i < motion.size(); i++)
//    {
//        // Does this work at all?
//        printf("Motion from %0.2f, %0.2f to %0.2f, %0.2f\n",
//               motion[i].first.x,
//               motion[i].first.y,
//               motion[i].second.x,
//               motion[i].second.y);
//    }

    #if !RELEASE
    // Show the motion in a new window.
    cv::imshow("Motion", drawMotion(motion, img));
    cv::waitKey();
    #endif

    return motion;
}



