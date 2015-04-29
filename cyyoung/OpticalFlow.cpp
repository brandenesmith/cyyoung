//
//  OpticalFlow.cpp
//  cyyoung
//
//  Created by MATHEW RENY on 4/18/15.
//  Copyright (c) 2015 MATHEW RENY. All rights reserved.
//

#include "OpticalFlow.h"



float radianAngle(cv::Point2f p1, cv::Point2f p2)
{
    return atan2f(p2.y - p1.y, p2.x - p1.x);
}

//void cvt2gray(cv::Mat &img)
//{
//    switch (img.channels())
//    {
//        case 1:
//            // Already gray.
//            return;
//        case 3:
//            cv::cvtColor(img, img, CV_BGR2GRAY);
//            return;
//        case 4:
//            cv::cvtColor(img, img, CV_BGRA2GRAY);
//            return;
//        default:
//            printf("Input image bad\n");
//            exit(1);
//            return;
//    }
//}

cv::Mat grayRoiCopy(cv::Mat &colorFrame, cv::Rect roi)
{
    if (colorFrame.empty())
    {
        printf("Do not feed empty images.");
        throw new std::runtime_error("Do not feed empty images.");
    }

    cv::Mat copy;
    switch (colorFrame.channels())
    {
        default:
            copy = colorFrame.clone();
        case 4:
            cv::cvtColor(colorFrame, copy, CV_BGRA2GRAY);
            break;
        case 3:
            cv::cvtColor(colorFrame, copy, CV_BGR2GRAY);
            break;
    }

    static cv::Rect sroi = roi;
    return copy(sroi);
}

OpticalFlow::OpticalFlow(cv::Mat &firstFrame, cv::Rect roi)
{
    cv::Mat copy = grayRoiCopy(firstFrame, roi);
    this->prevFrame = copy;
    cv::goodFeaturesToTrack(this->prevFrame,
                            this->prevFeatures,
                            MAX_FEATURES,
                            MIN_FEATURE_DISTANCE,
                            10);

    this->tc = cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.3);
}

cv::Mat OpticalFlow::drawMotion(MotionVec mv, cv::Mat &image)
{
    // Convert image to HSV so we can see the lines better
    // It is easier to work with colors using HSV
    cv::Mat imageColor = image.clone();
    switch(imageColor.channels())
    {
        case 1: cv::cvtColor(imageColor, imageColor, CV_GRAY2BGR); // intermediate step.
        case 3: cv::cvtColor(imageColor, imageColor, CV_BGR2HSV);
    }

    const float twopi = 2.0*3.14159265358979323846;
    for (int i = 0; i < mv.size(); i++)
    {
        const float angle = radianAngle(mv[i].first, mv[i].second);

        // get percentage of full circle.
        const float anglePercent = angle / (twopi);

        const cv::Scalar colorHSV = { anglePercent*255.f, 255.f, 255.f };

        //cv::Point2f extended = {mv[i].second.x * 5, mv[i].second.y*5};

        cv::line(imageColor, mv[i].first, mv[i].second, colorHSV);
    }

    // Convert it to bgr since that's the format everyone likes to use.
    cv::cvtColor(imageColor, imageColor, CV_HSV2BGR);

    return imageColor;
}


MotionVec OpticalFlow::feed(cv::Mat img)
{
    MotionVec motion = MotionVec();

    const cv::Rect emptyRect;
    cv::Mat copy = grayRoiCopy(img, emptyRect);
    if (1 != copy.channels())
    {
        printf("Should never reach here");
        exit(5);
    }

    // 0 if feature doesn't correspond to points.
    std::vector<uchar> status;

    // Ignoring for now.
    std::vector<float> err;

    this->nextFeatures.clear();
    cv::calcOpticalFlowPyrLK(this->prevFrame,
                             copy,
                             this->prevFeatures,
                             this->nextFeatures,
                             status,
                             err);

    //CV_DbgAssert(50 == status.size());

    std::vector<float> magnitudes;
    for (int i = 0; i < status.size(); i++)
    {
        if (0 != status[i])
        {
            cv::Point2f pp, pn;
            pp = this->prevFeatures[i];
            pn = this->nextFeatures[i];

            //if (3 < distance(pp,pn))
            //{
                motion.push_back(std::pair<cv::Point2f, cv::Point2f>(pp, pn));
            //}

            //float dist = distance(pp, pn);
            //magnitudes.push_back(dist);
        }
    }




    // Cleanup.
    this->prevFrame = copy;
    //this->prevFeatures = this->nextFeatures;

    cv::goodFeaturesToTrack(this->prevFrame,
                            this->prevFeatures,
                            MAX_FEATURES,
                            MIN_FEATURE_DISTANCE,
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
    //cv::imshow("Motion", drawMotion(motion, img));
    //cv::waitKey();
    #endif

    return motion;
}



