//
//  main.cpp
//  cyyoung
//
//  Created by MATHEW RENY on 3/21/15.
//  Copyright (c) 2015 MATHEW RENY. All rights reserved.
//
#include <iostream>
#include <opencv2/opencv.hpp>
#include <unistd.h>

#include "Segment.h"

#include "CompositeImageTests.h"

#include "OpticalFlow.h"

/**
 * Distance between motion points.
 */
#define MIN_TRACK_DISTANCE 3.f

/**
 * Size of the motion history image. To disable, make this zero.
 */
#define MOTION_HISTORY_IMAGE 4

/**
 * Number of tick marks in the compas.
 */
#define COMPASS_ROSE 4

#define MAX_FRAMES 100000


const float centerX = 100.f, centerY = 100.f, radius = 22;
typedef enum { north = 0, east = 1, south = 2, west = 3 } ClockwiseCardDir;
const cv::Point2f center = { centerX, centerY };
const cv::Point2f nesw[4] = {{centerX,centerY-radius},
    {centerX+radius,centerY},{centerX,centerY+radius},{centerX-radius,centerY}};
const std::pair< cv::Point2f, cv::Point2f > compass[COMPASS_ROSE] = {{center, nesw[north]},
    {center, nesw[east]}, {center, nesw[south]}, {center, nesw[west]}};

static int frameCount = 0;
int countFrame()
{
    frameCount++;
    if (MAX_FRAMES == frameCount)
    {
        exit(1);
    }
    return frameCount;
}

MotionVec shiftMotionVec(const MotionVec mv, cv::Rect roi)
{
    MotionVec shifted;
    // Shift |mv| by the |roi|. Save computation by doing it only on creation.
    for (int i = 0; i < mv.size(); i++)
    {
        const cv::Point2f first = mv[i].first;
        const cv::Point2f second = mv[i].second;
        shifted.push_back({{first.x+roi.x,first.y+roi.y},{second.x+roi.x,second.y+roi.y}});
    }
    return shifted;
}

bool humanTooSmallToBeValid(int frameHeight, int humanHeight)
{
    return humanHeight < frameHeight / 2;
}

// For convenience.
cv::Rect largestRectangle(std::vector<cv::Rect> humans)
{
    int maxIndex = 0;
    double maxArea = 0;
    for (int i = 0; i < humans.size(); i++)
    {
        double area = humans[i].width * humans[i].height;
        if (area > maxArea)
        {
            maxArea = area;
            maxIndex = i;
        }
    }

    return humans[maxIndex];
}

// Uses hog to find the body and then sees if there is no motion
// in the motion vector within the body.
cv::Rect findValidHuman(cv::HOGDescriptor &hogd, cv::Mat &frame)
{
    cv::Mat gray = frame.clone();
    switch (frame.channels())
    {
        case 1: break;
        case 3: cv::cvtColor(gray, gray, CV_BGR2GRAY); break;
        case 4: cv::cvtColor(gray, gray, CV_BGRA2GRAY); break;
        default: throw new std::runtime_error("Invalid channel count");
    }

    std::vector<cv::Rect> locations;
    hogd.detectMultiScale(gray, locations);

    if (0 != locations.size())
    {
        for (int i = 0; i < locations.size(); i++)
        {
            printf("Location[%d] = { %0.2f, %0.2f, %0.2f, %0.2f }\n",
                   i,
                   float(locations[i].x),
                   float(locations[i].y),
                   float(locations[i].width),
                   float(locations[i].height));
        }

        cv::Rect largest = largestRectangle(locations);

        if (!humanTooSmallToBeValid(gray.rows, largest.height))
        {
            return largest;
        }
    }

    return cv::Rect(0,0,0,0);
}


void drawBWRectangle(cv::Mat &frame, cv::Rect r, bool black)
{
    cv::Point2f topLeft, topRight, bottomLeft, bottomRight;
    bottomLeft  = cv::Point2f(r.x, r.y + r.height);
    topLeft     = cv::Point2f(r.x, r.y);
    topRight    = cv::Point2f(r.x + r.width, r.y);
    bottomRight = cv::Point2f(r.x + r.width, r.y + r.height);

    cv::Scalar color = black ? cv::Scalar(0,0,0) : cv::Scalar(255,255,255);
    cv::line(frame, topLeft,    topRight,    color);
    cv::line(frame, topLeft,    bottomLeft,  color);
    cv::line(frame, topRight,   bottomRight, color);
    cv::line(frame, bottomLeft, bottomRight, color);

}

// For convenience so we know it is correct.
inline bool pointWithinRectangle(cv::Rect r, cv::Point p)
{
    return p.x >= r.x && p.x <= r.x + r.width && p.y >= r.y && p.y <= p.y + r.height;
}

bool pitcherStationary(cv::Rect rectHuman, MotionVec &mvHistory)
{
    int shorten = rectHuman.height / 6;
    rectHuman.y += shorten;
    rectHuman.height -= shorten * 2;

    double thresh = rectHuman.width;
    double weight = 0;
    for (int i = 0; i < mvHistory.size(); i++)
    {
        const cv::Point2f p1 = mvHistory[i].first;
        const cv::Point2f p2 = mvHistory[i].second;
        if (pointWithinRectangle(rectHuman, p1) && pointWithinRectangle(rectHuman, p2))
        {
            weight += distance(p1, p2);

            if (weight > thresh)
            {
                return false;
            }
        }
    }

    return true;
}

double calculateRectDisplacement(cv::Rect previous, cv::Rect current)
{
    const cv::Point centerP(previous.x + (previous.width / 2.0), previous.y + (previous.height / 2.0));
    const cv::Point centerC(current.x + (current.width / 2.0), current.y + (current.height / 2.0));
    return distance(centerP, centerC);
}

// Sees if the rectangle shifted by a specified amount.
bool boundingBoxShifted(cv::Rect previous, cv::Rect current, double ratio)
{
    // First get width.
    int minDist = ((previous.width + current.width)/2.0)*ratio;

    cv::Point prevCenter(previous.x + (previous.width / 2.0), previous.y + (previous.height / 2.0));
    cv::Point currCenter(current.x + (current.width / 2.0), current.y + (current.height / 2.0));

    return minDist < distance(prevCenter, currCenter);
}


int main(int argc, const char * argv[])
{
    cv::HOGDescriptor hogd;
    hogd.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
    std::vector<MotionVec> timeMotionVecVec(MAX_FRAMES);
    cv::VideoCapture vc;
    cv::Mat firstFrame;
    vc.open("/Users/mahi/Desktop/trimmedbes.MP4");
    //vc.open(0);
    vc.grab();
    vc.retrieve(firstFrame);
//    const cv::Point2i roiOrigin = { 2 * firstFrame.cols / 7, 0};
//    const cv::Size2i roiSize = { 4 * firstFrame.cols / 7, firstFrame.rows};
//    const cv::Rect roi = cv::Rect(roiOrigin, roiSize);
    const cv::Rect roi = cv::Rect(cv::Point2i(0,0),cv::Size2i(firstFrame.cols, firstFrame.rows));
    OpticalFlow oflow = OpticalFlow(firstFrame, roi);
    firstFrame.deallocate();


    //std::string file;
    //std::getline(std::cin, file);

    cv::namedWindow("Computer Vision", CV_WINDOW_KEEPRATIO);

    while (vc.grab())
    {
        const int frameNumber = countFrame();
        timeMotionVecVec.push_back({{{0.f,0.f},{0.f,0.f}}});

        cv::Mat frame;
        if (!vc.retrieve(frame) || frame.empty())
        {
            break;
        }


        MotionVec mv = oflow.feed(frame);

        MotionVec mvThresholded, mvDrawn, mvHistory;


        // NOW DRAW THE mvThresholded VECTORS INTO mvDrawn
        for (int i = 0; i < mv.size(); i++)
        {
            if (MIN_TRACK_DISTANCE <= distance(mv[i].first,
                                               mv[i].second))
            {
                mvThresholded.push_back(mv[i]);
            }
        }

        for (int i = 0; i < mvThresholded.size(); i++)
        {
            mvHistory.push_back(mvThresholded[i]);
            mvDrawn.push_back(mvThresholded[i]);
        }

        // Only runs if not zero
        // DRAW THE MOTION HISTORY IMAGE
        timeMotionVecVec.push_back(mvThresholded);

        const size_t stop = timeMotionVecVec.size()-1;
        size_t i = stop - MOTION_HISTORY_IMAGE;
        for (; stop != i; i++)
        {
            for (int j = 0; j < timeMotionVecVec[i].size(); j++)
            {
                mvHistory.push_back(timeMotionVecVec[i][j]);
                mvDrawn.push_back(timeMotionVecVec[i][j]);
            }
        }

        // DRAW THE COMPASS.
        for (unsigned int i = 0; i < COMPASS_ROSE; i++)
        {
            mvDrawn.push_back(compass[i]);
        }


        // Find the pitcher.
        cv::Rect rectHuman = findValidHuman(hogd, frame);
        bool foundNewRectangle = false;
        bool rectHumanShifted = false;
        static cv::Rect drawnBorder1, drawnBorder2, drawnBorder3;
        if (rectHuman.x != 0 && rectHuman.y != 0 && rectHuman.width != 0 && rectHuman.height != 0)
        {
            foundNewRectangle = true;

            const double displacement = calculateRectDisplacement(drawnBorder1, rectHuman);
            char displacementText[100];
            bzero(displacementText, 100);
            sprintf(displacementText, "Displacement: %0.2f pixels", displacement);
            cv::putText(frame, displacementText, cv::Point(0, 50), CV_FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,0,0));


            if (displacement > rectHuman.width * 0.75)
            {
                printf("Captured large displacement");
            }


            drawnBorder1 = cv::Rect(int(rectHuman.x),
                                    int(rectHuman.y),
                                    int(rectHuman.width),
                                    int(rectHuman.height));

            drawnBorder2 = cv::Rect(int(rectHuman.x)      - 1,
                                    int(rectHuman.y)      - 1,
                                    int(rectHuman.width)  + 2,
                                    int(rectHuman.height) + 2);

            drawnBorder3 = cv::Rect(int(rectHuman.x)      - 2,
                                    int(rectHuman.y)      - 2,
                                    int(rectHuman.width)  + 4,
                                    int(rectHuman.height) + 4);
        }

        static unsigned int framesSinceStationary = 0;

        if (foundNewRectangle && pitcherStationary(rectHuman, mvHistory))
        {
            framesSinceStationary = 0;
        }
        else
        {
            framesSinceStationary++;
        }

        if (framesSinceStationary != frameNumber)
        {
            char text[50];
            bzero(text, 50);
            sprintf(text, "Frames since stationary: %d", framesSinceStationary);
            cv::putText(frame, text, cv::Point(0,30), CV_FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,0,0));
        }

        if (rectHumanShifted)
        {

        }

        drawBWRectangle(frame, drawnBorder1, true);
        drawBWRectangle(frame, drawnBorder2, false);
        drawBWRectangle(frame, drawnBorder3, true);

        // Name the saved image
        char outPath[50];
        bzero(outPath, 50);
        sprintf(outPath, "/Users/mahi/hoged_frames/frame%06d.png", frameNumber);
        cv::imshow("Computer Vision", oflow.drawMotion(mvDrawn, frame));
        printf("Wrote:%s\n", outPath);
        //cv::imwrite(cv::string(outPath), oflow.drawMotion(mvDrawn, frame));
    }

    vc.release();
    return 0;
}
