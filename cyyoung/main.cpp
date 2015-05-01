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

//#define SKIP_NUM_FRAMES 440 // Pitch 1 for output4_480p.mp4
  #define SKIP_NUM_FRAMES 810 // Pitch 2 for output4_480p.mp4
//#define SKIP_NUM_FRAMES 206

/**
 * Number of tick marks in the compas.
 */
#define COMPASS_ROSE 4

#define MAX_FRAMES 100000

#define FACE_DETECTOR_LOC "/Users/mahi/Downloads/opencv-2.4.9/data/haarcascades/haarcascade_frontalface_alt.xml"


#define VIDEO_LOC "/Users/mahi/100GOPRO/output4_480p.mp4"
//#define VIDEO_LOC "/Users/mahi/Desktop/trimmedbes.MP4"

#define WRITE_ABSOLUTE_DIR "/Users/mahi/hoged_frames3/"


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

//MotionVec shiftMotionVec(const MotionVec mv, cv::Rect roi)
//{
//    MotionVec shifted;
//    // Shift |mv| by the |roi|. Save computation by doing it only on creation.
//    for (int i = 0; i < mv.size(); i++)
//    {
//        const cv::Point2f first = mv[i].first;
//        const cv::Point2f second = mv[i].second;
//        shifted.push_back({{first.x+roi.x,first.y+roi.y},{second.x+roi.x,second.y+roi.y}});
//    }
//    return shifted;
//}

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

cv::CascadeClassifier face_classifier;

cv::Rect detectFace(cv::Mat &frame)
{
    cv::Mat gray;
    switch (frame.channels())
    {
        case 1: gray = frame; break;
        case 3: cv::cvtColor(frame, gray, CV_BGR2GRAY); break;
        case 4: cv::cvtColor(frame, gray, CV_BGRA2GRAY); break;
    }

    std::vector<cv::Rect> faces;
    face_classifier.detectMultiScale( gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30, 30) );

    for (int i = 0; i < faces.size(); i++)
    {
        drawBWRectangle(frame, faces[i], true);
    }

    if (0 != faces.size())
    {
        return largestRectangle(faces);
    }
    else
    {
        return cv::Rect(0,0,0,0);
    }
}


// Uses hog to find the body and then sees if there is no motion
// in the motion vector within the body.
cv::Rect findValidHuman(cv::HOGDescriptor &hogd, cv::Mat &frame)
{
    cv::Mat gray = frame;
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
            drawBWRectangle(frame, locations[i], true);

            printf("Location[%d] = { %0.2f, %0.2f, %0.2f, %0.2f }\n",
                   i,
                   float(locations[i].x),
                   float(locations[i].y),
                   float(locations[i].width),
                   float(locations[i].height));
        }

        return largestRectangle(locations);
    }

    return cv::Rect(0,0,0,0);
}

// For convenience so we know it is correct.
inline bool pointWithinRectangle(cv::Rect r, cv::Point p)
{
    return p.x >= r.x && p.x <= r.x + r.width && p.y >= r.y && p.y <= p.y + r.height;
}
inline bool movementWithinRectangle(std::pair<cv::Point2f, cv::Point2f> movement, cv::Rect r)
{
    return pointWithinRectangle(r, movement.first) && pointWithinRectangle(r, movement.second);
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
        if (movementWithinRectangle(mvHistory[i], rectHuman))
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

double calculateRectHorizontalDisplacement(cv::Rect previous, cv::Rect current)
{
    const cv::Point centerP(previous.x + (previous.width / 2.0), 0.0);
    const cv::Point centerC(current.x + (current.width / 2.0), 0.0);
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





typedef struct {
    unsigned int start, end;
} PitchFrameWindow;

typedef struct {
    unsigned int post, tee, release, follow;
} SegmentedPitchFrames;

inline double calculateRadians(std::pair<cv::Point2f, cv::Point2f> movement)
{
    const double angle = atan2(0 - (movement.second.y - movement.first.y), movement.second.x - movement.first.x);
    return angle >= 0.0 ? angle : 6.2831853072 + angle;
}

// Not inclusive of min and max.
double calculatePercentOfValidAnglesWithinRectangle(MotionVec &mv, cv::Rect r, double min, double max)
{
    unsigned int valid = 0, count = 0;

    for (int i = 0; i < mv.size(); ++i)
    {
        if (movementWithinRectangle(mv[i], r))
        {
            ++count;
            double angle = calculateRadians(mv[i]);
            if (min < angle && angle < max)
            {
                ++valid;
            }
        }
    }

    return 0 == count? 0.0 : double(valid)/double(count);
}


bool pitcherPastPost(MotionVec &history, cv::Rect stationaryRect)
{
    const double minPercent = 0.8;

    // these two values are between 9𝜋/8 and 15𝜋/8
    const double min = 3.5342917353;
    const double max = 5.8904862255;

    double goodBadRatio = calculatePercentOfValidAnglesWithinRectangle(history, stationaryRect, min, max);
    printf("Good Bad Percent: %0.2f\n", goodBadRatio);
    return goodBadRatio > minPercent;
}


// Goes back in time and returns the frame where the leg is fully lifted. This means there is very
// little to no down movement.
unsigned int findPostFrameNumber(std::vector<MotionVec> &histories, cv::Rect stationaryRect, unsigned int start)
{
    const double maxPercentDown = 0.10;
    const double strictlyDownMin = 4.1233403578; // 21𝜋/16
    const double strictlyDownMax = 5.1050880621; // 13𝜋/8


    // Two places from the back since we know the last one is not the post already.
    const size_t firstLoopIndex = histories.size()-2;
    for (size_t i = firstLoopIndex; i > start; --i)
    {
        double percent = calculatePercentOfValidAnglesWithinRectangle(histories[i],
                                                                      stationaryRect,
                                                                      strictlyDownMin,
                                                                      strictlyDownMax);
        if (maxPercentDown > percent)
        {
            return (unsigned int)i;
        }
    }

    return 0;
}


bool pitcherPastLegDown(MotionVec &mv, cv::Rect stationaryRect)
{

    const double maxPercent = 0.15;

    // these two values are between 9𝜋/8 and 15𝜋/8
    const double min = 3.14159;
    const double max = 5.8904862255;

    double percent = calculatePercentOfValidAnglesWithinRectangle(mv, stationaryRect, min, max);
    printf("Good Bad Percent: %0.2f\n", percent);
    return percent < maxPercent;
}


bool pitcherPastTee(MotionVec &mv, cv::Rect stationaryRect)
{
    // Find downwards motion outside and left of rectangle.
    cv::Rect outsideRect(stationaryRect.x + int(stationaryRect.width*0.75),
                         stationaryRect.y + stationaryRect.height/5,
                         stationaryRect.width,
                         stationaryRect.height/2);

    // Must be pointing towards the west.
    const double minPercent = 0.40;
    const double min = 2.3561944902;
    const double max = 3.926990817;
    const double percent = calculatePercentOfValidAnglesWithinRectangle(mv, outsideRect, min, max);
    printf("The percent value in 'past tee' is %0.2f\n", percent);
    return percent > minPercent;
}

bool pitcherAtTeePosition(MotionVec &mv, cv::Rect stationaryRect)
{
    stationaryRect.height /= 2;

    const double minValidPercent = 0.90;

    const double min1 = -100;
    const double max1 = 2.3561944902;
    const double min2 = 5.4977871438;
    const double max2 = 100; // some number past 2pi.

    const double percent1 = calculatePercentOfValidAnglesWithinRectangle(mv, stationaryRect, min1, max1);
    const double percent2 = calculatePercentOfValidAnglesWithinRectangle(mv, stationaryRect, min2, max2);
    const double percent = percent1 + percent2;
    printf("Pitcher At Tee percent %0.4f\n", percent);
    return percent > minValidPercent;
}


// PASS IN TIME MOTION VEC
bool pitcherAtReleasePosition(MotionVec &mvTime, cv::Rect stationaryRect)
{
    cv::Rect armRoi(stationaryRect.x + stationaryRect.width,
                    stationaryRect.y,
                    stationaryRect.width,
                    stationaryRect.height);

    const double minPercent = 0.60;
    const double min = 3.14159;
    const double max = 5.8904862255;
    const double percent = calculatePercentOfValidAnglesWithinRectangle(mvTime, stationaryRect, min, max);
    printf("Percent of arm release going down %0.4f\n", percent);
    return percent > minPercent;
}




unsigned int frameNumberOfTee(PitchFrameWindow window, std::vector<MotionVec> &histories)
{



    return 0;
}

unsigned int frameNumberOfRelease(PitchFrameWindow window, std::vector<MotionVec> &histories)
{



    return 0;
}

unsigned int frameNumberOfFollow(PitchFrameWindow window, std::vector<MotionVec> &histories)
{



    return 0;
}


int main(int argc, const char * argv[])
{
    // |pitches| is what we use to generate composites at the end!
    std::vector<SegmentedPitchFrames> pitches;

    cv::HOGDescriptor hogd;
    hogd.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
    std::vector<MotionVec> timeMotionVecVec;
    std::vector<MotionVec> historyMotionVecVec;
    cv::VideoCapture vc;
    cv::Mat firstFrame;
    vc.open(VIDEO_LOC);
    //vc.open("/Users/mahi/Desktop/trimmedbes.MP4");
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
    cv::namedWindow("Time Motion", CV_WINDOW_KEEPRATIO);


    while (vc.grab())
    {
        const unsigned int frameNumber = countFrame();
        timeMotionVecVec.push_back({{{0.f,0.f},{0.f,0.f}}});

        printf("Frame number %d\n", frameNumber);

        // This is for convienence.
        if (SKIP_NUM_FRAMES > frameNumber)
        {
            continue;
        }

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

        historyMotionVecVec.push_back(mvHistory);

        // DRAW THE COMPASS.
        for (unsigned int i = 0; i < COMPASS_ROSE; i++)
        {
            mvDrawn.push_back(compass[i]);
        }


        // Find the pitcher.
        cv::Rect rectHuman = findValidHuman(hogd, frame);
        //cv::Rect rectHuman = detectFace(frame);
        bool foundNewRectangle = false;
        // divide by zero error mitigation.
        static cv::Rect drawnBorder1(0,0,1,1), drawnBorder2(0,0,1,1), drawnBorder3(0,0,1,1);
        static unsigned int framesSinceStationary = 0;
        double displacement = 0;

        const double rectShrinkTolerance = 0.75;
        bool validNewRect = double(rectHuman.height) / drawnBorder1.height > rectShrinkTolerance;
        if (rectHuman.x != 0 && rectHuman.y != 0 && rectHuman.width != 0 && rectHuman.height != 0 && validNewRect)
        {
            foundNewRectangle = true;

            displacement = calculateRectHorizontalDisplacement(drawnBorder1, rectHuman);

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


        static bool foundPost = false, foundLegDown = false, foundTee = false;
        static bool foundRelease = false, foundFollow = false;
        static unsigned int postFrameNumber = 0, teeFrameNumber = 0;
        static unsigned int releaseFrameNumber = 0, followFrameNumber = 0;

        if (foundNewRectangle && pitcherStationary(rectHuman, mvHistory))
        {
            framesSinceStationary = 0;
            foundPost = false;
            foundLegDown = false;
            foundTee = false;
            foundRelease = false;
            foundFollow = false;
        }
        else
        {
            framesSinceStationary++;

            // FIND THE POST POSITION
            const unsigned int lowMotionBuffer = 5;
            if (!foundPost && framesSinceStationary > lowMotionBuffer)
            {
                if (pitcherPastPost(historyMotionVecVec.back(), drawnBorder1))
                {
                    unsigned int pitchStart = (unsigned int)historyMotionVecVec.size()-framesSinceStationary;
                    postFrameNumber = findPostFrameNumber(historyMotionVecVec, drawnBorder1, pitchStart);
                    postFrameNumber += SKIP_NUM_FRAMES;
                    if (0 != postFrameNumber)
                    {
                        foundPost = true;
                    }
                }
            }

            if (foundPost && !foundTee)
            {
                foundTee = pitcherPastLegDown(mv, drawnBorder1);
                teeFrameNumber = frameNumber;
            }

            if (foundPost && foundTee && !foundRelease)
            {
                foundRelease = pitcherAtReleasePosition(timeMotionVecVec.back(), drawnBorder1);
                releaseFrameNumber = frameNumber;
            }

            if (foundRelease)
            {
                printf("Found release");
            }

        }

        drawBWRectangle(frame, drawnBorder1, true);
        drawBWRectangle(frame, drawnBorder2, false);
        drawBWRectangle(frame, drawnBorder3, true);



        // Name the saved image
        char outPath[50];
        bzero(outPath, 50);
        sprintf(outPath, "%sframe%06d.png", WRITE_ABSOLUTE_DIR, frameNumber);
//        cv::imshow("Computer Vision", frame);

        cv::Mat frameClone = frame;
        cv::imshow("Time Motion", oflow.drawMotion(timeMotionVecVec.back(), frameClone));


        cv::Mat drawnFrame = oflow.drawMotion(mvDrawn, frame);

        cv::Mat outputFrame;
        if (0 != drawnFrame.cols % 2)
        {
            outputFrame = drawnFrame(cv::Rect(0,0,drawnFrame.cols-1, drawnFrame.rows));
        }
        else
        {
            outputFrame = drawnFrame;
        }

        char text[100];
        bzero(text, 100);
        sprintf(text, "Frame %d", frameNumber);
        cv::putText(outputFrame, text, cv::Point(0,10), CV_FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,0,0));

        if (framesSinceStationary != frameNumber)
        {
            bzero(text, 100);
            sprintf(text, "Frames since stationary: %d", framesSinceStationary);
            cv::putText(outputFrame, text, cv::Point(0,30), CV_FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,0,0));
        }

//        if (displayFootDownNotification)
//        {
//            bzero(text, 100);
//            sprintf(text, "PITCHER FOOT DOWN");
//            cv::putText(outputFrame, text, cv::Point(drawnBorder1.x, drawnBorder1.y + 20), CV_FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,255,0));
//        }

        bzero(text, 100);
        sprintf(text, "Displacement: %0.2f pixels", displacement);
        cv::putText(outputFrame, text, cv::Point(0, 50), CV_FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,0,0));

        cv::imshow("Computer Vision", outputFrame);
        //cv::imwrite(cv::string(outPath), outputFrame);
        printf("Wrote:%s\n", outPath);
    }

    vc.release();
    return 0;
}
