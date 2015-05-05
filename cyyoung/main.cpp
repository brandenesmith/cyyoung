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

#include "OpticalFlow.h"

#include "PitchComposite.h"

#define WRITE_VIDEO_OUTPUT_AND_CLEAN_DIR 0


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

#define FACE_DETECTOR_LOC "/Users/mahi/Downloads/opencv-2.4.9/data/haarcascades/haarcascade_frontalface_alt.xml"


#define VIDEO_LOC "/Users/mahi/100GOPRO/output4_480p.mp4"
#define INPUT_VIDEO_FPS 30
#define SKIP_NUM_FRAMES 400


//#define VIDEO_LOC "/Users/mahi/100GOPRO/output5_480p.mp4"
//#define INPUT_VIDEO_FPS 60
//#define SKIP_NUM_FRAMES 2

//#define VIDEO_LOC "/Users/mahi/100GOPRO/output6_480p.mp4"
//#define INPUT_VIDEO_FPS 60
//#define SKIP_NUM_FRAMES 90

//#define VIDEO_LOC "/Users/mahi/Desktop/trimmedbes.MP4"
//#define INPUT_VIDEO_FPS 30
//#define SKIP_NUM_FRAMES 440 // Pitch 1 for output4_480p.mp4
//#define SKIP_NUM_FRAMES 810 // Pitch 2 for output4_480p.mp4


#define WRITE_ABSOLUTE_DIR "/Users/mahi/hoged_frames3/"

#define OUTPUT_VIDEO_NAME "output.mp4"


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


//// Uses hog to find the body and then sees if there is no motion
//// in the motion vector within the body.
//cv::Rect findValidHuman(cv::HOGDescriptor &hogd, cv::Mat &frame)
//{
//    cv::Mat gray = frame;
//    switch (frame.channels())
//    {
//        case 1: break;
//        case 3: cv::cvtColor(gray, gray, CV_BGR2GRAY); break;
//        case 4: cv::cvtColor(gray, gray, CV_BGRA2GRAY); break;
//        default: throw new std::runtime_error("Invalid channel count");
//    }
//
//    std::vector<cv::Rect> locations;
//    hogd.detectMultiScale(gray, locations);
//
//    if (0 != locations.size())
//    {
//        for (int i = 0; i < locations.size(); i++)
//        {
//            drawBWRectangle(frame, locations[i], true);
//
//            printf("Location[%d] = { %0.2f, %0.2f, %0.2f, %0.2f }\n",
//                   i,
//                   float(locations[i].x),
//                   float(locations[i].y),
//                   float(locations[i].width),
//                   float(locations[i].height));
//        }
//
//        return largestRectangle(locations);
//    }
//
//    return cv::Rect(0,0,0,0);
//}

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

    double scale = 240.0 / gray.rows;

    cv::resize(gray, gray, cv::Size(gray.cols*scale,gray.rows*scale));

    std::vector<cv::Rect> locations;
    hogd.detectMultiScale(gray, locations);

    if (0 != locations.size())
    {
        for (int i = 0; i < locations.size(); i++)
        {
            cv::Rect drawn = locations[i];
            drawn.x /= scale;
            drawn.y /= scale;
            drawn.width /= scale;
            drawn.height /= scale;

            drawBWRectangle(frame, drawn, true);

            printf("Location[%d] = { %0.2f, %0.2f, %0.2f, %0.2f }\n",
                   i,
                   float(locations[i].x/scale),
                   float(locations[i].y/scale),
                   float(locations[i].width/scale),
                   float(locations[i].height/scale));
        }

        cv::Rect scaled = largestRectangle(locations);
        scaled.x /= scale;
        scaled.y /= scale;
        scaled.width /= scale;
        scaled.height /= scale;

        // This allows us to ignore the rectangle if it is on the right
        // side of the frame. The pitcher is assumed to be right handed
        // so he/she should start out on the left side or middle of the
        // frame.  We do this in case the HOG detects the pitcher as they
        // walk back to the mound but the HOG doesn't capture the pitcher
        // when they are actually on the mound. It's a rare case but I have
        // seen it enough to include this.
        if ((frame.cols / 2) > scaled.x)
        {
            return scaled;
        }
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
unsigned int findPostFrameIndex(std::vector<MotionVec> &histories, cv::Rect stationaryRect, unsigned int start)
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


inline double correctRadians(std::pair<cv::Point2f, cv::Point2f> motion)
{
    double angle = atan2f(0 - (motion.second.y - motion.first.y), motion.second.x - motion.first.x);
    if (angle < 0)
    {
        angle = 2*3.14159 + angle;
    }
    return angle;
}


bool pitcherLegUp(MotionVec &mv, cv::Rect stationaryRect, const double minPercentUp)
{
    double min = 0.7853981634;
    double max = 3.14159;

    int valid = 1;
    int good = 0;


    for (int i = 0; i < mv.size(); i++)
    {
        if (movementWithinRectangle(mv[i] , stationaryRect ))
        {
            valid++;

            double angle = correctRadians(mv[i]);

            if (min < angle && angle < max)
            {
                good++;
            }
        }
    }

    const double percent = double(good)/double(valid);
    printf("Good %d, Valid %d, Percent %0.2f\n", good, valid, percent);

    return percent > minPercentUp;
}


double averageAngleWithinRect(MotionVec &mv, cv::Rect r)
{
    // Create a unit vector and then find its angle.
    float xtot = 0, ytot = 0;

    for (int i = 0; i < mv.size(); i++)
    {
        if (movementWithinRectangle(mv[i], r))
        {
            xtot += mv[i].second.x - mv[i].first.x;
            ytot += mv[i].second.y - mv[i].first.y;
        }
    }
    return correctRadians({cv::Point2f(0.f,0.f),cv::Point2f(xtot,ytot)});
}


// Goes back in time from the release and tries to find the tee position.
unsigned int pitcherTeeFrameNumber(std::vector<MotionVec> &motions,
                                   cv::Rect stationaryRect,
                                   unsigned int post)
{
    cv::Rect body(stationaryRect.x + stationaryRect.width/2,
                  stationaryRect.y,
                  stationaryRect.width * 0.75,
                  stationaryRect.height);

    std::vector<double> averages;

    size_t size = motions.size();
    for (size_t i = size-2; i > post; i--)
    {
        averages.push_back(averageAngleWithinRect(motions[i], body));
        printf("Frame %lu average: %0.2f\n", i+SKIP_NUM_FRAMES, averages.back());
    }

    return 0;
}


typedef struct {
    unsigned int start, post, tee, release, follow, end;
} PitchCandidate;


void showAllPitches(std::vector<PitchCandidate > pitches, std::string path)
{
    cv::VideoCapture vc;
    cv::Mat firstFrame;
    vc.open(path);
    vc.grab();
    vc.retrieve(firstFrame);

    if (firstFrame.empty())
    {
        printf("Could not open the first frame");
    }


    unsigned int count = 0;
    for (int i = 0; i < pitches.size(); i++)
    {
        while (count != pitches[i].start-1)
        {
            count++;
            vc.grab();
        }

        std::vector<cv::Mat > pitchFrames;
        PitchComposite comp(firstFrame.cols, firstFrame.rows, CV_8UC3);
        while (count != pitches[i].end)
        {
            count++;
            cv::Mat frame;
            if (!vc.grab() || !vc.retrieve(frame) || frame.empty())
            {
                printf("An error occurred getting frame");
                break;
            }

            cv::Mat frameClone = frame.clone();

            cv::imshow("Segmented Video", frameClone);
            //cv::waitKey();

            // We write the image later.
            pitchFrames.push_back(frameClone);

            if (count == pitches[i].post)
            {
                comp.setPosition(frameClone, PositionPost);
            }
            else if (count == pitches[i].tee)
            {
                comp.setPosition(frameClone, PositionTee);
            }
            else if (count == pitches[i].release)
            {
                comp.setPosition(frameClone, PositionRelease);
            }
            else if (count == pitches[i].follow)
            {
                comp.setPosition(frameClone, PositionFollow);
            }
        }

        cv::imshow("Segments", comp.compose());
        cv::waitKey();
    }
}


#pragma mark MAIN

int main(int argc, const char * argv[])
{
    int totalPitchesFound = 0;
    int fps = 30, skip = 0;
    std::string path;

    switch (argc)
    {
        case 0:
        case 1:
            path = VIDEO_LOC;
            fps = INPUT_VIDEO_FPS;
            skip = SKIP_NUM_FRAMES;
            break;
        case 4:
            skip = atoi(argv[3]);
        case 3:
            fps = atoi(argv[2]);
        case 2:
            path = argv[1];
            break;
        default:
            printf("The input should be '<video path> [<fps> [<start frame>]]'\n");
            return 1;
    }


    if (2 <= argc)
    {
        path = argv[1];
        fps = atoi(argv[2]);
    }

    else
    {
        path = VIDEO_LOC;
        fps = INPUT_VIDEO_FPS;
    }

    // |pitches| is what we use to generate composites at the end!
    std::vector<PitchCandidate> pitches;

    cv::HOGDescriptor hogd;
    hogd.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
    cv::VideoCapture vc;
    cv::Mat firstFrame;
    vc.open(path);

    printf("Frames per second %0.2f\n", vc.get(CV_CAP_PROP_FPS));

    //vc.open(0);
    vc.grab();
    vc.retrieve(firstFrame);
//    const cv::Point2i roiOrigin = { 2 * firstFrame.cols / 7, 0};
//    const cv::Size2i roiSize = { 4 * firstFrame.cols / 7, firstFrame.rows};
//    const cv::Rect roi = cv::Rect(roiOrigin, roiSize);

    double ffscale = 480.0 / firstFrame.rows;
    int ffheight = 480;
    int ffwidth = ffscale*firstFrame.cols;

    if (0 != ffwidth % 2)
    {
        ffwidth -= 1;
    }

    cv::resize(firstFrame, firstFrame, cv::Size(ffwidth,ffheight));

    const cv::Rect roi = cv::Rect(cv::Point2i(0,0),cv::Size2i(firstFrame.cols, firstFrame.rows));
    OpticalFlow oflow = OpticalFlow(firstFrame, roi);
    firstFrame.deallocate();


    //std::string file;
    //std::getline(std::cin, file);

    cv::namedWindow("Computer Vision", CV_WINDOW_KEEPRATIO);
    cv::namedWindow("Time Motion", CV_WINDOW_KEEPRATIO);

    std::vector<std::pair<unsigned int, unsigned int>> pitch;
    std::vector<MotionVec> timeMotionVecVec;
    std::vector<MotionVec> historyMotionVecVec;
    std::vector<cv::Mat> cachedFrames;
    while (vc.grab())
    {
        const unsigned int frameNumber = countFrame();
        //timeMotionVecVec.push_back({{{0.f,0.f},{0.f,0.f}}});

        printf("Frame number %d\n", frameNumber);

        // This is for convienence.
        if (skip > frameNumber)
        {
            continue;
        }

        if (fps == 60 && 0 == frameNumber%2)
        {
            continue;
        }

        cv::Mat frame;
        if (!vc.retrieve(frame) || frame.empty())
        {
            break;
        }
        else
        {
            cv::resize(frame, frame, cv::Size(ffwidth, ffheight));
        }

        // perform optical flow.
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

        // Push the thresholded values into an array of all motion vectors.
        timeMotionVecVec.push_back(mvThresholded);

        for (int i = 0; i < mvThresholded.size(); i++)
        {
            mvHistory.push_back(mvThresholded[i]);
            mvDrawn.push_back(mvThresholded[i]);
        }

        const size_t stop = timeMotionVecVec.size()-1;
        long i = stop - MOTION_HISTORY_IMAGE;
        for (long i = stop > MOTION_HISTORY_IMAGE? stop-MOTION_HISTORY_IMAGE : 0; stop != i; i++)
        {
            for (int j = 0; j < timeMotionVecVec[i].size(); j++)
            {
                mvHistory.push_back(timeMotionVecVec[i][j]);
                mvDrawn.push_back(timeMotionVecVec[i][j]);
            }
        }

        // Push the motion history into the history vector.
        historyMotionVecVec.push_back(mvHistory);


        // Find the pitcher.
        cv::Rect rectHuman(0,0,0,0);
        //cv::Rect rectHuman = detectFace(frame);
        bool foundNewRectangle = false;
        // divide by zero error mitigation.
        static cv::Rect drawnBorder1(0,0,1,1), drawnBorder2(0,0,1,1), drawnBorder3(0,0,1,1);
        static unsigned int framesSinceStationary = 0;
        double displacement = 0;

        if (pitcherStationary(drawnBorder1, historyMotionVecVec.back()))
        {
            rectHuman = findValidHuman(hogd, frame);
        }

        const double rectShrinkTolerance = 0.75;
        const bool validNewRect = double(rectHuman.height) / drawnBorder1.height > rectShrinkTolerance;
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


        static bool foundLegUpBeforePost = false,
                    foundPost = false,
                    foundTee = false,
                    foundRelease = false,
                    foundFollow = false;

        static int validLegUpCount = 0;

        static unsigned int startFrameIndex = 0,
                            postFrameIndex = 0,
                            teeFrameIndex = 0,
                            releaseFrameIndex = 0,
                            followFrameIndex = 0;

        static unsigned int cachedPostIndex = 0,
                            cachedTeeIndex = 0,
                            cachedReleaseIndex = 0,
                            cachedFollowIndexFuture = 0;


        static cv::Mat positions[4];


        if (pitcherStationary(drawnBorder1, mvHistory))
        {
            framesSinceStationary = 0;
            startFrameIndex = (unsigned int)timeMotionVecVec.size();

            cachedFrames.clear();
            cachedFrames.push_back(frame.clone());

            foundLegUpBeforePost = false;
            foundPost = false;
            foundTee = false;
            foundRelease = false;
            foundFollow = false;

            cachedPostIndex = 0;
            cachedTeeIndex = 0;
            cachedReleaseIndex = 0;
            cachedFollowIndexFuture = 0;

            validLegUpCount = 0;
        }
        else
        {
            framesSinceStationary++;
            cachedFrames.push_back(frame.clone());


            // The first leg up must occur quickly.
            const int stationaryNoiseBufferIndex = 3;
            const int maximumLegUpIndex = 10;
            bool withinLegUpWindow = framesSinceStationary > stationaryNoiseBufferIndex &&
                                     framesSinceStationary < maximumLegUpIndex;

            if (!foundLegUpBeforePost && withinLegUpWindow)
            {
                if (pitcherLegUp(historyMotionVecVec.back(), drawnBorder1, 0.75))
                {
                    ++validLegUpCount;

                    printf("Found leg up %d\n", validLegUpCount);

                    const int minValidLegUpCount = 4;
                    if (minValidLegUpCount == validLegUpCount)
                    {
                        foundLegUpBeforePost = true;
                    }
                }
            }

            // FIND THE POST POSITION
            if (foundLegUpBeforePost && !foundPost)
            {
                if (pitcherPastPost(historyMotionVecVec.back(), drawnBorder1))
                {
                    unsigned int pitchStart = (unsigned int)historyMotionVecVec.size()-framesSinceStationary;
                    postFrameIndex = findPostFrameIndex(historyMotionVecVec, drawnBorder1, pitchStart);
                    //postFrameNumber += SKIP_NUM_FRAMES;
                    if (0 != postFrameIndex)
                    {
                        foundPost = true;
                        cachedPostIndex = postFrameIndex - pitchStart;
                    }
                }
            }


            if (foundPost && !foundRelease)
            {
                if (pitcherLegUp(timeMotionVecVec.back(), drawnBorder1, 0.45))
                {
                    printf("Found leg up again");
                    foundRelease = true;

                    cachedReleaseIndex = framesSinceStationary;
                    cachedTeeIndex = framesSinceStationary - 8;
                    cachedFollowIndexFuture = framesSinceStationary + 8;

//                    releaseFrameIndex = (unsigned int)timeMotionVecVec.size()*(fps/30);
//                    teeFrameIndex = (releaseFrameIndex - 8)*(fps/30);
//                    followFrameIndex = (releaseFrameIndex + 8)*(fps/30);
//
//                    PitchCandidate pc = {SKIP_NUM_FRAMES + startFrameIndex*(fps/30),
//                        SKIP_NUM_FRAMES + postFrameIndex*(fps/30),
//                        SKIP_NUM_FRAMES + teeFrameIndex,
//                        SKIP_NUM_FRAMES + releaseFrameIndex,
//                        SKIP_NUM_FRAMES + followFrameIndex,
//                        SKIP_NUM_FRAMES + (unsigned int)(timeMotionVecVec.size() + 30)*(fps/30)};
//                    pitches.push_back(pc);

                }
            }

            if (foundRelease && framesSinceStationary == cachedFollowIndexFuture)
            {
                totalPitchesFound++;
                positions[0] = cachedFrames[cachedPostIndex];
                positions[1] = cachedFrames[cachedTeeIndex];
                positions[2] = cachedFrames[cachedReleaseIndex];
                positions[3] = cachedFrames[framesSinceStationary];

                for (int i = 0; i < 4; i++)
                {
                    printf("Key position %d", i+1);
                    cv::imshow("Key Positions", positions[i]);
                    cv::waitKey();
                }
            }

        }

        drawBWRectangle(frame, drawnBorder1, true);
        drawBWRectangle(frame, drawnBorder2, false);
        drawBWRectangle(frame, drawnBorder3, true);



        // Name the saved image
        char outPath[50];
        bzero(outPath, 50);
        sprintf(outPath, "%sframe%06d.png", WRITE_ABSOLUTE_DIR, frameNumber);

        //cv::Mat frameClone = frame;
        //cv::imshow("Time Motion", oflow.drawMotion(timeMotionVecVec.back(), frameClone));

        // DRAW THE COMPASS.
        for (unsigned int i = 0; i < COMPASS_ROSE; i++)
        {
            mvDrawn.push_back(compass[i]);
        }

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

        bzero(text, 100);
        sprintf(text, "Pitches Found: %d", totalPitchesFound);
        cv::putText(outputFrame, text, cv::Point(0,70), CV_FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,0,0));

        cv::imshow("Computer Vision", outputFrame);

//        #if WRITE_VIDEO_OUTPUT_AND_CLEAN_DIR
//        bzero(text, 100);
//        sprintf(text, "Displacement: %0.2f pixels", displacement);
//        cv::putText(outputFrame, text, cv::Point(0, 50), CV_FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,0,0));
//        cv::imwrite(cv::string(outPath), outputFrame);
//        printf("Wrote:%s\n", outPath);
//        #endif

        printf("Done with frame %d\n", frameNumber);
    }

    printf("CYYOUNG COMPLETED SUCCESSFULLY\n");

    // Stitch all the frames together and output into a new video.
//    #if WRITE_VIDEO_OUTPUT_AND_CLEAN_DIR
//    // Use ffmpeg to stitch the frames together.
//    char ffmpegSystemCall[500];
//    bzero(ffmpegSystemCall, 500);
//    sprintf(ffmpegSystemCall, "ffmpeg -framerate 15 -i %sframe%%06d.png -c:v libx264 -r 15 -pix_fmt yuv420p %s%s",
//            WRITE_ABSOLUTE_DIR, WRITE_ABSOLUTE_DIR, OUTPUT_VIDEO_NAME);
//    system(ffmpegSystemCall);
//
//    // Clean up the folder
//    bzero(ffmpegSystemCall, 500);
//    sprintf(ffmpegSystemCall, "rm %sframe*.png", WRITE_ABSOLUTE_DIR);
//    #endif

    return 0;
}
