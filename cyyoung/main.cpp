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

#define WRITE_TO_FILE_SYSTEM 1


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

// Draws either a black or white rectangle on the frame.
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
cv::Rect largestRectangle(std::vector<cv::Rect> rects)
{
    int maxIndex = 0;
    double maxArea = 0;
    for (int i = 0; i < rects.size(); i++)
    {
        double area = rects[i].width * rects[i].height;
        if (area > maxArea)
        {
            maxArea = area;
            maxIndex = i;
        }
    }

    return rects[maxIndex];
}


// Uses the Histogram of Oriented Gradients to find the pitcher within the frame.
// It only works well when the pitcher is standing straight up and facing the camera.
// Returns a zero rect if no valid human is found.
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

    // In our testing 240p gets the fastest and most accurate results.
    // Keeping the original aspect ratio detects the pitcher less and
    // becomes the biggest bottleneck in the program.
    double scale = 240.0 / gray.rows;
    cv::resize(gray, gray, cv::Size(gray.cols*scale,gray.rows*scale));

    std::vector<cv::Rect> locations;
    hogd.detectMultiScale(gray, locations);

    if (0 != locations.size())
    {
        // Draw every rectangle found so the user can see how many bad rectangles
        // are not fully surrounding the pitcher.
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

        // Scale up the largest location since we scaled down before.
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

// Returns true if the some of all vector magnitudes within mv are less than the width
// of the rectangle. In testing, I have found the width to be a very good number to use.
bool pitcherStationary(cv::Rect rectHuman, MotionVec &mv)
{
    // The feet and top of head are cut off sice they seem to have lots of extraneous movement.
    int shorten = rectHuman.height / 6;
    rectHuman.y += shorten;
    rectHuman.height -= shorten * 2;

    double thresh = rectHuman.width;
    double weight = 0;
    for (int i = 0; i < mv.size(); i++)
    {
        const cv::Point2f p1 = mv[i].first;
        const cv::Point2f p2 = mv[i].second;
        if (movementWithinRectangle(mv[i], rectHuman))
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

//inline double calculateRadians(std::pair<cv::Point2f, cv::Point2f> movement)
//{
//    const double angle = atan2(0 - (movement.second.y - movement.first.y), movement.second.x - movement.first.x);
//    return angle >= 0.0 ? angle : 6.2831853072 + angle;
//}

// Calculates a movement line segment's radians from 0 to 2pi.
inline double calculateRadians(std::pair<cv::Point2f, cv::Point2f> motion)
{
    // Y is at the top left corner and it must be at the bottom left for atan2.
    const double angle = atan2f(0 - (motion.second.y - motion.first.y), motion.second.x - motion.first.x);

    // Atan2 returns negative numbers for angles pointing downwards.
    // I want the angles to be between 0 and 2pi.
    if (angle < 0)
    {
        return 2*3.14159 + angle;
    }
    else
    {
        return angle;
    }
}

// Not inclusive of min and max.
double calculatePercentWithinRectangle(MotionVec &mv, cv::Rect r, double min, double max)
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

    double goodBadRatio = calculatePercentWithinRectangle(history, stationaryRect, min, max);
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
        double percent = calculatePercentWithinRectangle(histories[i],
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


bool pitcherLegUp(MotionVec &mv, cv::Rect stationaryRect, const double minPercentUp)
{
    // valid angles between pi/4 and pi.
    double min = 0.7853981634;
    double max = 3.14159;

    int good = 0;
    int total = 0.001; // solves divide by zero error.

    for (int i = 0; i < mv.size(); i++)
    {
        if (movementWithinRectangle(mv[i] , stationaryRect ))
        {
            total++;

            double angle = calculateRadians(mv[i]);

            if (min < angle && angle < max)
            {
                good++;
            }
        }
    }

    const double percent = double(good)/double(total);
    printf("Up %d, Total %d, Percent %0.2f\n", good, total, percent);

    return percent > minPercentUp;
}

//// Normalizes vectors and adds them together to find the average direction.
//// Thought it would be helpful but I never needed it. I'm keeping it here
//// commented out so you can see the math.
//double averageAngleWithinRect(MotionVec &mv, cv::Rect r)
//{
//    // Create a unit vector and then find its angle.
//    float xtot = 0, ytot = 0;
//
//    for (int i = 0; i < mv.size(); i++)
//    {
//        if (movementWithinRectangle(mv[i], r))
//        {
//            xtot += mv[i].second.x - mv[i].first.x;
//            ytot += mv[i].second.y - mv[i].first.y;
//        }
//    }
//    return correctRadians({cv::Point2f(0.f,0.f),cv::Point2f(xtot,ytot)});
//}

// Fills |raw| and |history| using mv.
void processMotion(MotionVec mv,
                   const double minimumDistance,
                   std::vector<MotionVec> &raw,
                   const unsigned int historyCount,
                   std::vector<MotionVec> &history)
{
    raw.push_back(MotionVec());
    history.push_back(MotionVec());

    // Throw out vectors too small.
    for (int i = 0; i < mv.size(); i++)
    {
        if (minimumDistance <= distance(mv[i].first,
                                        mv[i].second))
        {
            raw.back().push_back(mv[i]);
            history.back().push_back(mv[i]);
        }
    }

    const size_t stop = raw.size()-1;
    for (size_t i = stop > historyCount? stop-historyCount : 0; stop != i; ++i)
    {
        for (int j = 0; j < raw[i].size(); ++j)
        {
            // Put the motion history into the history vector.
            history.back().push_back(raw[i][j]);
        }
    }
}

cv::Rect pitcherMoundPosition(cv::Mat &frame,
                              cv::HOGDescriptor &hogd,
                              std::vector<MotionVec> &history)
{

    // Find the pitcher.
    cv::Rect rectHuman(0,0,0,0);
    static cv::Rect cachedRectHuman = cv::Rect(0,0,1,1);

    // Save time by not finding the human if there is motion. When watching the output, I noticed
    // that the hog never captures the human during movement. It does well capturing humans when
    // they are in an upright position such as when they are about to enter the post position.
    if (pitcherStationary(cachedRectHuman, history.back()))
    {
        rectHuman = findValidHuman(hogd, frame);
    }

    const bool humanRectNotZero = rectHuman.x != 0 && rectHuman.y != 0 && rectHuman.width != 0 && rectHuman.height != 0;
    if (humanRectNotZero)
    {
        // Make sure the rectangle hasn't shrunken by an impossible amount. This sometimes happens with the HOG.
        // It will highlight the upper torso or a leg or the head. It's unpredictable.
        const double maxRectShrinkPercent = 0.75;
        const double currentRectShrinkPercent = double(rectHuman.height) / cachedRectHuman.height;
        if (currentRectShrinkPercent > maxRectShrinkPercent)
        {
            cachedRectHuman = rectHuman;
        }
    }

    return cachedRectHuman;
}

cv::Mat drawOutputFrame(cv::Mat &frame,
                        cv::Rect rectHuman,
                        std::vector<MotionVec> &history,
                        OpticalFlow &oflow,
                        unsigned int frameNumber,
                        unsigned int framesSinceStationary,
                        unsigned int totalPitchesFound)
{

    // DRAW THE COMPASS.
    MotionVec mvDrawn = history.back();
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
    cv::putText(outputFrame, text, cv::Point(0,60), CV_FONT_HERSHEY_PLAIN, 2, cv::Scalar(0,0,0));

    // Put a border around the human on the pitchers mound.
    cv::Rect borderInside, borderMiddle, borderOutside;
    borderInside = rectHuman;
    borderMiddle = rectHuman;
    borderMiddle.x -= 1;
    borderMiddle.y -= 1;
    borderMiddle.width += 2;
    borderMiddle.height += 2;
    borderOutside = rectHuman;
    borderOutside.x -= 2;
    borderOutside.y -= 2;
    borderOutside.width += 4;
    borderOutside.height += 4;
    drawBWRectangle(outputFrame, borderInside, true);
    drawBWRectangle(outputFrame, borderMiddle, false);
    drawBWRectangle(outputFrame, borderOutside, true);

    return outputFrame;
}


#pragma mark MAIN

int main(int argc, const char * argv[])
{
    int totalPitchesFound = 0;
    std::string path;

    if (0 != system("which -s ffmpeg"))
    {
        printf("Fatal warning\n");
        printf("ffmpeg must be installed and reachable by the $PATH environment variable.\n");
        printf("Download and install the free and open source program here:\n");
        printf("https://www.ffmpeg.org/download.html\n");
        printf("Or use the homebrew package manager (the method I prefer)\n");
        printf("$ brew install ffmpeg –with-faac -with-fdk-aac\n");
        exit(1);
    }

    switch (argc)
    {
        default:
        case 1:
            printf("Please run from the command line like so:\n");
            printf("$ ./cyyoung <video_path>\n");
            printf("The second two arguments are optional.\n");
            printf("Videos will be written to <video_path>.pitch%%d.mp4\n");
            printf("Image segments will be written to <video_path>.pitch%%d.position%%d.png\n");
            exit(0);
            break;
        case 2:
            path = argv[1];
            break;
    }

    cv::HOGDescriptor hogd;
    hogd.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());


    cv::VideoCapture vc;
    cv::Mat firstFrame;
    vc.open(path);
    vc.grab();
    vc.retrieve(firstFrame);

    // Resize the frame to 480p. 'ff' here stands fo first frame.
    double f1scale = 480.0 / firstFrame.rows;
    int f1height = 480;
    int f1width = f1scale*firstFrame.cols;

    // Make sure the width divides by two evenly. It mitigates bug I ran into.
    if (0 != f1width % 2)
    {
        f1width -= 1;
    }

    cv::resize(firstFrame, firstFrame, cv::Size(f1width,f1height));

    // The roi is legacy code that I don't want to break yet.
    const cv::Rect roi = cv::Rect(cv::Point2i(0,0),cv::Size2i(firstFrame.cols, firstFrame.rows));
    OpticalFlow oflow = OpticalFlow(firstFrame, roi);
    firstFrame.deallocate();

    cv::namedWindow("Computer Vision", CV_WINDOW_KEEPRATIO);
    cv::namedWindow("Key Positions",   CV_WINDOW_KEEPRATIO);

    // Used for creating a clip and segmented image
    std::vector<cv::Mat> cachedFrames;
    unsigned int cachedPostIndex         = 0,
                 cachedTeeIndex          = 0,
                 cachedReleaseIndex      = 0,
                 cachedFollowIndexFuture = 0,
                 cachedLegUpCount        = 0;

    // Allows the program to know the start of a pitch.
    unsigned int framesSinceStationary = 0;

    // These states have been shown to eliminate greater than 99% of false positives.
    bool foundLegUpBeforePost = false,
         foundPost = false,
         foundRelease = false;

    // Where the raw captured optical flow motion goes.
    std::vector<MotionVec> timeMotionVecVec;

    // Combined optical flow motion. Used to find leg up and leg down.
    std::vector<MotionVec> historyMotionVecVec;

    // Iterate through the video frame by frame.
    while (vc.grab())
    {
        const unsigned int frameNumber = countFrame();

        printf("Frame number %d\n", frameNumber);

        cv::Mat frame;
        if (!vc.retrieve(frame) || frame.empty())
        {
            printf("CYYOUNG couldn't read next frame.\n");
            exit(1);
            return 1;
        }
        else
        {
            // This size speeds up each frame considerably.
            cv::resize(frame, frame, cv::Size(f1width, f1height));
        }

        // perform optical flow.
        processMotion(oflow.feed(frame),
                      MIN_TRACK_DISTANCE,
                      timeMotionVecVec,
                      MOTION_HISTORY_IMAGE,
                      historyMotionVecVec);

        // The rectangle on the pitcher mound surrounding the pitcher.
        cv::Rect pitcherMoundBound = pitcherMoundPosition(frame, hogd, historyMotionVecVec);


        // The pitcher will not be stationary for the duration of the pitch. In the future, this should be smarter and take into
        // account the chance that the pitcher pauses at the top of the post. Every instance where a pitch isn't captured occured
        // when the pitcher pauses at the top of the post. This is something to look into but it adds too much stateism to the
        // program and dramatically complicates things.  This would also draw in more false positives and I'm happy with the less
        // than one percent false positive rate the algorithm is currently achieving.
        if (pitcherStationary(pitcherMoundBound, historyMotionVecVec.back()))
        {
            framesSinceStationary   = 0;
            cachedLegUpCount        = 0;
            cachedPostIndex         = 0;
            cachedTeeIndex          = 0;
            cachedReleaseIndex      = 0;
            cachedFollowIndexFuture = 0;

            cachedFrames.clear();
            cachedFrames.push_back(frame.clone());

            foundLegUpBeforePost = false;
            foundPost            = false;
            foundRelease         = false;
        }
        else
        {
            framesSinceStationary++;
            cachedFrames.push_back(frame.clone());


            // The first leg up must occur quickly. If it doesn't, then anything past the
            // following if statement will not run.
            const int maximumLegUpIndex = 10;
            const int stationaryNoiseBufferIndex = 3;
            bool withinLegUpWindow = framesSinceStationary > stationaryNoiseBufferIndex &&
                                     framesSinceStationary < maximumLegUpIndex;

            if (!foundLegUpBeforePost && withinLegUpWindow)
            {
                if (pitcherLegUp(historyMotionVecVec.back(), pitcherMoundBound, 0.75))
                {
                    ++cachedLegUpCount;

                    const int minValidLegUpCount = 4;

                    printf("Found leg up %d of %d\n", cachedLegUpCount, minValidLegUpCount);
                    if (minValidLegUpCount == cachedLegUpCount)
                    {
                        foundLegUpBeforePost = true;
                    }
                }
            }

            // FIND THE POST POSITION
            if (foundLegUpBeforePost && !foundPost)
            {
                if (pitcherPastPost(historyMotionVecVec.back(), pitcherMoundBound))
                {
                    size_t pitchStart = historyMotionVecVec.size()-framesSinceStationary;
                    size_t postIndex = findPostFrameIndex(historyMotionVecVec,
                                                          pitcherMoundBound,
                                                          (unsigned int)pitchStart);
                    if (0 != postIndex)
                    {
                        foundPost = true;
                        cachedPostIndex = (unsigned int)(postIndex - pitchStart);
                    }
                }
            }

            // Find release.
            if (foundPost && !foundRelease)
            {
                if (pitcherLegUp(timeMotionVecVec.back(), pitcherMoundBound, 0.45))
                {
                    foundRelease = true;

                    // this number was chosen by testing.
                    const int teeFollowOffset = 8;

                    printf("Found leg up again. In %d frames the pitch is clipped.\n", teeFollowOffset);
                    cachedReleaseIndex      = framesSinceStationary;
                    cachedTeeIndex          = framesSinceStationary - teeFollowOffset;
                    cachedFollowIndexFuture = framesSinceStationary + teeFollowOffset;
                }
            }

            // When the following is true we have successully reached all segments of a pitch.
            // I output a clip and image segments to the file system.
            if (foundRelease && cachedFollowIndexFuture == framesSinceStationary)
            {
                totalPitchesFound++;
                printf("Pitch %d has been found.\n", totalPitchesFound);

                cv::Mat positions[4];
                positions[0] = cachedFrames[cachedPostIndex];
                positions[1] = cachedFrames[cachedTeeIndex];
                positions[2] = cachedFrames[cachedReleaseIndex];
                positions[3] = cachedFrames[framesSinceStationary];

                for (int i = 0; i < 4; i++)
                {
                    printf("Key position %d\n", i+1);
                    cv::imshow("Key Positions", positions[i]);

                    #if WRITE_TO_FILE_SYSTEM
                    // Save all the pitch positions.
                    char positionPath[path.size()+50];
                    bzero(positionPath, path.size()+50);
                    sprintf(positionPath, "%s.pitch%d.position%d.png", path.c_str(), totalPitchesFound, i+1);
                    cv::imwrite(positionPath, positions[i]);
                    printf("Wrote: %s\n", positionPath);
                    #endif

                    printf("\nPress any key to continue.\n");
                    cv::waitKey();
                }

                #if WRITE_TO_FILE_SYSTEM
                const size_t outPathSize = path.size()+50;
                const size_t numCachedFrames = cachedFrames.size();
                for (int i = 0; i < numCachedFrames; i++)
                {
                    char outPath[outPathSize];
                    bzero(outPath, outPathSize);
                    sprintf(outPath, "%s.%d.frame%06d.png", path.c_str(), totalPitchesFound, i+1);
                    cv::imwrite(outPath, cachedFrames[i]);
                    printf("Wrote: %s\n", outPath);
                }
                cv::Mat frameBuffer;
                const size_t stop = numCachedFrames + 30;
                for (size_t i = numCachedFrames; i < stop && vc.grab() && vc.retrieve(frameBuffer) && !frameBuffer.empty(); ++i)
                {
                    countFrame();
                    cv::resize(frameBuffer, frameBuffer, cv::Size(f1width,f1height));
                    char outPath[outPathSize];
                    bzero(outPath, outPathSize);
                    sprintf(outPath, "%s.%d.frame%06d.png", path.c_str(), totalPitchesFound, int(i+1));
                    cv::imwrite(outPath, frameBuffer);
                    printf("Wrote: %s\n", outPath);
                }

                // Use ffmpeg to stitch the frames together into a clip.
                char ffmpegSystemCall[path.size()+500];
                bzero(ffmpegSystemCall, path.size()+500);
                sprintf(ffmpegSystemCall,
                        "ffmpeg -framerate 15 -i %s.%d.frame%%06d.png -c:v libx264 -r 15 -pix_fmt yuv420p %s.pitch%d.mp4",
                        path.c_str(),
                        totalPitchesFound,
                        path.c_str(),
                        totalPitchesFound);
                if (0 != system(ffmpegSystemCall))
                {
                    printf("Something might have went wrong when invoking ffmpeg.\n");
                }
                printf("Wrote: %s.pitch%d.mp4\n", path.c_str(), totalPitchesFound);


                // Clean up the intermediate images.
                char rmSystemCall[outPathSize];
                bzero(rmSystemCall, outPathSize);
                sprintf(rmSystemCall, "rm %s.%d.frame*.png", path.c_str(), totalPitchesFound);
                if (0 != system(rmSystemCall))
                {
                    printf("Error cleaning up intermediate frames.\n");
                }
                else
                {
                    printf("Successfully removed intermediate pitch frames.\n");
                }
                #endif
            }
        }

        // Make a pretty output image to show the user that the program is working correctly.
        // I don't want to hide the magic!
        cv::Mat show = drawOutputFrame(frame,
                                       pitcherMoundBound,
                                       historyMotionVecVec,
                                       oflow,
                                       frameNumber,
                                       framesSinceStationary,
                                       totalPitchesFound);

        cv::imshow("Computer Vision", show);
    }

    printf("CYYOUNG COMPLETED SUCCESSFULLY\n");
    exit(0);
    return 0;
}
