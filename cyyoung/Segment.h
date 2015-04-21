//
//  SegmentedImage.h
//  cyyoung
//
//  Created by MATHEW RENY on 3/21/15.
//  Copyright (c) 2015 MATHEW RENY. All rights reserved.
//

#ifndef __cyyoung__Segment__
#define __cyyoung__Segment__

#include <stdio.h>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>


// This class is a lightweight container for the information related
// to a pitch. It is a dumb class.
class Segment
{
public:
    // Corresponds to the frame number.
    unsigned long start, end;
    Segment(unsigned long start, unsigned long end);
};

typedef enum {
    // Helper enumerated types to make code reading easier.
    PositionFirst = 0,
    PositionLast = 3,

    // The actual pitch positions.
    PositionPost = 0,
    PositionTee = 1,
    PositionRelease = 2,
    PositionFollowThrough = 3,
} PitchPosition;

class SegmentedImage
{
private:
    // Contains all the images corresponding to the valid pitch positions. The pitch positions are
    // indexed based on the enumerated type `PitchPosition`.
    cv::Mat images[PositionLast+1];

public:
    // Initializes the segmented image.
    SegmentedImage();
    SegmentedImage(cv::Mat post, cv::Mat tee, cv::Mat release, cv::Mat followThrough);

    // Writes the provided image into the `images` array at the provided pitch position `pp`.
    void setPositionImage(PitchPosition pp, cv::Mat image);

    // Creates a composite image by combining all the pitch positions together. Don't worry about releasing
    // this
    // Returns an empty cv::Mat if the image is not complete yet. (fooMat.empty() == true)
    cv::Mat compositImage();
};

// This class contains the video of the pitch in question.
class Video
{
private:
    cv::VideoCapture vc;

public:
    // Should contain no spaces.
    std::string path;

    // The current frame to return when `nextFrame()` is called.
    unsigned long frameCounter;

    // Construct a video at the specified path beginning with
    // the UNIX root.
    Video(std::string path);


    // contains all the segments for the video.
    std::vector<Segment> videos;

    // contains all the segmented images;
    std::vector<SegmentedImage> images;

    // Chops the video up into the segments with names based on the index of the
    // segment in the `segments` vector. Segments are written to the same directory
    // as the input video.
    bool writeSegments();

    // Writes the images to the file system at the same directory as `path`.
    bool writeImages();

    // Returns the next frame in the video and increments `frameCounter`.
    bool nextFrame(cv::Mat &frame);

    bool resetToBeginning();
};

#endif /* defined(__cyyoung__Segment__) */
