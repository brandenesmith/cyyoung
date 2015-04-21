//
//  SegmentedImage.cpp
//  cyyoung
//
//  Created by MATHEW RENY on 3/21/15.
//  Copyright (c) 2015 MATHEW RENY. All rights reserved.
//

#include "Segment.h"

#include <strings.h>

Segment::Segment(unsigned long start, unsigned long end)
{
    this->start = start;
    this->end = end;
}

SegmentedImage::SegmentedImage()
{

}

SegmentedImage::SegmentedImage(cv::Mat post, cv::Mat tee, cv::Mat release,  cv::Mat followThrough)
{
    this->setPositionImage(PositionPost, post);
    this->setPositionImage(PositionTee, tee);
    this->setPositionImage(PositionRelease, release);
    this->setPositionImage(PositionFollowThrough, followThrough);
}

void SegmentedImage::setPositionImage(PitchPosition pp, cv::Mat image)
{
    this->images[pp] = image;
}

cv::Mat SegmentedImage::compositImage()
{
    for (PitchPosition i = PositionFirst; i <= PositionLast; i = (PitchPosition)((int)i + 1))
    {
        if (this->images[i].empty())
        {
            return cv::Mat();
        }
    }
    int type = this->images[0].type();
    int rows = this->images[0].rows;
    int cols = (PositionLast+1) * this->images[0].cols;
    cv::Mat composite = cv::Mat(rows, cols, type);

    // Construct the image point by point.  There is probably a better way to do this,
    // but since this isn't production code, I quite frankly do not care.
    for (int img = PositionFirst; img <= PositionLast; img++)
    {
        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            {
                cv::Point2i pOld = cv::Point2i(row, col);
                cv::Point2i pNew = cv::Point2i(row, (1+img)*col);
                composite.at<cv::Point2i>(pNew) = this->images[img].at<cv::Point2i>(pOld);
            }
        }
    }

    return composite;
}


Video::Video(std::string path)
{
    this->path = path;
    this->frameCounter = 0;
    this->videos = std::vector<Segment>();
    this->images = std::vector<SegmentedImage>();

    this->vc = cv::VideoCapture();

    if (!vc.open(path))
    {
        throw std::runtime_error("Could not open video");
    }
}

bool Video::nextFrame(cv::Mat &frame)
{
    return this->vc.read(frame);
}

bool Video::writeSegments()
{
    for (int i = 0; i < this->videos.size(); i++)
    {
        Segment s = this->videos[i];

        cv::VideoCapture c = cv::VideoCapture();
        bool opened = c.open(this->path);
        if (!opened)
        {
            c.release();
            return false;
        }

        // Skip over frames we do not need.
        unsigned long skip = s.start;
        while (skip != 0)
        {
            skip--;
            if (!c.grab())
            {
                c.release();
                return false;
            }
        }

        // These were defined in the constraints section.
        std::ostringstream wpath;
        wpath << this->path << "." << i << ".mp4";
        int h264     = CV_FOURCC('X', '2', '6', '4');
        double fps    = 30;
        cv::Size size = cv::Size(1280,720);
        int isColor   = true;

        cv::VideoWriter w = cv::VideoWriter(wpath.str(), h264, fps, size, isColor);

        // Write the images read in from the capture object.
        unsigned long step = s.end - s.start;
        while (step != 0)
        {
            step--;
            bool grabbed = c.grab();
            if (!grabbed)
            {
                w.release();
                c.release();
                return false;
            }
            cv::Mat next;
            bool retrieved = c.retrieve(next);
            if (!retrieved)
            {
                w.release();
                c.release();
                return false;
            }

            w.write(next);
        }

        w.release();
        c.release();
    }

    return true;
}

bool Video::writeImages()
{
    for (int i = 0; i < this->images.size(); i++)
    {
        std::ostringstream wpath;
        wpath << this->path << "." << i << ".png";

        if (!cv::imwrite(wpath.str(), this->images[i].compositImage()))
        {
            return false;
        }
    }

    return true;
}

bool Video::resetToBeginning()
{
    this->vc.release();
    this->vc = cv::VideoCapture();
    return this->vc.open(this->path);
}














































