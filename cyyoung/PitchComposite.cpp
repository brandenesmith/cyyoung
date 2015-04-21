//
//  PitchComposite.cpp
//  cyyoung
//
//  Created by MATHEW RENY on 3/29/15.
//  Copyright (c) 2015 MATHEW RENY. All rights reserved.
//

#include "PitchComposite.h"


PitchComposite::PitchComposite(const int inputWidth,
                               const int inputHeight,
                               const int inputChannelType)
{
    this->imageWidth = inputWidth;
    this->imageHeight = inputHeight;
    this->imageChannelType = inputChannelType;
}

void PitchComposite::setPosition(cv::Mat image, Position p)
{
    if (image.empty())
    {
        printf("Image should not be empty.");
        throw new std::runtime_error("The provided image is empty.");
    }
    else if (image.rows != this->imageHeight)
    {
        printf("Image height supposed to be %d. It was %d.\n", this->imageHeight, image.rows);
        throw new std::runtime_error("The provided image doesn't have the correct height.");
    }
    else if (image.cols != this->imageWidth)
    {
        printf("Image width supposed to be %d. It was %d.\n", this->imageWidth, image.cols);
        throw new std::runtime_error("The provided image doesn't have the correct width.");
    }

    switch (p)
    {
        case PositionPost:
        case PositionTee:
        case PositionRelease:
        case PositionFollow:
            this->images[p] = image;
            return;
        default:
            throw new std::runtime_error("Invalid position");
    }
}

cv::Mat PitchComposite::compose()
{
    // make sure all images were set.
    for (int p = PositionFirst; p < PositionCount; p++)
    {
        if (images[p].empty())
        {
            return cv::Mat(); // return an empty matrix.
        }
    }

    cv::Mat composite(this->imageHeight, this->imageWidth*4, this->imageChannelType);
    for (int p = PositionFirst; p < PositionCount; p++)
    {
        // the first image will have offset 0. the second will have an offset
        // of 1280, the third will have an offset of 2560...
        int offset = p*this->imageWidth;

        // copy 1 column at a time into the composite at the correct offset.
        for (int c = 0; c < this->imageWidth; c++) // lol c++.
        {
            images[p].col(c).copyTo(composite.col(c + offset));
        }
    }

    return composite;
}