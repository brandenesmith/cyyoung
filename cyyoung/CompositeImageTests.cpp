//
//  CompositeImageTests.cpp
//  cyyoung
//
//  Created by MATHEW RENY on 3/29/15.
//  Copyright (c) 2015 MATHEW RENY. All rights reserved.
//

#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

#include "CompositeImageTests.h"
#include "PitchComposite.h"

bool CompositeImageTests::testAllCases()
{
    return //this->testValidUse1() &&
           this->testValidUse2() &&
           this->testNotAllSet() &&
           this->testUnequivolentImagePropertiesUsed();
}

bool CompositeImageTests::testValidUse2()
{
    cv::Mat a = cv::imread("/Users/mahi/Desktop/post.png");
    cv::Mat b = cv::imread("/Users/mahi/Desktop/tee.png");
    cv::Mat c = cv::imread("/Users/mahi/Desktop/release.png");
    cv::Mat d = cv::imread("/Users/mahi/Desktop/follow.png");

//    cv::resize(a, a, cv::Size(,160));
//    cv::resize(b, b, cv::Size(256,160));
//    cv::resize(c, c, cv::Size(256,160));
//    cv::resize(d, d, cv::Size(256,160));


    printf("a rows %d, cols %d\n", a.rows, a.cols);
    printf("b rows %d, cols %d\n", b.rows, b.cols);
    printf("c rows %d, cols %d\n", c.rows, c.cols);
    printf("d rows %d, cols %d\n", d.rows, d.cols);

    printf("a stride %d\n", int(a.step));
    printf("b stride %d\n", int(b.step));
    printf("c stride %d\n", int(c.step));
    printf("d stride %d\n", int(d.step));


    PitchComposite pc(a.cols, a.rows, CV_8UC3);
    pc.setPosition(a, PositionPost);
    pc.setPosition(b, PositionTee);
    pc.setPosition(c, PositionRelease);
    pc.setPosition(d, PositionFollow);

    
    cv::imshow("ValidUse2", pc.compose());

    return true;
}

bool CompositeImageTests::testNotAllSet()
{
    return true;
}

bool CompositeImageTests::testUnequivolentImagePropertiesUsed()
{
    return true;
}

bool CompositeImageTests::testValidUse1()
{
    cv::Mat imgs[4];
    imgs[0] = cv::Mat::zeros(720,1280,CV_8UC3);
    imgs[1] = cv::Mat::zeros(720,1280,CV_8UC3);
    imgs[2] = cv::Mat::zeros(720,1280,CV_8UC3);
    imgs[3] = cv::Mat::zeros(720,1280,CV_8UC3);

    // Seed random number generator. This is the fast one.
    srand((int)time(NULL));

    cv::Vec3b colors[4];
    colors[0] = cv::Vec3b(abs(rand())%256,abs(rand())%256,abs(rand())%256 );
    colors[1] = cv::Vec3b(abs(rand())%256,abs(rand())%256,abs(rand())%256);
    colors[2] = cv::Vec3b(abs(rand())%256,abs(rand())%256,abs(rand())%256);
    colors[3] = cv::Vec3b(abs(rand())%256,abs(rand())%256,abs(rand())%256);

    // Randomly set every pixel.
    for (int i = 0; i < 4; i++)
    {
        for (int r = 0; r < 720; r++)
        {
            for (int c = 0; c < 1280; c++)
            {
                imgs[i].at<cv::Vec3b>(r,c) = colors[i];
            }
        }
    }

    PitchComposite pc(0,0,0);

    pc.setPosition(imgs[PositionPost], PositionPost);
    pc.setPosition(imgs[PositionTee], PositionTee);
    pc.setPosition(imgs[PositionRelease], PositionRelease);
    pc.setPosition(imgs[PositionFollow], PositionFollow);

    cv::Mat composite = pc.compose();

    cv::imshow("Composite random colors image", composite);

    // Now check that the composite is good.
    for (int i = 0; i < 4; i++)
    {
        int offset = i*1280;
        cv::Vec3b color = colors[i];

        printf("Color %d %d %d\n", color[0], color[1], color[2]);

        for (int r = 0; r < 720; r++)
        {
            for (int c = 0; c < 1280; c++)
            {
                cv::Vec3b pixel = composite.at<cv::Vec3b>(r,c+offset);

                //printf ("Pixel %d %d %d\n", pixel[0], pixel[1], pixel[2]);

                if (pixel[0] != color[0])
                {
                    printf("Pixel at row %d col %d chan 1 failed\n", r,c+offset);
                    return false;
                }

                if (pixel[1] != color[1])
                {
                    printf("Pixel at row %d col %d chan 2 failed\n", r,c+offset);
                    return false;
                }

                if (pixel[2] != color[2])
                {
                    printf("Pixel at row %d col %d chan 3 failed\n", r,c+offset);
                    return false;
                }
            }
        }
    }

    return true;
}