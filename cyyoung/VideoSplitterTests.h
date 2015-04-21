//
//  VideoSplitterTests.h
//  cyyoung
//
//  Created by MATHEW RENY on 4/4/15.
//  Copyright (c) 2015 MATHEW RENY. All rights reserved.
//

#ifndef __cyyoung__VideoSplitterTests__
#define __cyyoung__VideoSplitterTests__

#include <stdio.h>
#include <opencv2/opencv.hpp>

// Used for test storage.
// Identical on Unix (OS X) and Linux.
// Writable by any user.
// Change to whatever floats your boat. Right now I use this:
//
// $ echo CSCE473project|md5
// 5980d9729d3c89890a99983a5ec0f2a5
cv::string tmpDirPath = "/tmp/5980d9729d3c89890a99983a5ec0f2a5/";
cv::string vidDirPath = "~/BaseBallPitchTestVideos/";
cv::string vidExt = "mp4";
std::vector<cv::string> vidNames = { "video1", "video2", "video3", "video4"};


#pragma mark TESTS
bool test1SplitOn1Video(cv::string videoPath);
bool test2SplitOn1Video(cv::string videoPath);
bool test3SplitOn1Video(cv::string videoPath);
bool testSplitAtFirstFrameOfVideo(cv::string videoPath);
bool testSplitAtLastFrameOfVideo(cv::string videoPath);
#pragma end TESTS

#pragma mark HELPERS
bool videoEqual(cv::string v1, cv::string v2, cv::Vec2i range);

#pragma end HELPERS

#endif /* defined(__cyyoung__VideoSplitterTests__) */
