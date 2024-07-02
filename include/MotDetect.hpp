//
// Created by root on 5/21/24.
//

#ifndef ECHOVISIONRKTOOLKIT2_MOTDETECT_HPP
#define ECHOVISIONRKTOOLKIT2_MOTDETECT_HPP

#include <opencv2/opencv.hpp>
#include <iostream>
#include "globaltypes.h"



using namespace std;

class motionDetector{
public:
    cv::Mat frame, fgMask;
    cv::Ptr<cv::BackgroundSubtractor> pBackSub = cv::createBackgroundSubtractorMOG2();


    vector<bbox_t> simpleMog2(cv::Mat _frame,string prefix)
    {
        pBackSub->apply(_frame, fgMask,MOG_LEARN_RATE);
        erode(fgMask, fgMask, cv::Mat());
        dilate(fgMask, fgMask, cv::Mat());

        std::vector<std::vector<cv::Point>> contours;
        findContours(fgMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        std::vector<bbox_t> motionvector;
        for(int i=0; i<contours.size(); ++i)
        {
            cv::Rect boundingBox = boundingRect(contours[i]);
            if(boundingBox.area() > MIN_BBOX_AREA && boundingBox.height > MIN_OBJEC_HEIGHT) {
                rectangle(fgMask, boundingBox, cv::Scalar(255, 255, 255), 2);
                bbox_t result;
                result.x = ((float)boundingBox.x*dispcoef);
                result.y = ((float)boundingBox.y*dispcoef);
                result.w = ((float)boundingBox.width*dispcoef);
                result.h = ((float)boundingBox.height*dispcoef);
                motionvector.push_back(result);

                //  int mx = result.x + result.w/2;
               //   int my = result.y + result.h/2;
               // cv::circle(fgMask,cv::Point (mx,my),5,cv::Scalar(0,255,0),1,cv::LINE_8,0);
            }
        }
         // cv::imshow(prefix, fgMask);
        return motionvector;
    }

    motionDetector()
    {

    }




};

#endif //ECHOVISIONRKTOOLKIT2_MOTDETECT_HPP
