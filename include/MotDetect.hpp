//
// Created by root on 5/21/24.
//

#ifndef ECHOVISIONRKTOOLKIT2_MOTDETECT_HPP
#define ECHOVISIONRKTOOLKIT2_MOTDETECT_HPP

#include <opencv2/opencv.hpp>
#include <iostream>
#include "globaltypes.h"

using namespace cv;
using namespace std;

class motionDetector{
public:
    cv::Mat frame, fgMask;
    cv::Ptr<cv::BackgroundSubtractor> pBackSub = cv::createBackgroundSubtractorMOG2();
    motionDetector()
    {

    }
    Rect mergeRect(const Rect& boxA, const Rect& boxB)
    {
        int x = min(boxA.x, boxB.x);
        int y = min(boxA.y, boxB.y);
        int width = max(boxA.x + boxA.width, boxB.x + boxB.width) - x;
        int height = max(boxA.y + boxA.height, boxB.y + boxB.height) - y;
        return Rect(x, y, width, height);
    }


    void mergeOverlappingRects(vector<Rect>& rects, vector<Rect>& mergedRects, int gridSize) {
        map<pair<int, int>, vector<int>> grid;
        for (int i = 0; i < rects.size(); i++) {
            int xCell = rects[i].x / gridSize;
            int yCell = rects[i].y / gridSize;
            grid[{xCell, yCell}].push_back(i);
        }

        vector<bool> merged(rects.size(), false);

        for (auto& cell : grid) {
            vector<int>& indices = cell.second;
            for (int i = 0; i < indices.size(); i++) {
                if (merged[indices[i]])
                    continue;
                Rect mergedBox = rects[indices[i]];
                for (int j = i + 1; j < indices.size(); j++) {
                    if (merged[indices[j]])
                        continue;
                    Rect intersect = mergedBox & rects[indices[j]];
                    if (intersect.area() > 0) {
                        mergedBox = mergeRect(mergedBox, rects[indices[j]]);
                        merged[indices[j]] = true;
                    }
                }
                mergedRects.push_back(mergedBox);
            }
        }
    }

    vector<bbox_t> motdet(Mat frame)
    {
        pBackSub->apply(frame, fgMask);
        erode(fgMask, fgMask, cv::Mat());
        dilate(fgMask, fgMask, cv::Mat());

        std::vector<std::vector<cv::Point>> contours;
        findContours(fgMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        vector<Rect> boxes;

        for (size_t i = 0; i < contours.size(); i++) {
            double area = contourArea(contours[i]);
            if (area >= 200) {
                Rect boundingBox = boundingRect(contours[i]);
                boxes.push_back(boundingBox);
            }
        }

        // Dikdörtgenleri birleştir
        vector<Rect> mergedBoxes;
        int gridSize = 50;
        mergeOverlappingRects(boxes, mergedBoxes, gridSize);

        vector<bbox_t> resultvec;

        for (size_t i = 0; i < mergedBoxes.size(); i++) {
            rectangle(frame, mergedBoxes[i], Scalar(0, 255, 0), 2);


            bbox_t box;
            box.x= mergedBoxes[i].x;
            box.y= mergedBoxes[i].y;
            box.w= mergedBoxes[i].width;
            box.h= mergedBoxes[i].height;
            box.obj_id= 79;

            box.color="b";
            resultvec.push_back(box);
        }
        cout <<  mergedBoxes.size() << endl;

        return resultvec;
    }

};

#endif //ECHOVISIONRKTOOLKIT2_MOTDETECT_HPP
