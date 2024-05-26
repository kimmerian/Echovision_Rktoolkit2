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

    vector<bbox_t> detectRectanglesInsideObjects(Mat frame,vector<bbox_t> objects)
    {
        pBackSub->apply(frame, fgMask);
        erode(fgMask, fgMask, cv::Mat());
        dilate(fgMask, fgMask, cv::Mat());

        std::vector<std::vector<cv::Point>> contours;
        findContours(fgMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        vector<bbox_t> boxes ={};

        if(objects.size() == 0){return boxes;}

        for(int t= 0;t<objects.size();t++)
        {
            bbox_t box = objects[t];
            for(int n = 0;n< contours.size();n++)
            {
                Rect motionBox = boundingRect(contours[n]);
                double area = contourArea(contours[n]);
                if(area > 500) {
                    int x = motionBox.x + motionBox.width / 2;
                    int y = motionBox.y + motionBox.height / 2;
                    if (
                            (x > box.x) && (x < box.x + box.w) &&
                            (y > box.y) && (y < box.y + box.h)
                            ) {
                        bbox_t point;
                        point.x = motionBox.x;
                        point.y = motionBox.y;
                        point.w = motionBox.width;
                        point.h = motionBox.height;
                        point.track_id = box.track_id;
                        point.obj_id = box.obj_id;
                        boxes.push_back(point);
                        break;
                    }
                }
            }
        }
        return boxes;
    }

};

#endif //ECHOVISIONRKTOOLKIT2_MOTDETECT_HPP
