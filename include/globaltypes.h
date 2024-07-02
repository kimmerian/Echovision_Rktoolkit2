//
// Created by root on 2/29/24.
//

#ifndef ECHOVISION_NEWCORETEST_GLOBALTYPES_H
#define ECHOVISION_NEWCORETEST_GLOBALTYPES_H

#include <opencv2/opencv.hpp>            // C++
#include <opencv2/highgui/highgui_c.h>   // C
#include <opencv2/imgproc/imgproc_c.h>   // C

#define MIN_OBJEC_HEIGHT 32
#define MIN_BBOX_AREA 128
#define MOG_LEARN_RATE 0.08
#define OVERLAP_RATIO 0.7


typedef struct{
    int r;
    int g;
    int b;
}t_detectedColor;

struct bbox_t {
    float x, y, w, h;       // (x,y) - top-left corner, (w, h) - width & height of bounded box
    float prob;                    // confidence - probability that the object was found correctly
    std::string obj_id;           // class of object - from range [0, classes-1]
    unsigned int track_id;         // tracking id for video (0 - untracked, 1 - inf - tracked object)
    t_detectedColor DetectedColor[4];
    std::string color;
};

#endif //ECHOVISION_NEWCORETEST_GLOBALTYPES_H
