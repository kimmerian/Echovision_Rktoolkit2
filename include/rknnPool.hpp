#ifndef _rknnPool_H
#define _rknnPool_H

#include <queue>
#include <vector>
#include <atomic>         // std::atomic, std::atomic_flag, ATOMIC_FLAG_INIT
#include <iostream>
#include "rga.h"
#include "im2d.h"
#include "RgaUtils.h"
#include "rknn_api.h"
#include "postprocess.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "ThreadPool.hpp"
#include "list"
#include "globaltypes.h"
#include "rk_common.h"
#include "kalman.hpp"


#include <math.h>

using cv::Mat;
using std::queue;
using std::vector;

float dispcoef = 0.0015625f;

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz);
static unsigned char *load_model(const char *filename, int *model_size);

class rknn_lite
{
private:
    rknn_context ctx;
    unsigned char *model_data;
    rknn_sdk_version version;
    rknn_input_output_num io_num;
    rknn_tensor_attr *input_attrs;
    rknn_tensor_attr *output_attrs;
    rknn_input inputs[1];
    int ret;
    int channel = 3;
    int previousDetectionCount;
    rknn_app_context_t rknn_app_ctx;
    track_kalman_t c1_kalman;


    int nextId = 0;
    const int maxFramesWithoutUpdate = 10;

public:
  //  sortproc sorting;
    std::vector<bbox_t> p;
    Mat ori_img;
    vector<bbox_t>  interf();
    rknn_lite(char *dst, int n);
    ~rknn_lite();
    std::string source;
    vector<std::string> ClassList;
};

rknn_lite::rknn_lite(char *model_path, int n)
{
    /* Create the neural network */
    printf("Loading mode...\n");
    int model_data_size = 0;
    // 读取模型文件数据
    model_data = load_model(model_path, &model_data_size);
    // 通过模型文件初始化rknn类
    ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        exit(-1);
    }
    // 
    rknn_core_mask core_mask;
    if (n == 0)
        core_mask = RKNN_NPU_CORE_0;
    else if(n == 1)
        core_mask = RKNN_NPU_CORE_1;
    else
        core_mask = RKNN_NPU_CORE_2;
    int ret = rknn_set_core_mask(ctx, core_mask);
    if (ret < 0)
    {
        printf("rknn_init core error ret=%d\n", ret);
        exit(-1);
    }
    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0) {
        printf("rknn_init error ret=%d\n", ret);

    }
    printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

    // Get Model Input Output Number
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query fail! ret=%d\n", ret);

    }
    printf("\nmodel input num: %d\n", io_num.n_input);
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for(uint32_t i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret                  = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
            printf("rknn_init error ret=%d\n", ret);

        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    printf("\nmodel output num: %d\n", io_num.n_output);
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for(uint32_t i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret                   = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        dump_tensor_attr(&(output_attrs[i]));
    }

    // Set to context
    rknn_app_ctx.rknn_ctx = ctx;

    // TODO
    if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && output_attrs[0].type != RKNN_TENSOR_FLOAT16)
    {
        rknn_app_ctx.is_quant = true;
    }
    else
    {
        rknn_app_ctx.is_quant = false;
    }

    rknn_app_ctx.io_num = io_num;
    rknn_app_ctx.input_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(rknn_app_ctx.input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    rknn_app_ctx.output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(rknn_app_ctx.output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW){
        printf("model input is NCHW\n");
        rknn_app_ctx.model_channel = input_attrs[0].dims[1];
        rknn_app_ctx.model_height = input_attrs[0].dims[2];
        rknn_app_ctx.model_width = input_attrs[0].dims[3];
    }
    else{
        printf("model input is NHWC\n");
        rknn_app_ctx.model_height = input_attrs[0].dims[1];
        rknn_app_ctx.model_width = input_attrs[0].dims[2];
        rknn_app_ctx.model_channel = input_attrs[0].dims[3];
    }
    printf("model input height=%d, width=%d, channel=%d\n",
           rknn_app_ctx.model_height, rknn_app_ctx.model_width, rknn_app_ctx.model_channel);

    if(ret != 0){
        printf("init_yolox_model fail! ret=%d model_path=%s\n", ret, model_path);

    }
    //track_kalman_t c1_kalman = track_kalman_t(64, 3,40, cv::Size(1920, 1080));
}

vector<bbox_t> rknn_lite::interf(){

    cv::Mat img;

    int img_width = ori_img.cols;
    int img_height = ori_img.rows;

    cv::cvtColor(ori_img, img, cv::COLOR_BGR2RGB);

    // init rga context

    rga_buffer_t src;
    rga_buffer_t dst;
    memset(&src, 0, sizeof(src));
    memset(&dst, 0, sizeof(dst));
    im_rect src_rect;
    im_rect dst_rect;
    memset(&src_rect, 0, sizeof(src_rect));
    memset(&dst_rect, 0, sizeof(dst_rect));

    int width   = rknn_app_ctx.model_width;
    int height  = rknn_app_ctx.model_height;
    rknn_input inputs[rknn_app_ctx.io_num.n_input];
    rknn_output outputs[rknn_app_ctx.io_num.n_output];

    // You may not need resize when src resulotion equals to dst resulotion
    void *resize_buf = nullptr;

    if (img_width !=  width || img_height !=  height)
    {
        resize_buf = malloc( height *  width *  channel);
        memset(resize_buf, 0x00,  height *  width *  channel);

        src = wrapbuffer_virtualaddr((void *)img.data, img_width, img_height, RK_FORMAT_RGB_888);
        dst = wrapbuffer_virtualaddr((void *)resize_buf,  width,  height, RK_FORMAT_RGB_888);
        ret = imcheck(src, dst, src_rect, dst_rect);
        if (IM_STATUS_NOERROR !=  ret)
        {
            printf("%d, check error! %s", __LINE__, imStrError((IM_STATUS) ret));
            exit(-1);
        }
        IM_STATUS STATUS = imresize(src, dst);

        cv::Mat resize_img(cv::Size( width,  height), CV_8UC3, resize_buf);
        inputs[0].buf = resize_buf;
    }
    else
    {
        inputs[0].buf = (void *)img.data;
    }

    object_detect_result_list od_results;

    // Set Input Data
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = rknn_app_ctx.model_width * rknn_app_ctx.model_height * rknn_app_ctx.model_channel;
    inputs[0].pass_through = 0;

    // allocate inputs
    ret = rknn_inputs_set(rknn_app_ctx.rknn_ctx, rknn_app_ctx.io_num.n_input, inputs);
    if (ret < 0) {
        printf("rknn_input_set fail! ret=%d\n", ret);
    }
    // allocate outputs
    memset(outputs, 0, sizeof(outputs));
    for (uint32_t i = 0; i < rknn_app_ctx.io_num.n_output; i++) {
        outputs[i].index = i;
        outputs[i].want_float = (!rknn_app_ctx.is_quant);
    }
    // run
    rknn_run(rknn_app_ctx.rknn_ctx, nullptr);
    rknn_outputs_get(rknn_app_ctx.rknn_ctx, rknn_app_ctx.io_num.n_output, outputs, NULL);

    // post process
    float scale_w = (float) width / img_width;
    float scale_h = (float) height / img_height;

    post_process(&rknn_app_ctx, outputs, BOX_THRESH, NMS_THRESH, scale_w, scale_h, &od_results);

    vector<bbox_t> od_bbox_vect;
    for(int t=0;t<od_results.count;t++)
    {
        object_detect_result det_obj = od_results.results[t];
        if ((det_obj.box.right-det_obj.box.left) *(det_obj.box.bottom-det_obj.box.top) > 256 && det_obj.box.bottom-det_obj.box.top > height*0.048 ) {

            t_detectedColor centerColor,topColor1,topColor2,bottomColor;

            int cx = det_obj.box.left+((det_obj.box.right-det_obj.box.left)/2);
            int cy =det_obj.box.top + ((det_obj.box.bottom-det_obj.box.top)/2);
            int h = det_obj.box.bottom-det_obj.box.top;

            cv::Vec3b color = ori_img.at<cv::Vec3b>(cx,det_obj.box.top+(h/4));
            topColor1.r = color[2];topColor1.g=color[1];topColor1.b = color[0];

            color = ori_img.at<cv::Vec3b>(cx,det_obj.box.top+(h/8)*3);
            topColor2.r = color[2];topColor2.g=color[1];topColor2.b = color[0];

            color = ori_img.at<cv::Vec3b>(cx,det_obj.box.top+(h/8)*4);
            centerColor.r = color[2];centerColor.g=color[1];centerColor.b = color[0];

            color = ori_img.at<cv::Vec3b>(cx,det_obj.box.top+(h/8)*6);
            bottomColor.r = color[2];centerColor.g=color[1];centerColor.b = color[0];


            bbox_t v;
            v.x = ((float)det_obj.box.left *dispcoef);
            v.y =((float)det_obj.box.top*dispcoef);
            v.w = ((float)det_obj.box.right*dispcoef)-((float)det_obj.box.left*dispcoef);
            v.h = ((float)det_obj.box.bottom*dispcoef)-((float)det_obj.box.top*dispcoef);
            v.obj_id = ClassList[det_obj.cls_id];
            v.track_id = det_obj.track_id;
            v.prob = det_obj.prop;
            v.color= "["+ std::to_string(color[0])+","+std::to_string(color[1])+","+std::to_string(color[2])+"]";
            v.DetectedColor[0]=topColor1;
            v.DetectedColor[1]=topColor2;
            v.DetectedColor[2]=centerColor;
            v.DetectedColor[3]=bottomColor;

            od_bbox_vect.push_back(v);
        }
    }

    vector<bbox_t> kalmanResultVector;

       kalmanResultVector = c1_kalman.correct(&od_bbox_vect);

    previousDetectionCount = od_bbox_vect.size();

    rknn_outputs_release(ctx, rknn_app_ctx.io_num.n_output, outputs);
       if (resize_buf) {
             free(resize_buf);
        }

       return kalmanResultVector;
}

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp)
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL)
    {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp)
    {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}

#endif