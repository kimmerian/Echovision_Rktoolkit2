#ifndef _rknnPool_H
#define _rknnPool_H

#include <queue>
#include <vector>
#include <atomic>         // std::atomic, std::atomic_flag, ATOMIC_FLAG_INIT
#include <iostream>
#include "rga.h"
//#include "sort.h"
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

#include <math.h>

using cv::Mat;
using std::queue;
using std::vector;


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
    int width = 0;
    int height = 0;
    rknn_app_context_t rknn_app_ctx;
public:
  //  sortproc sorting;
    std::vector<bbox_t> p;
    Mat ori_img;
    int interf();
    rknn_lite(char *dst, int n);
    ~rknn_lite();
    std::string source;
};

rknn_lite::rknn_lite(char *model_name, int n)
{
    /* Create the neural network */
    printf("Loading mode...\n");
    int model_data_size = 0;
    // 读取模型文件数据
    model_data = load_model(model_name, &model_data_size);
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

    // 初始化rknn类的版本
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        exit(-1);
    }

    // 获取模型的输入参数
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        exit(-1);
    }

    // 设置输入数组
    input_attrs = new rknn_tensor_attr[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_init error ret=%d\n", ret);
            exit(-1);
        }
    }

    // 设置输出数组
    output_attrs = new rknn_tensor_attr[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs) );
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
    }

    // 设置输入参数
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("model is NCHW input fmt\n");
        channel = input_attrs[0].dims[1];
        height = input_attrs[0].dims[2];
        width = input_attrs[0].dims[3];
    }
    else
    {
        printf("model is NHWC input fmt\n");
        height = input_attrs[0].dims[1];
        width = input_attrs[0].dims[2];
        channel = input_attrs[0].dims[3];
    }

    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = width * height * channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;
}

rknn_lite::~rknn_lite()
{
    ret = rknn_destroy(ctx);
    delete[] input_attrs;
    delete[] output_attrs;
    if (model_data)
        free(model_data);
}

int rknn_lite::interf()
{
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
        inputs[0].buf = (void *)img.data;

    rknn_inputs_set(ctx, io_num.n_input, inputs);

    //std::cout<<"1"<<std::endl;

    rknn_output outputs[ io_num.n_output]; // 0x49 0x4C 0x47 0x49 0x4D
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i <  io_num.n_output; i++)
        outputs[i].want_float = 0;
     ret = rknn_run(ctx, NULL);
     ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
    // std::cout<<"2"<<std::endl;
    // post process
    const float nms_threshold = NMS_THRESH;
    const float box_conf_threshold = BOX_THRESH;
    float scale_w = (float) width / img_width;
    float scale_h = (float) height / img_height;

    object_detect_result_list detect_result_group;
    std::vector<float> out_scales;
    std::vector<int32_t> out_zps;

    for (int i = 0; i <  io_num.n_output; ++i)
    {
        out_scales.push_back( output_attrs[i].scale);
        out_zps.push_back( output_attrs[i].zp);
    }

    rknn_app_ctx.rknn_ctx=ctx;
    rknn_app_ctx.model_channel=channel;
    rknn_app_ctx.model_width=width;
    rknn_app_ctx.model_height=height;

    rknn_app_ctx.io_num =io_num;
    rknn_app_ctx.input_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(rknn_app_ctx.input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    rknn_app_ctx.output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(rknn_app_ctx.output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && output_attrs[0].type != RKNN_TENSOR_FLOAT16)
    {
        rknn_app_ctx.is_quant = true;
    }
    else
    {
        rknn_app_ctx.is_quant = false;
    }

    post_process(&rknn_app_ctx, outputs, box_conf_threshold, nms_threshold, scale_w, scale_h, &detect_result_group);



    // Draw Objects
    // std::cout<<"3"<<std::endl;
     std::vector<bbox_t> result_vec;
    for (int i = 0; i < detect_result_group.count; i++)
    {
        object_detect_result *det_result = &(detect_result_group.results[i]);

        bbox_t f;
        f.x =  det_result->box.left;
        f.y = det_result->box.top;
        f.h = det_result->box.bottom - det_result->box.top;
        f.w = det_result->box.right - det_result->box.left;
        f.obj_id = det_result->cls_id;
        f.prob = det_result->prop/10.0f;
        result_vec.push_back(f);
    }

    p =  result_vec;

    char text[256];

    //std::cout<<"5"<<std::endl;
    ret = rknn_outputs_release(ctx, io_num.n_output, outputs);
    if (resize_buf)
    {
        free(resize_buf);
    }
   // std::cout<<"6"<<std::endl;
    return 0;
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