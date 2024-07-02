#include <iostream>
#include <csignal>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "rknnPool.hpp"
#include "ThreadPool.hpp"
#include "kalman.hpp"
#include "socketsender.h"
#include <thread>
#include "future"
#include "src/cqueue.hpp"
#include "file_manager.h"
#include "MotDetect.hpp"
#include "JsonPacker.h"


void consumerBoxQ(CQueue<objectPack>& queue);
std::string conv_toJson(vector<bbox_t> val,int camId);

void pipelineRestart(std::string pipeline,int cameraId);

uint16_t n = 2,frames=0;
uint8_t fps = 1;

cv::VideoCapture rtsp1, rtsp2;

using namespace  std;

void init_pipe1(string ipstream);
void init_pipe2(string ipstream);

uint16_t pipeCounter1=0,pipeCounter2=0;
vector<string> Classes;



float scaler;
std::string movementStatus1,movementStatus2 ="static";
int movementStatusCounter1,movementStatusCounter2 = 0;


CQueue<objectPack>  c1_boxQ;
CQueue<objectPack>  c2_boxQ;

int main() {


    Classes = FILEMAN::readClasses();



    vector<string> pipes =  FILEMAN::readIniFile();

    /*  std::string stream1 ="rtspsrc location=rtsp://root:Belixys123*@172.16.103.123/axis-media/media.amp?camera=1&resolution=640x480&fps=15 ! decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1";
      std::string stream2 ="rtspsrc location=rtsp://root:Belixys123*@172.16.103.123/axis-media/media.amp?camera=2&resolution=640x480&fps=15 ! decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1";
      std::string stream3 ="rtspsrc location=rtsp://root:Belixys123*@172.16.103.95/axis-media/media.amp?resolution=640x480&fps=15 ! decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1";
  */


    string stream1 = pipes[0] ;//"rtspsrc location=rtsp://admin:Belixys123@192.168.88.178:554/cam/realmonitor?channel=1&subtype=0 ! rtph264depay ! h264parse ! tee name=t"
                     " t. ! queue ! mppvideodec format=BGR height=1080 width=1920 ! queue ! appsink sync=false drop=1 "
                     " t. ! queue ! mppvideodec format=NV12 !  mpph264enc rc-mode=vbr bps-min=2000000 bps=4000000 bps-max=5000000 gop=50 qp-max=51 qp-min=35 ! rtspclientsink location=rtsp://localhost:8554/c1"
                     " t. ! queue ! mppvideodec format=NV12  !  mpph264enc rc-mode=vbr bps-min=250000 bps=500000 bps-max=1000000 gop=50 qp-max=51 qp-min=35 ! rtspclientsink location=rtsp://localhost:8554/c1L";

    string stream2 = pipes[1];//"rtspsrc location=rtsp://admin:Belixys123@192.168.88.178:554/cam/realmonitor?channel=1&subtype=0 protocols=tcp ! rtph264depay ! h264parse ! tee name=t"
                      " t. ! queue ! mppvideodec format=BGR ! queue ! appsink sync=false drop=1 "
                      " t. ! queue ! mppvideodec format=NV12 !  mpph264enc rc-mode=vbr bps-min=2000000 bps=4000000 bps-max=5000000 gop=50 qp-max=51 qp-min=35 ! rtspclientsink location=rtsp://localhost:8554/c2";
    string model =pipes[2];//"/home/rock/YoloModels/yolov8m.rknn";

    scaler = stof(pipes[3]);

    init_pipe1(stream1);
    init_pipe2(stream2);

    char *model_name = (char *) model.c_str();

    vector<rknn_lite *> rkpool;
    dpool::ThreadPool pool(n);
    vector<std::future<vector<bbox_t>>> futs;

    rknn_lite *ptr;
    vector<string> classes = FILEMAN::readClasses();
    for (int i = 0; i < n; i++) {
        ptr = new rknn_lite(model_name, i % 3);
        rkpool.push_back(ptr);
        ptr->ClassList=classes;
        rtsp1 >> ptr->ori_img;
        futs.push_back(pool.submit(&rknn_lite::interf, &(*ptr)));
    }

    track_kalman_t c1_kalman;
    track_kalman_t c2_kalman;

    SSocket::init();

    std::thread c1_consumerThread(consumerBoxQ, std::ref(c1_boxQ));
    std::thread c2_consumerThread(consumerBoxQ, std::ref(c2_boxQ));

    motionDetector m_Detect1,m_Detect2;

    std::thread reStartCamera();

    while(true)
    {
        if (cv::waitKey(1) == 'q') {
            break;
        }
       /// usleep(210000);
        try {
            if(rtsp1.read(rkpool[0]->ori_img))
            {

                rkpool[0]->source = "2";
                 futs[0]=(pool.submit(&rknn_lite::interf, &(*rkpool[0])));
                 //usleep(200000);
                vector<bbox_t> kalmanres = futs[0].get();
                vector<bbox_t> mogResult = m_Detect1.simpleMog2(rkpool[0]->ori_img,"1");


              /* if(movementStatus1 == "movement" && mogResult.size() == 0 && ((movementStatusCounter1+=1) % 5 == 0) ? true : false)
               {
                   movementStatus1="static";
               }*/

                if (kalmanres.size() > 0) {
                    objectPack _tempob(kalmanres, mogResult, "c1");
                    c1_boxQ.push(_tempob);

                }


            }
            else
            {
                std::cout << "Pipe 1 cant read: " << std::endl;
                pipeCounter1++;

                if(pipeCounter1 >= 5)
                {
                    init_pipe1(stream1);
                }
                sleep(1);
            }
        }
        catch( cv::Exception& e )
        {
            const char* err_msg = e.what();
            std::cout << "Pipe 1 exception caught: " << err_msg << std::endl;
            pipeCounter1++;
            if(pipeCounter1 >= 5)
            {
                init_pipe1(stream1);
            }
            sleep(1);
        }



       try {
           if(rtsp2.read(rkpool[1]->ori_img))
           {

               rkpool[1]->source = "2";
               futs[1]=(pool.submit(&rknn_lite::interf, &(*rkpool[1])));
               //usleep(200000);
               vector<bbox_t> kalmanres = futs[1].get();
               vector<bbox_t> mogResult = m_Detect2.simpleMog2(rkpool[1]->ori_img,"2");


               /* if(movementStatus1 == "movement" && mogResult.size() == 0 && ((movementStatusCounter1+=1) % 5 == 0) ? true : false)
                {
                    movementStatus1="static";
                }*/

               if (kalmanres.size() > 0) {
                   objectPack _tempob(kalmanres, mogResult, "c2");
                   c2_boxQ.push(_tempob);
               }

           }
           else
           {
               std::cout << "Pipe 2 cant read: " << std::endl;
               pipeCounter1++;

               if(pipeCounter2 >= 5)
               {
                   init_pipe2(stream2);
               }
               sleep(1);
           }
        }
        catch( cv::Exception& e )
        {
            const char* err_msg = e.what();
            std::cout << "Pipe2 exception caught: " << err_msg << std::endl;
            pipeCounter2++;
            if(pipeCounter2 >= 5)
            {
                init_pipe2(stream2);
            }
            sleep(1);
        }

        frames++;
    }

    rtsp1.release();
    rtsp2.release();
    c1_boxQ.shutdown();
    c2_boxQ.shutdown();
    c1_consumerThread.join();
    c2_consumerThread.join();

    cv::destroyAllWindows();
    return 0;
}

void consumerBoxQ(CQueue<objectPack>& queue)
{
    while (true) {
        if (queue.empty() && queue.shouldTerminate())
        {
            break;
        }

        objectPack items;
        if (queue.pop(items)) {
            std::system("clear");
            string result = serializeJsonObject(items);


            string r1="["+result+"]\n";
            cout<< r1 <<endl;
            char* buf = new char[strlen(r1.c_str())+1];
            strcpy(buf, r1.c_str());


            int len = strlen(r1.c_str());
              SSocket::m_send(buf,len);

            delete [] buf;
        }
    }
}

void init_pipe1(string ipstream)
{
    rtsp1.release();
    rtsp1.open(ipstream,cv::CAP_GSTREAMER);
    if (!rtsp1.isOpened()) {
        cout << "Exception: pipeline 1 issue!" << endl;
    }
    else {
        pipeCounter1 = 0;
    }
}

void init_pipe2(string ipstream)
{
    rtsp2.release();
    rtsp2.open(ipstream,cv::CAP_GSTREAMER);
    if (!rtsp2.isOpened()) {
        cout << "Exception: pipeline 2 issue!" << endl;
    }
    else {
        pipeCounter2 = 0;
    }
}


void pipelineRestart(std::string pipeline,int cameraId,cv::VideoCapture rtsp)
{
    rtsp.release();
    bool waitlock=true;
    while(waitlock)
    {
        sleep(10);
        waitlock=false;
    }
    rtsp.open(pipeline,cv::CAP_GSTREAMER);

    if (!rtsp1.isOpened()) {
        cout << "Exception: pipeline 1 issue!" << endl;
    }
}