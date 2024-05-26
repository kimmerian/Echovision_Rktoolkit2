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


void consumerBoxQ(CQueue<vector<bbox_t>>& queue,int camId);
std::string conv_toJson(vector<bbox_t> val,int camId);

uint16_t n = 1,frames=0;
uint8_t fps = 1;

cv::VideoCapture rtsp1, rtsp2;

using namespace  std;

void init_pipe1(string ipstream);
void init_pipe2(string ipstream);

uint16_t pipeCounter1=0,pipeCounter2=0;
vector<string> Classes;



float scaler;
std::string movementStatus="static";

int main() {
    CQueue<vector<bbox_t>> c1_boxQ;
   // CQueue<vector<bbox_t>> c2_boxQ;

    Classes = FILEMAN::readClasses();

    vector<rknn_lite *> rkpool;
    dpool::ThreadPool pool(n);
    queue<std::future<vector<bbox_t>>> futs;

    vector<string> pipes =  FILEMAN::readIniFile();

    /*  std::string stream1 ="rtspsrc location=rtsp://root:Belixys123*@172.16.103.123/axis-media/media.amp?camera=1&resolution=640x480&fps=15 ! decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1";
      std::string stream2 ="rtspsrc location=rtsp://root:Belixys123*@172.16.103.123/axis-media/media.amp?camera=2&resolution=640x480&fps=15 ! decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1";
      std::string stream3 ="rtspsrc location=rtsp://root:Belixys123*@172.16.103.95/axis-media/media.amp?resolution=640x480&fps=15 ! decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1";
  */


    string stream1 = pipes[0];//"rtspsrc location=rtsp://root:Belixys123*@192.168.88.179/axis-media/media.amp?resolution=640x480&fps=10 ! watchdog timeout=1000  ! decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1";
   // string stream2 = pipes[1];//"rtspsrc location=rtsp://admin:Belixys123*@192.168.88.178:554//cam/realmonitor?channel=1&subtype=2 ! watchdog timeout=1000  !  decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1";
    string model ="/home/rock/YoloModels/yolov8m.rknn";

    usleep(100000);
   // init_pipe2(stream2);
  //  usleep(100000);
   std::string hede ="v4l2src device=/dev/video0 ! tee name=t"
                     " t. ! queue ! mppjpegdec format=NV12 ! mpph264enc ! rtspclientsink location=rtsp://localhost:8554/c1"
                     " t. ! queue ! mppjpegdec format=BGR ! queue ! appsink sync=false drop=1 ";

   std::string bede="rtspsrc location=rtsp://root:pass@192.168.88.179/axis-media/media.amp?camera=1&resolution=1920x1080&fps=25 protocols=tcp !  rtph264depay ! h264parse ! tee name=d"
                    " d. ! queue ! mppvideodec format=NV12 !  mpph264enc rc-mode=vbr bps-min=2000000 bps=4000000 bps-max=5000000 gop=50 qp-max=51 qp-min=35 ! rtspclientsink location=rtsp://localhost:8554/c1"
                    " d. ! queue ! mppvideodec format=BGR ! appsink sync=false drop=1";
  scaler = stof(pipes[3]);

    std::string demo ="filesrc location=/mnt/ssd/videos/nvr/2024/05/21/c1/2024.05.21T16:39:49.167753.mp4 ! qtdemux ! h264parse !  mppvideodec format=BGR framerate=25/1  ! appsink sync=false ";
    std::string demo1 ="filesrc location=/home/rock/Videos/boels.mp4  ! qtdemux  ! h264parse  ! appsink ";



    init_pipe1(hede);

    char *model_name = (char *) model.c_str();

    rknn_lite *ptr;
    for (int i = 0; i < n; i++) {
        ptr = new rknn_lite(model_name, i % 3);
        rkpool.push_back(ptr);
        rtsp1 >> ptr->ori_img;
        futs.push(pool.submit(&rknn_lite::interf, &(*ptr)));
    }

    track_kalman_t c1_kalman(64, 2,20, cv::Size(1920, 1080));
   // track_kalman_t c2_kalman(64, 10,100, cv::Size(1920, 1080));

    SSocket::init();

    std::thread c1_consumerThread(consumerBoxQ, std::ref(c1_boxQ),1);
 //   std::thread c2_consumerThread(consumerBoxQ, std::ref(c2_boxQ),2);



   motionDetector m_Detect;
    //cv::Mat frame;
    while(true)
    {
        if (cv::waitKey(1) == 'q') {
            break;
        }



        try {
            if(rtsp1.read(rkpool[0]->ori_img))
            {

               /* if (futs.front().get() == NULL) {
                    break;
                }*/

                futs.pop();
                rkpool[0]->source = "1";
                 futs.push(pool.submit(&rknn_lite::interf, &(*rkpool[0])));
                vector<bbox_t> proccResult =   futs.front().get();

                vector<bbox_t> c1_kalmanresult = c1_kalman.correct(&proccResult);
                vector<bbox_t> tempc1result;

                for (auto item: c1_kalmanresult) {
                    item.frames_counter = frames % fps;
                    tempc1result.push_back(item);

                    cv::rectangle(rkpool[0]->ori_img, cv::Point(item.x, item.y),
                                  cv::Point(item.x + item.w,
                                            item.y + item.h),
                                  cv::Scalar(0, 0, 250), 1);
                    cv::putText(rkpool[0]->ori_img, to_string(item.obj_id), cv::Point(item.x, item.y + 10), cv::FONT_HERSHEY_SIMPLEX, 1,
                                cv::Scalar(150, 200, 0));
                    cv::putText(rkpool[0]->ori_img, to_string(item.track_id), cv::Point(item.x+50, item.y + 10), cv::FONT_HERSHEY_SIMPLEX, 1,
                                cv::Scalar(150, 250, 0));
                }

                vector<bbox_t> motVector = m_Detect.detectRectanglesInsideObjects(rkpool[0]->ori_img,c1_kalmanresult);


                movementStatus="static";
                if(motVector.size()>0) {
                    movementStatus="movement";
                    for (auto item: motVector) {
                        tempc1result.push_back(item);
                        cv::rectangle(rkpool[0]->ori_img, cv::Point(item.x+50, item.y+40),
                                      cv::Point(item.x + item.w,
                                                item.y + item.h),
                                      cv::Scalar(0, 250, 250), 3);
                    }
                }

                c1_boxQ.push(tempc1result);

                /*if(c1_kalmanresult.size()>0) {
                    for (int t = 0; t < c1_kalmanresult.size() - 1; t++) {

                        bbox_t res = c1_kalmanresult[t];

                        cv::rectangle(rkpool[0]->ori_img, cv::Point(res.x, res.y),
                                      cv::Point(res.x + res.w,
                                                res.y + res.h),
                                      cv::Scalar(0, 0, 250), 1);

                        putText(rkpool[0]->ori_img, std::to_string(res.obj_id), cv::Point(res.x, res.y + 12),
                                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));

                    }
                }*/

                //And we copy the data from frame to old_frame




                cv::imshow("fr",rkpool[0]->ori_img);
            }
            else
            {
                std::cout << "Pipe 1 cant read: " << std::endl;
                pipeCounter1++;

                if(pipeCounter1 >= 10)
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
            if(pipeCounter1 >= 100)
            {
                init_pipe1(stream1);
            }
        }



       /* try {
            if (rtsp2.read(rkpool[1]->ori_img))
            {

                if (futs.front().get() != 0) {
                    break;
                }

                futs.pop();
                rkpool[1]->source = "2";
                futs.push(pool.submit(&rknn_lite::interf, &(*rkpool[1])));

                vector<bbox_t> c2_kalmanresult = c2_kalman.correct(&rkpool[1]->p);
                vector<bbox_t> tempc2result;

                for (auto item: c2_kalmanresult) {
                    item.frames_counter = frames % fps;
                    tempc2result.push_back(item);
                }

                c2_boxQ.push(tempc2result);

                 for (int t = 0; t < rkpool[1]->p.size(); t++) {
                     cv::rectangle(rkpool[1]->ori_img, cv::Point(rkpool[1]->p.data()[t].x, rkpool[1]->p.data()[t].y),
                                   cv::Point(rkpool[1]->p.data()[t].x + rkpool[1]->p.data()[t].w,
                                             rkpool[1]->p.data()[t].y + rkpool[1]->p.data()[t].h),
                                   cv::Scalar(0, 0, 250), 1);
                 }

                // cv::imshow("fr",rkpool[1]->ori_img);
            }
        }
        catch( cv::Exception& e )
        {
            const char* err_msg = e.what();
            std::cout << "Pipe2 exception caught: " << err_msg << std::endl;
            pipeCounter2++;
            if(pipeCounter2 >= 100)
            {
                init_pipe2(stream2);
            }
        }*/

        frames++;
    }

    rtsp1.release();
   // rtsp2.release();
    c1_consumerThread.join();
    //c2_consumerThread.join();

    cv::destroyAllWindows();
    return 0;
}


int dupcounter= 10;
std::string conv_toJson(vector<bbox_t> val,int camId)
{
    if(val.size() == 0){return "0" ;}

    time_t now = time(0);

    std::tm* utc_tm = std::localtime(&now);

    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count() % 1000;

    string mon,day,hour,min,sec;

    mon = utc_tm->tm_mon+1;
    day=to_string(utc_tm->tm_mday);
    hour=to_string(utc_tm->tm_hour);
    min=to_string(utc_tm->tm_min);
    sec=to_string(utc_tm->tm_sec);


    if(utc_tm->tm_mon <10)
    {
        mon="0"+to_string(utc_tm->tm_mon+1);
    }
    if(utc_tm->tm_mday <10)
    {
        day="0"+to_string(utc_tm->tm_mday);
    }
    if(utc_tm->tm_hour <10)
    {
        hour="0"+to_string(utc_tm->tm_hour);
    }
    if(utc_tm->tm_min <10)
    {
        min="0"+to_string(utc_tm->tm_min);
    }
    if(utc_tm->tm_sec < 10)
    {
        sec="0"+to_string(utc_tm->tm_sec);
    }


    string fcounter = to_string(val[0].frames_counter);
    if(val[0].frames_counter < 10)
    {
        fcounter = "0"+ to_string(val[0].frames_counter);
    }
    string CamID = "c"+to_string(camId);

    std::string bsonId = to_string(utc_tm->tm_year+1900) +"."+ mon +"."+day+"."+hour+"."+min+"."+sec+ "."+ to_string(millis)+"."+ CamID+"."+ fcounter;
    std::string t = to_string(utc_tm->tm_year+1900) +"-"+ mon +"-"+day+"T"+hour+":"+min+":"+sec+".000Z";
    //std::string t = "2024-03-12T12:26:15.000Z";
    dupcounter++;
    std::string msg ;


         msg +=
                "[{\"Id\":\"" + bsonId + "\"" + "," + "\"Time\":\"" + t + "\",\"CameraId\":\"" + CamID + "\"," +
                "\"MovementStatus\":\"" + movementStatus + "\"," + "\"Objects\":[";



    int z =0;
    for(int i = 0;i< val.size();i++)
    {
        if(Classes[val[i].obj_id] == Classes[79])
        {
            msg.pop_back();
            break;
        }

        val[i].color ="a";
        std::string empty = "{\"CId\":""\""+ Classes[val[i].obj_id]+"\"" +
                            ",\"TId\":"+ to_string(val[i].track_id)+",\"Prob\":"+ to_string((uint16_t)val[i].prob*10)+",\"Clr\":\""+ val[i].color+"\",\"X\":"+ to_string(int (val[i].x*scaler))+",\"Y\":"+ to_string(int(val[i].y*scaler))+",\"W\":"+ to_string(int(val[i].w*scaler))+",\"H\":"+ to_string(int(val[i].h*scaler))+
                            "}";
        msg+=empty;
        if(i<val.size()-1)
        {
            msg+=",";
        }
       z++;
    }
    if(val.size() == 0)
    {
        msg+="[";
    }
    if(movementStatus=="static") {
        msg += "]}]\n";
    }
    else {
        msg += "]},{\"Id\":\"" + bsonId + "\"" + "," + "\"Time\":\"" + t + "\",\"CameraId\":\"" + "motionDetectFrame" +
               "\"," + "\"MovementStatus\":\"" + movementStatus + "\"," + "\"Objects\":[";

        for (int i = z-1; i < val.size(); i++) {
            val[i].color = "a";
            std::string empty = "{\"CId\":""\"" + Classes[val[i].obj_id] + "\"" +
                                ",\"TId\":" + to_string(val[i].track_id) + ",\"Prob\":" +
                                to_string((uint16_t) val[i].prob * 10) + ",\"Clr\":\"" + val[i].color + "\",\"X\":" +
                                to_string(int(val[i].x * scaler)) + ",\"Y\":" + to_string(int(val[i].y * scaler)) +
                                ",\"W\":" + to_string(int(val[i].w * scaler)) + ",\"H\":" +
                                to_string(int(val[i].h * scaler)) +
                                "}";
            msg += empty;
            if (i < val.size() - 1) {
                msg += ",";
            }
        }
        msg+="]}]\n";
    }



    return msg;
}


void consumerBoxQ(CQueue<vector<bbox_t>>& queue,int camId)
{
    while (true) {
        if (queue.empty() && queue.shouldTerminate())
        {
            break;
        }

        vector<bbox_t> items;
        if (queue.pop(items)) {
            std::system("clear");
            string result = conv_toJson(items,camId);
            cout<< result <<endl;

            char* buf = new char[strlen(result.c_str())+1];
            strcpy(buf, result.c_str());


            int len = strlen(result.c_str());
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
    pipeCounter1=0;
}

void init_pipe2(string ipstream)
{
    rtsp2.release();
    rtsp2.open(ipstream,cv::CAP_GSTREAMER);
    if (!rtsp2.isOpened()) {
        cout << "Exception: pipeline 2 issue!" << endl;
    }
    pipeCounter2=0;
}
