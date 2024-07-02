//
// Created by root on 6/4/24.
//
#include "JsonPacker.h"




std::string serializeJsonObject(const objectPack& obj){

    rapidjson::Document doc;
    doc.SetObject();
    rapidjson::Document::AllocatorType& allocator = doc.GetAllocator();

    doc.AddMember("Id", rapidjson::Value().SetString(obj.bsonId.c_str(), allocator), allocator);
    doc.AddMember("Time", rapidjson::Value().SetString(obj.Time.c_str(), allocator), allocator);
    doc.AddMember("CameraId", rapidjson::Value().SetString(obj.CameraId.c_str(), allocator), allocator);
    doc.AddMember("MovementStatus", rapidjson::Value().SetString(obj.MovementStatus.c_str(), allocator), allocator);

    rapidjson::Value objects(rapidjson::kArrayType);
    for (const auto& bbox : obj.ai.objects) {
        rapidjson::Value bboxObj(rapidjson::kObjectType);
        bboxObj.AddMember("X", bbox.x, allocator);
        bboxObj.AddMember("Y", bbox.y, allocator);
        bboxObj.AddMember("W", bbox.w, allocator);
        bboxObj.AddMember("H", bbox.h, allocator);
        bboxObj.AddMember("CId", rapidjson::Value().SetString(bbox.obj_id.c_str(), allocator),allocator);
        bboxObj.AddMember("TId", bbox.track_id, allocator);
        bboxObj.AddMember("Prob", bbox.prob, allocator);
        bboxObj.AddMember("Clr", rapidjson::Value().SetString(bbox.color.c_str(), allocator),allocator);

        rapidjson::Value ClrA(rapidjson::kArrayType);
        for(const auto& color : bbox.DetectedColor){
          /*  rapidjson::Value t_color(rapidjson::kObjectType);

            t_color.AddMember("Red",color.r,allocator);
            t_color.AddMember("Green",color.g,allocator);
            t_color.AddMember("Blue",color.b,allocator);*/
         /*   rapidjson::Value t_color(rapidjson::kArrayType);
            t_color.PushBack(color.r,allocator);
            t_color.PushBack(color.g,allocator);
            t_color.PushBack(color.b,allocator);*/

         std::string col = std::to_string(color.r)+","+std::to_string(color.g)+","+std::to_string(color.b);


            ClrA.PushBack(rapidjson::Value().SetString(col.c_str(),allocator),allocator);

        }
        bboxObj.AddMember("ClrA",ClrA,allocator);

        objects.PushBack(bboxObj, allocator);
    }
    doc.AddMember("Objects", objects, allocator);

    rapidjson::Value Motion(rapidjson::kArrayType);
    for (const auto &bbox: obj.movement.objects) {
        rapidjson::Value bboxObj(rapidjson::kObjectType);
        bboxObj.AddMember("X", bbox.x, allocator);
        bboxObj.AddMember("Y", bbox.y, allocator);
        bboxObj.AddMember("W", bbox.w, allocator);
        bboxObj.AddMember("H", bbox.h, allocator);
        Motion.PushBack(bboxObj, allocator);
    }
    doc.AddMember("Motion", Motion, allocator);

    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    doc.Accept(writer);

    return buffer.GetString();
}

using namespace std;
std::string get_TimeStamp() {

    time_t now = time(0);

    std::tm* utc_tm = std::localtime(&now);

    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count() % 100;

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

    std::string t = to_string(utc_tm->tm_year+1900) +"-"+ mon +"-"+day+"T"+hour+":"+min+":"+sec+"."+to_string(millis);
    //std::string t = "2024-03-12T12:26:15.00";

    return t;
}