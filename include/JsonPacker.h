//
// Created by root on 6/4/24.
//

#ifndef ECHOVISIONPARALLELS_JSONPACKER_H
#define ECHOVISIONPARALLELS_JSONPACKER_H

#include <vector>
#include "globaltypes.h"
#include "rapidjson/rapidjson.h"
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

std::string get_TimeStamp();

class jsonobject
{
public:
    std::vector<bbox_t> objects;
    jsonobject() = default;
};

class objectPack
{
public:
    std::string bsonId="20240605120564";
    std::string Time="20240605120564";
    std::string CameraId="c1";
    std::string MovementStatus="static";

    jsonobject ai ;
    jsonobject movement ;
    objectPack() = default;
    objectPack(std::vector<bbox_t> a, std::vector<bbox_t> m,std::string camid){
        CameraId=camid;
        ai.objects = a;
        movement.objects = m;
        MovementStatus="static";
        if(m.size()>0)
        {
            MovementStatus="motion";
        }
        bsonId= "0";
        Time= get_TimeStamp();
    }

private:

};

std::string serializeJsonObject(const objectPack& obj);



namespace JSONCONVERT
{
    std::string convert_toJson(objectPack data);

}

#endif //ECHOVISIONPARALLELS_JSONPACKER_H
