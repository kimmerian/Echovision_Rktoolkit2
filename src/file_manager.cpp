//
// Created by root on 4/7/24.
//
#include "file_manager.h"
#include <iostream>
#include <fstream>

vector<string>  FILEMAN::readIniFile()
{
    vector<string> result ;
    fstream newfile;
    newfile.open("/home/rock/boelslib/init.txt",ios::in);
    if (newfile.is_open()){ //checking whether the file is open
        string tp;
        while(getline(newfile, tp)){ //read data from file object and put it into string.
            result.push_back(tp);
            cout << tp << "\n"; //print the data of the string
        }
        newfile.close(); //close the file object.
    }


    return result;
}


vector<string> FILEMAN::readClasses()
{
    vector<string> result ;
    fstream newfile;
    newfile.open("/home/rock/boelslib/model/classes.txt",ios::in);

    if (newfile.is_open()){ //checking whether the file is open
        string tp;
        while(getline(newfile, tp)){ //read data from file object and put it into string.
            result.push_back(tp);
            cout << tp << "\n"; //print the data of the string
        }
        newfile.close(); //close the file object.
    }
    return result;
}

