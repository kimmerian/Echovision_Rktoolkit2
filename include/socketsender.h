//
// Created by root on 3/10/24.
//

#ifndef ECHOVISION_NEWCORETEST_SOCKETSENDER_H
#define ECHOVISION_NEWCORETEST_SOCKETSENDER_H

#include <bits/stdc++.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cstring>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>

#define PORT	 23714
#define MAXLINE 2048

namespace SSocket{



    void init();
    int m_send(char* md,int len);

}

#endif //ECHOVISION_NEWCORETEST_SOCKETSENDER_H
