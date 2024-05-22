//
// Created by root on 3/10/24.
//
#include "socketsender.h"
#include <cerrno>

class SocketException : public std::runtime_error {
public:
    SocketException(int error_code, const std::string& message)
            : std::runtime_error(message + ": " + std::strerror(error_code)) {}
};

namespace SSocket{



    struct sockaddr_in	 servaddr;

    int sockfd, portno, n;
    struct sockaddr_in serv_addr;
    struct hostent *server;
    bool is_connected;

    void init()
    {
        if ( (sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0 ) {
            perror("socket creation failed");
            exit(EXIT_FAILURE);
        }

        //////server = gethostbyname("192.168.88.190");
         server = gethostbyname("localhost");


        bzero((char *) &serv_addr, sizeof(serv_addr));
        serv_addr.sin_family = AF_INET;
        bcopy((char *)server->h_addr,
              (char *)&serv_addr.sin_addr.s_addr,
                server->h_length);
        serv_addr.sin_port = htons(PORT);

        if (connect(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
            std::cout << "ERROR connecting" << std::endl;
            is_connected=false;
        }
        else
        {
            std::cout<<"Socket Connected"<<std::endl;
            is_connected=true;
        }

    }


    int connectionAttemptCounter=0;
    int m_send(char* md, int len)
    {
        if(len < 10){return -1;}
        if(md == nullptr)
        {
            return 0;
        }

        if(!is_connected){
            connectionAttemptCounter++;
            if(connectionAttemptCounter >= 100)
            {
                connectionAttemptCounter=0;
                init();
            }
            return -1;
        }
        try {

            int _size= send(sockfd, md, len, 0);
            if (_size <= 0) {
                is_connected=false;
                connectionAttemptCounter=0;
                std::cout<<"ERROR writing to socket"<< std::endl;
            }
        }
        catch(const std::exception e)
        {
            std::cout<<"Socket Error! Connection Lost!"<<std::endl;
            is_connected=false;
            connectionAttemptCounter = 0;
        }
        catch(const SocketException& e)
        {
            is_connected=false;
            connectionAttemptCounter = 0;
        }

        return 1;
    }
}