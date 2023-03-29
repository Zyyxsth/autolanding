
#ifndef __safeserial_hpp
#define __safeserial_hpp

#include <ros/ros.h>  
#include <serial/serial.h>

namespace SafeSerial
{

class SafeSerialHandle: public serial::Serial
{

private:
    std::string serial_port;
    int bandrate;
    bool serial_offline = false;

public:
    SafeSerialHandle(void)
    {

    }


    SafeSerialHandle(std::string set_serial_port, int set_bandrate)
    {
        setport(set_serial_port, set_bandrate);
        init();
    }

public:
    void setport(std::string set_serial_port, int set_bandrate)
    {
        serial_port = set_serial_port;
        bandrate = set_bandrate;
        setPort(serial_port); 
        setBaudrate(bandrate);         
    }

    bool init(void)
    {
        serial::Timeout to = serial::Timeout::simpleTimeout(1000); 
        setTimeout(to);

        try 
        { 
            open(); 
        } 
        catch(serial::IOException& e) 
        { 
            ROS_ERROR_STREAM("Unable to open port.");  
            return false; 
        } 

        if(isOpen())
        {
            ROS_INFO_STREAM("opened.");
        }
        else
        {
            return false;
        }
        return true;
    }


    bool check()
    {
        // ROS_ERROR_STREAM("serial port is offline.");
        if(!serial_offline)
        {
            try 
            { 
                available();
            } 
            catch(serial::IOException& e) 
            { 
                serial_offline = true;
                close();
                setPort(serial_port); 
                setBaudrate(bandrate); 
                serial::Timeout to = serial::Timeout::simpleTimeout(1000); 
                setTimeout(to);
                ROS_ERROR_STREAM("serial lost!");  
            }
        }
        if(serial_offline)
        {
            try 
            { 
                open(); 
            } 
            catch(serial::IOException& e) 
            { 
                ROS_ERROR_STREAM("Unable to reopen port. Ignore cmd this time.");  
                return false; 
            } 
            serial_offline = false;
        }
        return true;
    }
};    

}


#endif


