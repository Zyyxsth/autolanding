
#ifndef __ykusrelay_hpp
#define __ykusrelay_hpp

#include <ros/ros.h>  
#include <vector>  

namespace YKUSRelay
{

const uint8_t PACK_HEADER           = 0xA0;

const uint8_t PACK_ID               = 0x01;

const uint8_t STATE_OFF             = 0x00;
const uint8_t STATE_ON              = 0x01;

const uint8_t CMD_OFF_NO_REPLY      = 0x00;
const uint8_t CMD_ON_NO_REPLY       = 0x01;
const uint8_t CMD_OFF_REPLY         = 0x02;
const uint8_t CMD_ON_REPLY          = 0x03;
const uint8_t CMD_REVERSE_REPLY     = 0x04;
const uint8_t CMD_REQUEST           = 0x05;


typedef std::list<uint8_t> UARTPackList_t;



class RelayHandle
{
public:
    bool state;
    ros::Time last_update_time;
    UARTPackList_t buff;

    uint8_t check_sum(uint8_t pack[])
    {
        
        int sum = pack[0] + pack[1] + pack[2];
        sum = (sum % 0x100)& 0xff;
        return (uint8_t)sum;
    }

    void generate_uartpack(uint8_t pack[], int *plen, uint8_t cmd)
    {
        pack[0] = PACK_HEADER;
        pack[1] = PACK_ID;
        pack[2] = cmd;
        pack[3] = check_sum(pack);
        *plen = 4;
    }

    void pushbyte(uint8_t newbyte)
    {
        buff.push_back(newbyte);
        while(buff.size() > 4)
        {
            buff.pop_front();
        }        
    }

    bool update_state_from_buff(void)
    {
        if(buff.size() == 4)
        {
            UARTPackList_t::iterator it = buff.begin();
            uint8_t pack[4];

            for(int i = 0; i <= 3; i++)
            {
                pack[i] = *it;
                it++;
            }

            if
            (
                pack[0] == PACK_HEADER
                && pack[1] == PACK_ID
                && pack[3] == check_sum(pack)
            )
            {
                state = pack[2];
                last_update_time = ros::Time::now();
                return true;
            }
        }

        return false;
    }
};

}
 




#endif


