
#ifndef __tfluna_laser_hpp
#define __tfluna_laser_hpp

#include <ros/ros.h>  
#include <vector>  

namespace TFLuna
{

const uint8_t PACK_HEADER           = 0x59;
const uint8_t PACK_LEN              = 9;


typedef std::list<uint8_t> UARTPackList_t;



class TFLunaHandle
{
public:
    ros::Time last_update_time;
    int dis_mm;
    int amp;
    int temp;

    UARTPackList_t buff;

    void pushbyte(uint8_t newbyte)
    {
        buff.push_back(newbyte);
        while(buff.size() > PACK_LEN)
        {
            buff.pop_front();
        }        
    }

    uint8_t check_sum(uint8_t pack[])
    {
        int sum = 0;

        for (int i = 0 ; i <= PACK_LEN - 2; i++)
        {
            sum += pack[i];
        }
        sum = sum & 0xff;
        return (uint8_t)sum;
    }    

    bool update_state_from_buff(void)
    {
        if(buff.size() == PACK_LEN)
        {
            UARTPackList_t::iterator it = buff.begin();
            uint8_t pack[PACK_LEN];

            for(int i = 0; i <= PACK_LEN - 1; i++)
            {
                pack[i] = *it;
                it++;
            }

            if
            (
                pack[0] == PACK_HEADER
                && pack[1] == PACK_HEADER
            )
            {
                if (pack[8] == check_sum(pack))                
                {
                    dis_mm  = pack[2] + (pack[3] * 256);
                    amp     = pack[4] + (pack[5] * 256);
                    temp    = pack[6] + (pack[7] * 256);

                    // for(int i = 0; i <= 8; i++)
                    // {
                    //     printf("%d ", pack[i]);
                    // }
                    // printf("check = %d ", check_sum(pack));

                    last_update_time = ros::Time::now();
                    return true;
                }
                // else
                // {
                //     printf("check error\r\n");
                // }
            }


        }

        return false;
    }

};

}
 




#endif


