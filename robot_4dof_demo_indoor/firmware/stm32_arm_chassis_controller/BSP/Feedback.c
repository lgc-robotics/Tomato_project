#include "stm32f10x.h"
#include "MyCAN.h"
#include "Feedback.h"

#define CHASSIS_FEEDBACK_ID   0x502

/****************************************
全局变量
供OLED读取显示
****************************************/
uint8_t Car_Status = 4;
uint8_t Car_Gear   = 0;


/****************************************
函数:解析底盘反馈
功能:检测CAN是否收到0x502
****************************************/
void Chassis_Feedback_Update(void)
{
    uint32_t ID;
    uint8_t Length;
    uint8_t Data[8];

    /********************
    是否收到CAN消息
    ********************/
    if(MyCAN_ReceiveFlag())
    {
        MyCAN_Receive(&ID,&Length,Data);

        /********************
        是否是底盘反馈帧0x502
        ********************/
        if(ID == CHASSIS_FEEDBACK_ID)
        {
            uint8_t byte0 = Data[0];

            /*********************************
            bit0~bit1
            实际车辆状态
            *********************************/
            Car_Status = byte0 & 0x03; //0x03=00000011 保留最低两位

            /*
            0 人工
            1 遥控
            2 自动驾驶
            3 紧急刹车
            */


            /*********************************
            bit2~bit3
            实际档位
            *********************************/
            Car_Gear = (byte0 >> 2) & 0x03;//先右移两位，再保留最低两位

            /*
            1 R
            2 N
            3 D
            */
        }
    }
}
