#include "stm32f10x.h"
#include "MyCAN.h"
#include "chassis.h"

#define CHASSIS_CAN_ID   0x501

static uint16_t TargetRPM = 0;
static int16_t  TargetAngle = 0;
uint8_t  IF = 2;


/************************************************
设置目标速度
0~5000 RPM
************************************************/
void Chassis_SetSpeed(uint16_t rpm)
{
    TargetRPM = rpm;
}


/************************************************
设置方向
左负右正
例如:
-20 左转
+20 右转
************************************************/
void Chassis_SetSteer(int16_t angle)
{
    TargetAngle = angle;
}


/************************************************
停车
************************************************/
void Chassis_Stop(void)
{
    TargetRPM = 0;
}


/************************************************
周期发送(10ms调用一次)
************************************************/

void Chassis_SendCmd(void)
{
    uint8_t data[8] = {0};

    uint16_t steer_code;

    /*************
    Byte0
    *************/

    /*
        bit0 = 1 油门使能
        bit1 = 1 转向使能
        bit6 = 0 转速控制
        gear = D档
    */

    data[0] = 0x33;

    /*
        0x33 =
        0011 0011

        bit0 = 1
        bit1 = 1
        gear = 3 (D档)
    */


    /*************
    Byte1 Byte2
    转速
    *************/
    data[1] = (TargetRPM >> 8) & 0xFF;
    data[2] = TargetRPM & 0xFF;


    /*************
    Byte4 Byte5
    转向角
    协议要求偏移+1024
    *************/
    steer_code = TargetAngle + 1024;

    data[4] = (steer_code >> 8) & 0xFF;
    data[5] = steer_code & 0xFF;


    /*************
    发送
    *************/	
		IF = MyCAN_Transmit1(CHASSIS_CAN_ID,8,data);
}
