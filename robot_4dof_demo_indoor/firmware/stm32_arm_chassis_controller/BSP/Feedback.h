#ifndef __CHASSIS_FEEDBACK_H
#define __CHASSIS_FEEDBACK_H

#include "stm32f10x.h"

/************** 全局变量 **************/
extern uint8_t Car_Status;   // 实际车辆状态
extern uint8_t Car_Gear;     // 实际档位

/************** 函数 **************/
void Chassis_Feedback_Update(void);

#endif
