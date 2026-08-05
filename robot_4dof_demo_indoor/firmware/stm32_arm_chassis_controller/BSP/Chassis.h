#ifndef __CHASSIS_H
#define __CHASSIS_H

#include "stm32f10x.h"

extern uint8_t  IF ;

void Chassis_SetSpeed(uint16_t rpm);
void Chassis_SetSteer(int16_t angle);
void Chassis_Stop(void);

void Chassis_SendCmd(void);

#endif
