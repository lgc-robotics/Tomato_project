#ifndef __Mycan_H
#define __Mycan_H
#include "stm32f10x.h"                  // Device header
#include "cybergear1.h"

void MyCAN_Init(void);
void MyCAN_Transmit(uint32_t ID, uint8_t Length, uint8_t *Data);
uint8_t MyCAN_ReceiveFlag(void);
void MyCAN_Receive(uint32_t *ID, uint8_t *Length, uint8_t *Data);
void Cybergear_Parse(CanRxMsg *msg, MI_Motor *motor);
uint8_t MyCAN_Transmit1(uint32_t ID,uint8_t Length,uint8_t *Data);

#endif
