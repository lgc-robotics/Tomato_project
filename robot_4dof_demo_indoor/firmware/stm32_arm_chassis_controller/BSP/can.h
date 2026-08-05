#ifndef __CAN_H
#define __CAN_H

#include "board.h"
#include "fifo.h"

typedef struct
{
	__IO CanRxMsg CAN_RxMsg;
	__IO CanTxMsg CAN_TxMsg;

	__IO bool rxFrameFlag;
} CAN_t;

uint8_t can_SendCmd(__IO uint8_t *cmd, uint8_t len);
void CAN_SendFrame(uint8_t id, uint8_t data[8]);
uint8_t CAN_TransmitAtomic(CanTxMsg *tx_msg);

extern volatile uint8_t Arm_CAN_Tx_Busy;
extern volatile uint8_t Arm_CAN_Tx_Error;
extern volatile uint8_t Chassis_Keepalive_Pending;

extern volatile bool xReached;  // X轴到位全局标志
extern volatile bool yReached;  // Y轴到位全局标志
extern volatile bool zReached;  // Z轴到位全局标志

extern volatile uint32_t debugExtId;
extern volatile uint8_t debugCommType;
extern volatile uint8_t debugFlag;

extern __IO CAN_t can;

#endif
