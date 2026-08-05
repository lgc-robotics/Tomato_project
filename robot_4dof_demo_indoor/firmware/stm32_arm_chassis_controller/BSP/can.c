#include "can.h"
#include "Mycan.h"
#include "cybergear1.h"
#include "stm32f10x.h"                  // Device header
#include "stm32f10x_usart.h"
#include <stdio.h>
#include "Miparam.h"
#include "board.h"
#include "Usart1.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

__IO CAN_t can = {0};

volatile bool xReached = false;
volatile bool yReached = false;
volatile bool zReached = false;

volatile uint8_t Arm_CAN_Tx_Busy = 0;
volatile uint8_t Arm_CAN_Tx_Error = 0;
volatile uint8_t Chassis_Keepalive_Pending = 0;
volatile uint32_t CAN_FIFO0_Overrun_Count = 0;

#define CAN_TX_WAIT_TIMEOUT  200000U
#define CAN_TX_RETRY_COUNT   3U

volatile uint32_t debugExtId;
volatile uint8_t debugCommType;
volatile uint8_t debugFlag = 0;
extern volatile uint8_t cyflag;
extern volatile uint16_t target_raw;
extern volatile float targetAngle;

 //第二步：修改中断处理函数
void USB_LP_CAN1_RX0_IRQHandler(void)
{
    while(CAN_MessagePending(CAN1, CAN_FIFO0) > 0)
    {
        uint8_t addr;

        CAN_Receive(CAN1, CAN_FIFO0, (CanRxMsg *)(&can.CAN_RxMsg));
        can.rxFrameFlag = true;

        if(can.CAN_RxMsg.IDE != CAN_Id_Extended)
        {
            continue;
        }

        addr = (uint8_t)(can.CAN_RxMsg.ExtId >> 8);

        // 检测到位反馈: [FD][9F][6B]
        if(can.CAN_RxMsg.DLC == 3 &&
           can.CAN_RxMsg.Data[0] == 0xFD &&
           can.CAN_RxMsg.Data[1] == 0x9F &&
           can.CAN_RxMsg.Data[2] == 0x6B)
        {
            switch(addr)
            {
                case MOTOR_ID_X: xReached = true; break;
                case MOTOR_ID_Y: yReached = true; break;
                case MOTOR_ID_Z: zReached = true; break;
                default: break;
            }
        }
    }

    if(CAN_GetFlagStatus(CAN1, CAN_FLAG_FOV0) != RESET)
    {
        CAN_FIFO0_Overrun_Count++;
        CAN_ClearFlag(CAN1, CAN_FLAG_FOV0);
    }
}

uint8_t CAN_TransmitAtomic(CanTxMsg *tx_msg)
{
    uint32_t primask = __get_PRIMASK();
    uint8_t mailbox;

    __disable_irq();
    mailbox = CAN_Transmit(CAN1, tx_msg);
    if(primask == 0U)
    {
        __enable_irq();
    }

    return mailbox;
}

static uint8_t CAN_TransmitReliable(CanTxMsg *tx_msg)
{
    uint8_t retry;

    for(retry = 0; retry < CAN_TX_RETRY_COUNT; retry++)
    {
        uint32_t timeout = CAN_TX_WAIT_TIMEOUT;
        uint8_t mailbox;
        uint8_t status;

        while((CAN1->TSR & CAN_TSR_TME) == 0U && timeout > 0U)
        {
            timeout--;
        }
        if(timeout == 0U)
        {
            continue;
        }

        mailbox = CAN_TransmitAtomic(tx_msg);
        if(mailbox == CAN_TxStatus_NoMailBox)
        {
            continue;
        }

        timeout = CAN_TX_WAIT_TIMEOUT;
        do
        {
            status = CAN_TransmitStatus(CAN1, mailbox);
            if(status == CAN_TxStatus_Ok)
            {
                return 1;
            }
            timeout--;
        }
        while(status == CAN_TxStatus_Pending && timeout > 0U);

        CAN_CancelTransmit(CAN1, mailbox);
    }

    return 0;
}

uint8_t can_SendCmd(__IO uint8_t *cmd, uint8_t len)
{
	__IO uint8_t i = 0, j = 0, k = 0, l = 0, packNum = 0;

	j = len - 2;

	while (i < j)
	{

		k = j - i;

		can.CAN_TxMsg.StdId = 0x00;
		can.CAN_TxMsg.ExtId = ((uint32_t)cmd[0] << 8) | (uint32_t)packNum;
		can.CAN_TxMsg.Data[0] = cmd[1];
		can.CAN_TxMsg.IDE = CAN_Id_Extended;
		can.CAN_TxMsg.RTR = CAN_RTR_Data;

		if (k < 8)
		{
			for (l = 0; l < k; l++, i++)
			{
				can.CAN_TxMsg.Data[l + 1] = cmd[i + 2];
			}
			can.CAN_TxMsg.DLC = k + 1;
		}

		else
		{
			for (l = 0; l < 7; l++, i++)
			{
				can.CAN_TxMsg.Data[l + 1] = cmd[i + 2];
			}
			can.CAN_TxMsg.DLC = 8;
		}

		if(!CAN_TransmitReliable((CanTxMsg *)(&can.CAN_TxMsg)))
		{
			Arm_CAN_Tx_Error = 1;
			return 0;
		}

		++packNum;
	}

	return 1;
}

void CAN_SendFrame(uint8_t id, uint8_t data[8])
{
    // 使用标准ID格式：0x140 + id
    can.CAN_TxMsg.StdId = 0x140 + id;
    can.CAN_TxMsg.IDE = CAN_Id_Standard;  // 标准帧
    can.CAN_TxMsg.RTR = CAN_RTR_Data;     // 数据帧
    can.CAN_TxMsg.DLC = 8;                // 数据长度8字节
    
    // 复制数据到发送缓冲区
    for(int i = 0; i < 8; i++) {
        can.CAN_TxMsg.Data[i] = data[i];
    }
	CAN_TransmitReliable((CanTxMsg *)(&can.CAN_TxMsg));
}
