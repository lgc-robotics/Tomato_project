#include "stm32f10x.h"                  // Device header
#include "cybergear1.h"
#include "Miparam.h"
#include <string.h>
#include <stdio.h>
#include "Usart1.h"
#include "can.h"
/*———————————————————————————————————————————————————
cMyCAN_Init   CAN 初始化函数
Param：No
Init Pin：GPIOA P11   CAN_RX，输入引脚是A11
          GPIOA P12   CAN_TX，输出引脚是A12         */

void Cybergear_Parse(CanRxMsg *msg, MI_Motor *motor)
{
    // 电机ID
    motor->CAN_ID = (msg->ExtId & 0xFFFF) >> 8;

    // ? 保存原始值（16位）
    raw_angle = (msg->Data[0] << 8) | msg->Data[1];

    // ? 如果还需要浮点角度，可以继续转
    motor->Angle = uint16_to_float(
        raw_angle,
        P_MIN, P_MAX, 16);

    motor->Speed = uint16_to_float(
        (msg->Data[2] << 8) | msg->Data[3],
        V_MIN, V_MAX, 16);

    motor->Torque = uint16_to_float(
        (msg->Data[4] << 8) | msg->Data[5],
        T_MIN, T_MAX, 16);
}
/***********************************************************
Name：MyCAN_Transmit CAN 发送报文
Param： ID       ID是32位，便于后面使用扩展帧
        Length   数据长度
        *Data    数据指针                                     
		******************************************************/		
void MyCAN_Transmit(uint32_t ID, uint8_t Length, uint8_t *Data)
{
	CanTxMsg TxMessage; //定义CanTxMsg结构体变量，表示待发送的报文
	TxMessage.StdId = ID; //标准ID
	TxMessage.ExtId = ID; //扩展ID
	TxMessage.IDE = CAN_Id_Standard; //扩展标志位，CAN_Id_Standard 标准ID ，CAN_Id_Extended扩展ID
	TxMessage.IDE = CAN_Id_Extended; //扩展标志位，CAN_Id_Standard 标准ID ，CAN_Id_Extended扩展ID
	TxMessage.RTR = CAN_RTR_Data; //遥控标志位，CAN_RTR_Remote 遥控帧，	CAN_RTR_Data数据帧
	TxMessage.DLC = Length; //数据段长度，传入的参数
	//把形参DATA传过来的数组赋值给TxMessage.Data
	for (uint8_t i = 0; i < Length; i ++)
	{
		TxMessage.Data[i] = Data[i];//将传入的Data数组的值赋值给结构体的Data，他们都是8字节的数组
	}
	//请求发送报文函数
	//CAN_Transmit的原理：选择空发送邮箱——如果邮箱有空位，则将报文写入指定寄存器——TXRQ置1，请求发送
	uint8_t TransmitMailbox = CAN_TransmitAtomic(&TxMessage);// 原子占用邮箱，避免与TIM4并发发送
	if(TransmitMailbox == CAN_TxStatus_NoMailBox)
	{
		return;
	}
	uint32_t Timeout = 0;
//	USART_SendByte(0xC1);
	//CAN_TransmitStatus表示返回传输状态函数，返回请求发送邮箱的邮箱状态，CAN_TxStatus_OK表示发送成功
	//等待函数返回OK，当CAN1的邮箱状态为CAN_TxStatus_Ok表示发送成功，如果不成功则进入循环
	while (CAN_TransmitStatus(CAN1, TransmitMailbox) != CAN_TxStatus_Ok)
	{
		Timeout ++;
		//如果大于超时时间，则跳出循环
		if (Timeout > 100000)
		{
			break;
		}
	}
}

uint8_t MyCAN_Transmit1(uint32_t ID,uint8_t Length,uint8_t *Data)
{
    CanTxMsg TxMessage;

    TxMessage.StdId = ID;
    TxMessage.IDE   = CAN_Id_Standard;
    TxMessage.RTR   = CAN_RTR_Data;
    TxMessage.DLC   = Length;

    for(int i=0;i<Length;i++)
    {
        TxMessage.Data[i]=Data[i];
    }

    // 非阻塞发送：只把报文放入空邮箱，不在中断里等待 CAN ACK。
    // 若总线繁忙或邮箱满，直接返回 0，避免 TIM4 中断拖住串口/机械臂控制。
    if((CAN1->TSR & CAN_TSR_TME) == 0)
    {
        return 0;
    }

    if(CAN_TransmitAtomic(&TxMessage) == CAN_TxStatus_NoMailBox)
    {
        return 0;
    }

    return 1;
}

/************************************************************************
* @name: MyCAN_ReceiveFlag
* @brief 函数用于判断接收FIFO里是否有报文，返回值为表示有报文，返回值为0表示没有报文  
*************************************************************************/
uint8_t MyCAN_ReceiveFlag(void)
{
	if (CAN_MessagePending(CAN1, CAN_FIFO1) > 0)//如果大于零，表明FIFO1里面有报文，此处的FIFO1与前面的设置一致
	{
		return 1;
	}
	return 0;
}

/*****************************************
@brief: 接收 CAN message
@param： *ID        the ID，用于做返回值
         *Length    the length of DATA，用于做返回值
         *DATA      the Data of CAN massage
注意，接收信息是需要输出参数，但C语言不支持多个值输出，
所以这里用指针表示，可以通过函数修改相应的值
*****************************************/
void MyCAN_Receive(uint32_t *ID, uint8_t *Length, uint8_t *Data)
{
	CanRxMsg RxMessage;//定义一个CanRxMsg结构体，用于存放接收报文
	CAN_Receive(CAN1, CAN_FIFO1, &RxMessage);//接收报文
	//判断接收的报文是标准ID还是扩展ID
	if (RxMessage.IDE == CAN_Id_Standard)
	{
				*ID = RxMessage.StdId;//标准ID
	}
	else
	{
				*ID = RxMessage.ExtId; //扩展ID
	}
	//判断接收报文是否为数据帧还是遥控帧
	if (RxMessage.RTR == CAN_RTR_Data)//是否为数据帧
	{
		//数据帧
		*Length = RxMessage.DLC;//数据长度
		//数据内容
		for (uint8_t i = 0; i < *Length; i ++)
		{
			Data[i] = RxMessage.Data[i];
		}
	}
	else
	{
		//遥控帧，暂时不做处理
	}
}
