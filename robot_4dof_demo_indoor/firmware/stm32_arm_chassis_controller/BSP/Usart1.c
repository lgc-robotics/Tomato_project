#include "stm32f10x.h"                  // Device header
#include "Protocol.h"
#include "Delay.h"

extern volatile uint8_t receive_action;
extern volatile uint8_t Serial_RxPacket[CAR_FRAME_SIZE];

void USART1_Init(void)
{
    RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA, ENABLE);
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_USART1, ENABLE);

    GPIO_InitTypeDef GPIO_InitStructure;

    // PA9 TX
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_9;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_Init(GPIOA, &GPIO_InitStructure);

    // PA10 RX
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_10;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IPU;
    GPIO_Init(GPIOA, &GPIO_InitStructure);

		  // USART控制器初始化
	USART_InitTypeDef USART_InitStructure;
	USART_InitStructure.USART_BaudRate = 115200;
	USART_InitStructure.USART_WordLength = USART_WordLength_8b;
	USART_InitStructure.USART_StopBits = USART_StopBits_1;
	USART_InitStructure.USART_Parity = USART_Parity_No;
	USART_InitStructure.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
	USART_InitStructure.USART_Mode = USART_Mode_Tx | USART_Mode_Rx;
	USART_Init(USART1, &USART_InitStructure);

    USART_ITConfig(USART1, USART_IT_RXNE, ENABLE);

    USART_Cmd(USART1, ENABLE);

    // 中断配置
    NVIC_EnableIRQ(USART1_IRQn);
}


void USART_SendByte(uint8_t byte)
{
//	NVIC_DisableIRQ(USB_LP_CAN1_RX0_IRQn);
    USART_SendData(USART1, byte);
    while(USART_GetFlagStatus(USART1, USART_FLAG_TXE) == RESET);
//	NVIC_EnableIRQ(USB_LP_CAN1_RX0_IRQn);
}

void USART1_IRQHandler(void)
{
    if(USART_GetITStatus(USART1, USART_IT_RXNE) != RESET)
    {
        uint8_t byte = USART_ReceiveData(USART1);
        //传给协议层
        Protocol_Input(byte);
    }
}

void RS485_UsartSend(uint8_t byte2)
{
	GPIO_SetBits(GPIOD,GPIO_Pin_7);
	
	USART_SendData(USART2,byte2);
	
	while(USART_GetFlagStatus(USART2,USART_FLAG_TC)==RESET);
	delay_ms(30);
	
	GPIO_ResetBits(GPIOD,GPIO_Pin_7);
}

void USART2_IRQHandler(void)
{
	if(USART_GetITStatus(USART2,USART_IT_RXNE)!=RESET)
	{
		uint8_t Data=USART_ReceiveData(USART2);
		
		if(Data==0x01||Data==0x02)
		{
			receive_action=1;
		}
	}
}
