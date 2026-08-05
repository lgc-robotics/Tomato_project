#ifndef __Usart1_H
#define __Usart1_H

void USART1_Init(void);
void USART_SendByte(uint8_t byte);
void USART1_IRQHandler(void);
void RS485_UsartSend(uint8_t byte2);
void RS485_USART2_IRQHandler(void);
int16_t Serial_GetError(void);
	
#endif
