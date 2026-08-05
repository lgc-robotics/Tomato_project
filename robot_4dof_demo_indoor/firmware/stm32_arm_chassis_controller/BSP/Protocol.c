#include "stm32f10x.h"                  // Device header
#include "Usart1.h"

#define FRAME_SIZE 15

uint8_t frameReady = 0;
uint8_t frame[FRAME_SIZE];
static uint8_t index = 0;

typedef enum
{
    WAIT_HEAD,
    WAIT_TYPE,
    WAIT_DATA,
    WAIT_TAIL
} State_t;

static State_t state = WAIT_HEAD;

//帧头-类型-数据-帧尾
void Protocol_Input(uint8_t byte)
{
    switch(state)
    {
        /************************************************
        等待帧头
        ************************************************/
        case WAIT_HEAD:

            if(byte == 0xFF)
            {
                // 清零索引
                index = 0;

                // 保存帧头
                frame[index++] = byte;

                // 进入下一状态
                state = WAIT_TYPE;
            }

        break;

        /************************************************
        接收功能码
        ************************************************/
        case WAIT_TYPE:

            frame[index++] = byte;

            state = WAIT_DATA;

        break;


        /************************************************
        接收数据区
        ************************************************/
        case WAIT_DATA:
					
        frame[index++] = byte;

        // 防止数组越界
           if(index >= 14)
            {
                state = WAIT_TAIL;
            }

        break;


        /************************************************
        等待帧尾
        ************************************************/
        case WAIT_TAIL:

            if(byte == 0xFE)
            {
                // 保存帧尾
                frame[index] = byte;

                // 一帧接收完成
                frameReady = 1;
            }

            // 不管成功失败
            // 都复位状态机
            index = 0;

            state = WAIT_HEAD;

        break;


        /************************************************
        异常保护
        ************************************************/
        default:

            index = 0;

            state = WAIT_HEAD;

        break;
    }
}
