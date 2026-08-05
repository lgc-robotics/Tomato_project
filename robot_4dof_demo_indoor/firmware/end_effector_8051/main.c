#include <REGX52.H>
#include "Timer0.h"
#include "Timer1.h"
#include "UART.h"
#include "Key.h"
#include "Delay.h"

//控制末端刀具的开口和闭合大小
#define  start  800
#define  end1  1000 //1450  1450是完全闭合
#define  end2  1450 //2400 //2300就是舵机能选转的最大角度了（不知道怎么回事，正常来说最大值应该是2400，转180°）

#define uint8	unsigned char    
#define uint16	unsigned short int
#define uint32	unsigned long 

sbit SERVO0 = P2^7;    //上舵机
sbit SERVO1 = P2^6;    //下舵机
bit Servo0Enable = 0;   // 1=上舵机输出PWM，0=停止PWM
bit Servo1Enable = 0;   // 1=下舵机输出PWM，0=停止PWM

//定义控制引脚IO
sbit RS485_DIR = P1^0; // 发送和接收控制

uint8 c,i;
uint8 KeyNum1,KeyNum2;
uint16 Servo0PwmDuty1 = start;	//PWM脉冲宽度
uint16 Servo0PwmDuty2 = start;

void ServoDelay1s(void)
{
    unsigned char j;
    for(j = 0; j < 100; j++)
    {
        delay_10us(1000);   // 约10ms
    }
}

void main()
{
	InitTimer0();	//定时器0初始化
	EA = 1;	
	UartInit();
	
    SERVO0 = 0;
    SERVO1 = 0;	
	
	while(1)
	{	
        KeyNum1 = Key();

        if(KeyNum1 == 0x01 || KeyNum2 == 0x01)
        {
            Servo0PwmDuty1 = start;    // 上舵机复位
            Servo0Enable = 1;          // 开始输出PWM

            ServoDelay1s();            // 输出1秒PWM

            Servo0Enable = 0;          // 停止PWM
            SERVO0 = 0;                // 信号脚拉低

            KeyNum2 = 0;               // 清除串口指令，防止重复执行
        }
		
        if(KeyNum1 == 0x02 || KeyNum2 == 0x02)
        {
            Servo0PwmDuty1 = end1;     // 上舵机转到目标角度
            Servo0Enable = 1;

            ServoDelay1s();

            Servo0Enable = 0;
            SERVO0 = 0;

            KeyNum2 = 0;
        }
		
        if(KeyNum1 == 0x03 || KeyNum2 == 0x03)
        { 
            Servo0PwmDuty2 = start;    // 下舵机复位
            Servo1Enable = 1;

            ServoDelay1s();

            Servo1Enable = 0;
            SERVO1 = 0;

            KeyNum2 = 0;
        }
		
        if(KeyNum1 == 0x04 || KeyNum2 == 0x04)
        {
            Servo0PwmDuty2 = end2;     // 下舵机转到目标角度
            Servo1Enable = 1;

            ServoDelay1s();

            Servo1Enable = 0;
            SERVO1 = 0;

            KeyNum2 = 0;
        }
		
	}
	
		
}

void Timer0Value(uint16 pwm)
{
	uint16 value;
	value=0xffff-pwm;	
	TR0 = 0;
	TL0=value;			//16位数据给8位数据赋值默认将16位数据的低八位直接赋给八位数据
    TH0=value>>8;		//将16位数据右移8位，也就是将高8位移到低八位，再赋值给8位数据	
	TR0 = 1;
}

void Timer0_isr(void) interrupt 1 using 1
{
    static uint16 i = 1;

    switch(i)
    {
        case 1:
            if(Servo0Enable)
            {
                SERVO0 = 1;
            }
            else
            {
                SERVO0 = 0;
            }

            Timer0Value(Servo0PwmDuty1);	
            break;

        case 2:
            SERVO0 = 0;
            Timer0Value(2500 - Servo0PwmDuty1);	
            break;	 
		
        case 3:
            if(Servo1Enable)
            {
                SERVO1 = 1;
            }
            else
            {
                SERVO1 = 0;
            }

            Timer0Value(Servo0PwmDuty2);	
            break;

        case 4:
            SERVO1 = 0;
            Timer0Value(2500 - Servo0PwmDuty2);
            i = 0;	
            break;	 
    }
	
    i++;
}

void uart() interrupt 4 //串口通信中断函数
{
    RI = 0;            // 清除接收中断标志位
    KeyNum2 = SBUF; // 存储接收到的数据
    delay_10us(100);
    RS485_DIR = 1; // 配置RS485为发送模式
    SBUF = KeyNum2; // 将接收到的数据放入到发送寄存器
    while(!TI);     // 等待发送数据完成
    TI = 0;    // 清除发送完成标志位
    RS485_DIR = 0; // 配置RS485为接收模式                
}