#include "stm32f10x.h"                  // Device header
#include "Chassis.h"
#include "PID.h"
#include "Feedback.h"
#include "stm32f10x_tim.h"
#include "Usart1.h"
#include "can.h"

extern volatile int16_t error;
extern volatile int16_t Steer_Angle;
extern volatile uint8_t debug_send_flag;
extern volatile uint16_t Dogtime;
extern volatile uint8_t Dog_en;
extern volatile uint8_t Dog_FLAG;
extern volatile uint8_t Chassis_Run_Flag;
extern volatile uint16_t Chassis_Run_Count;
extern volatile uint8_t Chassis_Run_Lock;
extern volatile uint16_t Chassis_Car_Silence_Count;
extern volatile uint16_t Chassis_Command_Age_Ticks;

#define CHASSIS_RUN_TICKS 250 //CHASSIS_RUN_TICKS*CHASSIS_RUN_SPEED_RPM=250000
#define CHASSIS_RUN_SPEED_RPM 500
#define CHASSIS_IDLE_KEEPALIVE_TICKS 5
#define CHASSIS_RESTART_UNLOCK_TICKS 30
#define CHASSIS_COMMAND_TIMEOUT_TICKS 20

static void Chassis_SendKeepaliveSafe(void)
{
    if(Arm_CAN_Tx_Busy)
    {
        Chassis_Keepalive_Pending = 1;
        return;
    }

    Chassis_SendCmd();
    Chassis_Keepalive_Pending = (IF == 0) ? 1 : 0;
}

void TIM4_Init(void)
{
	RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM4, ENABLE);
	
	TIM_InternalClockConfig(TIM4);
	
	TIM_TimeBaseInitTypeDef TIM_TimeBaseInitStructure;
	TIM_TimeBaseInitStructure.TIM_Period = 1000-1;
	TIM_TimeBaseInitStructure.TIM_Prescaler = 720-1;
	TIM_TimeBaseInitStructure.TIM_ClockDivision = TIM_CKD_DIV1;
	TIM_TimeBaseInitStructure.TIM_CounterMode = TIM_CounterMode_Up;
	TIM_TimeBaseInit(TIM4, &TIM_TimeBaseInitStructure);
	
	TIM_ITConfig(TIM4, TIM_IT_Update, ENABLE);
	
	NVIC_InitTypeDef NVIC_InitStructure;
	NVIC_InitStructure.NVIC_IRQChannel = TIM4_IRQn;
	NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 1;
	NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0;
	NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
	NVIC_Init(&NVIC_InitStructure);

	TIM_Cmd(TIM4, ENABLE);
}

void TIM2_Init(void)
{
	RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM2,ENABLE);
	
	TIM_InternalClockConfig(TIM2);
	
	TIM_TimeBaseInitTypeDef TIM_TimeBaseInitStructure;
	TIM_TimeBaseInitStructure.TIM_Period = 1000-1;
	TIM_TimeBaseInitStructure.TIM_Prescaler = 720-1;
	TIM_TimeBaseInitStructure.TIM_ClockDivision = TIM_CKD_DIV1;
	TIM_TimeBaseInitStructure.TIM_CounterMode = TIM_CounterMode_Up;
	TIM_TimeBaseInit(TIM2, &TIM_TimeBaseInitStructure);
	
	TIM_ITConfig(TIM2, TIM_IT_Update, ENABLE);
	
	NVIC_InitTypeDef NVIC_InitStructure;
	NVIC_InitStructure.NVIC_IRQChannel = TIM2_IRQn;
	NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 2;
	NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0;
	NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
	NVIC_Init(&NVIC_InitStructure);
}

void TIM4_IRQHandler(void)   //10ms杩涗竴娆?
{
    if(TIM_GetITStatus(TIM4,TIM_IT_Update)!=RESET)
    {
        TIM_ClearITPendingBit(TIM4,TIM_IT_Update);

        if(Chassis_Run_Flag)
        {
            if(Chassis_Command_Age_Ticks < CHASSIS_COMMAND_TIMEOUT_TICKS)
            {
                Chassis_Command_Age_Ticks++;
                Steer_Angle = PID_Calc(error);
                Chassis_SetSteer(Steer_Angle);
                Chassis_SetSpeed(CHASSIS_RUN_SPEED_RPM);
                Chassis_Run_Count++;
            }
            else
            {
                // 上位机控制帧超过200ms未更新时暂停，禁止持续沿用旧转向值。
                PID_Reset();
                error = 0;
                Steer_Angle = 0;
                Chassis_SetSteer(0);
                Chassis_SetSpeed(0);
            }

            Chassis_SendKeepaliveSafe();
            debug_send_flag = 1;

            if(Chassis_Run_Count >= CHASSIS_RUN_TICKS)
            {
                Chassis_Run_Flag = 0;
                Chassis_Run_Count = 0;
                Chassis_Command_Age_Ticks = 0;

                Chassis_SetSteer(0);
                Chassis_SetSpeed(0);
                Chassis_SendKeepaliveSafe();

                Chassis_Run_Lock = 1;
                Chassis_Car_Silence_Count = 0;

                USART_SendByte(0x07);
            }
        }
        else
        {
            static uint8_t idle_keepalive_count = 0;

            if(Chassis_Run_Lock)
            {
                if(Chassis_Car_Silence_Count < CHASSIS_RESTART_UNLOCK_TICKS)
                {
                    Chassis_Car_Silence_Count++;
                }
                else
                {
                    Chassis_Run_Lock = 0;
                    Chassis_Car_Silence_Count = 0;
                }
            }

            Chassis_SetSteer(0);
            Chassis_SetSpeed(0);

            idle_keepalive_count++;
            if(idle_keepalive_count >= CHASSIS_IDLE_KEEPALIVE_TICKS)
            {
                idle_keepalive_count = 0;
                Chassis_SendKeepaliveSafe();
            }
        }
    }
}

void TIM2_IRQHandler(void)   //10ms杩涗竴娆?.01s
{
	if(TIM_GetITStatus(TIM2,TIM_IT_Update)!=RESET)
	{
		TIM_ClearITPendingBit(TIM2,TIM_IT_Update);
		if(Dog_en==1)
		{
			Dogtime++;
			if(Dogtime>=2000)//20s
			{
				Dog_FLAG=1;
				Dog_en=0;
				Dogtime=0;
			}
		}
	}
}
