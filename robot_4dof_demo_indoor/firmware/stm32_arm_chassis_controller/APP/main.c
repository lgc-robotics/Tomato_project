#include "stm32f10x.h"                  // Device header
#include "Delay.h"                  // Device header
#include "Usart1.h"
#include "Protocol.h"
#include "BOARD.h"
#include "Emm_V5.h"
#include "can.h"
#include "cybergear1.h"
#include "Mycan.h"
#include "Miparam.h"
#include <string.h>
#include <math.h>
#include "stm32f10x_usart.h"
#include "stm32f10x_tim.h"   
#include <stdio.h>
#include <stdlib.h>
#include "PID.h"
#include "Chassis.h"
#include "PID.h"
#include "Feedback.h"
#include "Timer.h"

int debug_axis=3;
double path_x, path_y, path_z;
float x,y,z;
volatile float targetAngle = 0;
volatile uint16_t raw_angle = 0;     // 当前角度（原始值）
volatile uint16_t target_raw = 0;    // 目标角度（原始值）
volatile uint8_t receive_action=0;
volatile uint8_t cyflag=0;
volatile int16_t error;
volatile int16_t Steer_Angle;
volatile uint8_t debug_send_flag = 0;
volatile uint8_t Dog_en=0;
volatile uint8_t Dog_FLAG=0;
volatile uint16_t Dogtime=0;
volatile uint8_t Chassis_Run_Flag = 0;
volatile uint16_t Chassis_Run_Count = 0;
volatile uint8_t Chassis_Run_Lock = 0;
volatile uint16_t Chassis_Car_Silence_Count = 0;
volatile uint16_t Chassis_Command_Age_Ticks = 0;

#define MOVE_STATUS_X_REACHED 0x01
#define MOVE_STATUS_Y_REACHED 0x02
#define MOVE_STATUS_Z_REACHED 0x04
#define MOVE_STATUS_CAN_TX_ERROR 0x40
#define MOVE_STATUS_WATCHDOG  0x80
#define AXIS_NO_MOVE_EPS_MM   0.5f

volatile uint8_t Move_Last_Status = 0;
static float last_target_x_mm = 0.0f;
static float last_target_y_mm = 0.0f;
static float last_target_z_mm = 0.0f;
static uint8_t last_target_valid = 0;

static void Arm_CAN_Burst_Begin(void)
{
  // 机械臂组帧前先补一帧底盘停止保活，随后只短暂锁住CAN发送入口�?
  Chassis_SetSteer(0);
  Chassis_SetSpeed(0);
  Chassis_SendCmd();
  Chassis_Keepalive_Pending = (IF == 0) ? 1 : 0;

  Arm_CAN_Tx_Error = 0;
  Arm_CAN_Tx_Busy = 1;
}

static void Arm_CAN_Burst_End(void)
{
  Arm_CAN_Tx_Busy = 0;

  // TIM4在机械臂组帧期间若错过保活，解除锁后立即补发�?
  if(Chassis_Keepalive_Pending)
  {
    Chassis_SendCmd();
    Chassis_Keepalive_Pending = (IF == 0) ? 1 : 0;
  }
}

static void Recover_Stepper_After_Move_Failure(void)
{
  // 广播立即停止，用于清除驱动器内可能残留的不完整同步运动状态�?
  Arm_CAN_Burst_Begin();
  Emm_V5_Stop_Now(0x00, false);
  delay_ms(10);
  Arm_CAN_Burst_End();
}

// STM32F1 标准库，重定�?printf �?USART1
int fputc(int ch, FILE *f)
{
    USART_SendByte((uint8_t)ch); // 调用写好�?USART 发送函�?
    return ch;
}
//机械臂坐标解析函数，使用volatile指针
void ParseCoordinate(volatile uint8_t *data, float *x, float *y, float *z)
{
  // 小端序转换并除以100
  *x = (float)((data[3] << 24) | (data[2] << 16) | (data[1] << 8) | data[0]) / 100.0f;
  *y = (float)((data[7] << 24) | (data[6] << 16) | (data[5] << 8) | data[4]) / 100.0f;
  *z = (float)((data[11]<< 24) | (data[10] << 16)| (data[9] << 8) | data[8]) / 100.0f;
}

//旋转关节角度解析函数
float ParseAngle(volatile uint8_t *data)
{
  int32_t angle_int = (int32_t)(data[0] | (data[1] << 8) | (data[2] << 16) | (data[3] << 24));
  delay_ms(10);
  return (float)angle_int / 100.0f;
}

//机械臂坐标执行函数，发送机械臂坐标并等待ACK
uint8_t DebugMove(float x, float y, float z)
{
  uint8_t move_x = 0;
  uint8_t move_y = 0;
  uint8_t move_z = 0;
  uint8_t move_ok = 0;
  uint8_t can_tx_failed = 0;

  Move_Last_Status = 0;
  Dog_FLAG=0;
  Dogtime=0;

  move_y = ((debug_axis == 1 || debug_axis == 2 || debug_axis == 3) && (!last_target_valid || (fabsf(y - last_target_y_mm) > AXIS_NO_MOVE_EPS_MM)));
  move_z = ((debug_axis == 2 || debug_axis == 3) && (!last_target_valid || (fabsf(z - last_target_z_mm) > AXIS_NO_MOVE_EPS_MM)));
  move_x = ((debug_axis == 3) && (!last_target_valid || (fabsf(x - last_target_x_mm) > AXIS_NO_MOVE_EPS_MM)));

  xReached = move_x ? false : true;
  yReached = move_y ? false : true;
  zReached = move_z ? false : true;

  if(move_x || move_y || move_z)
  {
    Arm_CAN_Burst_Begin();

    if (move_y)
    {
      path_y = y * PULSE_PER_MM_XY;
      Emm_V5_Pos_Control(MOTOR_ID_Y, 0, Y_MAX_SPEED, MAX_ACCEL, path_y, 1, 1);
      delay_ms(10);
    }
    if (move_z)
    {
      path_z = z * PULSE_PER_MM_Z;
      Emm_V5_Pos_Control(MOTOR_ID_Z, 0, Z_MAX_SPEED, 225, path_z, 1, 1);
      delay_ms(10);
    }
    if (move_x)
    {
      path_x = x * PULSE_PER_MM_XY;
      Emm_V5_Pos_Control(MOTOR_ID_X, 0, X_MAX_SPEED, MAX_ACCEL, path_x, 1, 1);
      delay_ms(10);
    }

    if(!Arm_CAN_Tx_Error)
    {
      Emm_V5_Synchronous_motion(0x00);
    }

    can_tx_failed = Arm_CAN_Tx_Error;
    Arm_CAN_Burst_End();

    if(!can_tx_failed)
    {
      TIM_SetCounter(TIM2, 0);
      TIM_ClearITPendingBit(TIM2, TIM_IT_Update);
      TIM_Cmd(TIM2, ENABLE);
      Dog_en=1;
      Dog_FLAG=0;
      Dogtime=0;
      while (!(xReached && yReached && zReached)&&(Dog_FLAG==0));
    }
  }

  if(xReached) Move_Last_Status |= MOVE_STATUS_X_REACHED;
  if(yReached) Move_Last_Status |= MOVE_STATUS_Y_REACHED;
  if(zReached) Move_Last_Status |= MOVE_STATUS_Z_REACHED;
  if(can_tx_failed) Move_Last_Status |= MOVE_STATUS_CAN_TX_ERROR;
  if(Dog_FLAG) Move_Last_Status |= MOVE_STATUS_WATCHDOG;

  move_ok = (xReached && yReached && zReached && (Dog_FLAG == 0) && !can_tx_failed);

  if(move_ok)
  {
    last_target_x_mm = x;
    last_target_y_mm = y;
    last_target_z_mm = z;
    last_target_valid = 1;
  }
  else
  {
    // 超时后实际位置不再可信，下一�?4必须重新下发三轴命令�?
    last_target_valid = 0;
  }

  Dog_en=0;
  TIM_Cmd(TIM2, DISABLE);

  if(Dog_FLAG || can_tx_failed)
  {
    Recover_Stepper_After_Move_Failure();
  }

  yReached = false;
  zReached = false;
  xReached = false;

  return move_ok;
}
//发送末端执行器,RS485信号
void RS485_Action(uint8_t RS485)
{
	if(RS485==0x01)
	{
		RS485_UsartSend(0x01);
		delay_ms(1000);
		USART_SendByte(0x06);
	}
	if(RS485==0x02)
	{
		RS485_UsartSend(0x02);
		delay_ms(1000);
		USART_SendByte(0x06);
	}	
}

extern uint8_t Car_Status;
extern uint8_t Car_Gear;
extern uint8_t IF;

void Send_Debug_Frame(void)
{
    uint8_t data[9];

    data[0] = 0xFF;
    data[1] = 0x08;

    data[2] = error & 0xFF;
    data[3] = (error >> 8) & 0xFF;

    data[4] = Car_Status;
    data[5] = Car_Gear;
    data[6] = IF;

    data[7] = 0x00;

    data[8] = 0xFE;

    for(int i=0;i<9;i++)
    {
        USART_SendByte(data[i]);
    }
}

int main(void)
{	
	delay_ms(2000);
	USART1_Init();
	board_init();
	PID_Init();
	TIM4_Init();
	delay_ms(100);
	TIM2_Init();

	delay_ms(100);
	//设定机械0点
	Emm_V5_Reset_CurPos_To_Zero(MOTOR_ID_X);
	Emm_V5_Reset_CurPos_To_Zero(MOTOR_ID_Y);
	Emm_V5_Reset_CurPos_To_Zero(MOTOR_ID_Z);
	delay_ms(100);
	
	init_cybergear(&cyber, 0x7F, Position_mode);//初始化旋转关节，设定零位
	delay_ms(30);
	pos_mode(&cyber, 5, 0);  // 旋转关节固定在0点位置
	delay_ms(100);
	RS485_UsartSend(0x01);//初始化末端位姿张开
	
	while(1)
	{
		if(frameReady)  //检验是否完成串口数据收  帧格式FF 04/5/6/7 00 00 00 00 00 00 00 00 00 00 00 00 FE  机械臂04,旋转关节05,末端06，底盘07
		{
			frameReady = 0;  //清除标志
			float AngleA;
			float SPEED_MAX=4;
			switch(frame[1]) //根据不同类型信号执行相应动作
			{
				case coordinate://arm position command
				{
					uint8_t move_ok;
					Chassis_Run_Flag = 0;
					Chassis_Run_Count = 0;
					delay_ms(5);
					ParseCoordinate(&frame[2],&x,&y,&z);
					move_ok = DebugMove(x,y,z);
					delay_ms(10);
					if(move_ok)
					{
						USART_SendByte(0x04);
					}
					else
					{
						USART_SendByte(0xE4);
						USART_SendByte(Move_Last_Status);
					}
				}
				break;
		
				case ANgle://旋转关节角度信号
					Chassis_Run_Flag = 0;
					Chassis_Run_Count = 0;
					AngleA=ParseAngle(&frame[2]);
					AngleA = AngleA * 3.1415926f / 180.0f;
					if(AngleA > 12.5 || AngleA < -12.5)
					{
						USART_SendByte(0xEE);  
					}
					targetAngle = AngleA;
					target_raw = float_to_uint(targetAngle, P_MIN, P_MAX, 16);
					pos_mode(&cyber,SPEED_MAX,AngleA);
					delay_ms(5);
					USART_SendByte(0x05);
				break;
		
				case action://末端执行器信
					Chassis_Run_Flag = 0;
					Chassis_Run_Count = 0;
					RS485_Action(frame[2]);
				break;

				case car://底盘信号
				{
					error = (int16_t)(frame[2] |(frame[3] << 8));
					Chassis_Command_Age_Ticks = 0;

					// 定时运动结束后，Python 串口缓冲里可能还有几�?0x07 控制帧�?
					// 锁定期间这些旧帧只更�?error，不允许再次启动一段定时运动�?
					if(Chassis_Run_Lock)
					{
						Chassis_Car_Silence_Count = 0;
					}

					if(Chassis_Run_Flag == 0 && Chassis_Run_Lock == 0)
					{
						PID_	Reset();
						Chassis_Run_Flag = 1;
						Chassis_Run_Count = 0;
						Chassis_Car_Silence_Count = 0;
					}
				}
				break;
			}
		}
		if(debug_send_flag)
		{
			debug_send_flag = 0;
			Send_Debug_Frame();
		}
		Chassis_Feedback_Update();
	}
}

/*手眼标定
FF 04 00 00 00 00 00 00 00 00 00 00 00 00 FE
FF 04 88 13 00 00 B8 0B 00 00 D0 07 00 00 FE
FF 04 08 52 00 00 F0 55 00 00 60 6D 00 00 FE
FF 04 28 23 00 00 B0 36 00 00 E8 03 00 00 FE
FF 05 28 23 00 00 00 00 00 00 00 00 00 00 FE
FF 05 94 11 00 00 00 00 00 00 00 00 00 00 FE
FF 05 50 46 00 00 00 00 00 00 00 00 00 00 FE
FF 05 00 00 00 00 00 00 00 00 00 00 00 00 FE
FF 06 02 00 00 00 00 00 00 00 00 00 00 00 FE
*/
