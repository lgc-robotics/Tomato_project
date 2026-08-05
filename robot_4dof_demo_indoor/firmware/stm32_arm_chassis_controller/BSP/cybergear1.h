#ifndef __cybergear1_H
#define __cybergear1_H
#include "stm32f10x.h"                  // Device header
#include "Miparam.h"

//主机CANID设置
#define Master_CAN_ID 0x00                      //主机ID
#define Temp_Gain   0.1
#define Motor_Error 0x00
#define Motor_OK 0X01
#define pi 3.141592654

/***小米电机结构体，包括ID号、回传角度、速度、力矩等参数（有待补充？）*********/
typedef struct{           //小米电机结构体
	uint8_t CAN_ID;       //CAN ID
    uint8_t MCU_ID;       //MCU唯一标识符[后8位，共64位]
	float Angle;          //回传角度
	float Speed;          //回传速度
	float Torque;         //回传力矩
	float Temp;			  //回传温度
	//uint16_t set_current;//设置电流
	//uint16_t set_speed;//设置速度
	//uint16_t set_position;//设置位置
	//uint16_t set_torque;//设置力矩
	uint8_t error_code;//错误码
	float Angle_Bias;//
	uint8_t mode;//表示电机运动模式
} MI_Motor;
extern MI_Motor cyber;
extern volatile uint16_t raw_angle;
extern volatile uint16_t target_raw;
typedef struct
{
  float Position; //设置的角度
	float Speed; //设置的角速度
	float Kp;
	float Kd;
	float Torque; //力矩
	uint8_t Current;
} SM_Param;

/*********函数声明*********/
void init_cybergear(MI_Motor *Motor, uint8_t Can_Id, uint8_t mode);//小米电机初始化
uint32_t check_cybergear(uint8_t ID);//小米电机ID检查，通信类型为0
void motor_controlmode(MI_Motor *Motor,float torque, float MechPosition, float speed, float kp, float kd);//小米运控模式指令，通信类型：1
void Rx_Fifo0_Msg(MI_Motor *Motor);//电机反馈数据，通信类型2
void start_cybergear(uint8_t Can_ID);//使能小米电机，通信类型为3
void stop_cybergear(MI_Motor *Motor,uint8_t clear_error);//停止电机，通信类型为4
void set_zeropos_cybergear(MI_Motor *Motor);//设置电机零点,通信类型为6
void set_CANID_cybergear(MI_Motor *Motor,uint8_t CAN_ID);//设置电机CANID,通信类型为7
void check_param_cybergear(MI_Motor *Motor, uint16_t index);//单个参数读取，通信类型17
void Set_Motor_Parameter(MI_Motor *Motor,uint16_t Index,float Ref);//写入电机参数，对应说明书的单个参数写入，通信类型为18
void Set_Mode_Cybergear(MI_Motor *Motor,uint8_t Mode);//设置电机模式(必须停止时调整！)
void pos_mode(MI_Motor *Motor, float speed_max,float pos);//设计电机控制模式为位置控制
void speed_mode(MI_Motor *Motor, float Cur_max,float Spd);//设置电机控制模式为速度控制
float uint16_to_float(uint16_t x,float x_min,float x_max,int bits);
int float_to_uint(float x, float x_min, float x_max, int bits);

#endif










