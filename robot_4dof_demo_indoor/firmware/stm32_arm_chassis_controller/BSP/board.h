#ifndef __BOARD_H
#define __BOARD_H
#include "stm32f10x.h"

#define STEPS_PER_REV 3200.0f                        // 1.8° * 16细分 = 3200脉冲
// XY轴丝杆参数（根据实际机械结构调整以下值）
#define SCREW_LEAD 10                            // 丝杆导程10mm
#define PULSE_PER_MM_XY (STEPS_PER_REV / SCREW_LEAD) // 320脉冲/mm  （每毫米需要脉冲 = 3200 / 10）
// Z轴导轨滑台参数
#define GEAR_RATIO_Z        1.0f   // 齿轮减速比
#define BELT_PITCH_Z        2      // 同步带齿距(mm)
#define PULLEY_TEETH_Z      35     // 主动轮齿数
#define CALIBRATION_SCALE_Z (150.0f / 160.0f)   // 误差修正：目标 150mm / 实际 215mm = 0.697674y 
#define PULSE_PER_MM_Z      ((STEPS_PER_REV * GEAR_RATIO_Z) / (BELT_PITCH_Z * PULLEY_TEETH_Z) * CALIBRATION_SCALE_Z)

#define X_MAX_SPEED 1000                          // RPM 1500
#define Y_MAX_SPEED 1000                           //200
#define Z_MAX_SPEED 250                           //100
#define MAX_ACCEL 240 // 加速度档位

#define MOTOR_ID_X 0x03 // X轴升降电机 CAN ID
#define MOTOR_ID_Y 0x01 // Y轴直线电机 CAN ID
#define MOTOR_ID_Z 0x02 // Z轴直线电机 CAN ID
#define MOTOR_ID_R 0x04

#define TASK_COUNT 3 // 任务数量

// 添加串口帧协议相关定义
#define FRAME_HEADER 0xFF
#define FRAME_FOOTER 0xFE
#define FRAME_SIZE 15
#define FRAME_SIZE2     6


// 动作指令定义
#define coordinate      0x04 // 坐标传输 
#define ANgle           0x05 // 角度控制 
#define action          0x06 // 动作指令 
#define car             0x07 // 底盘指令 

// 确认信号
#define ACK_COORD_DONE  0x05 // 坐标完成信号 
#define ACK_ACTION_DONE 0x07 // 角度/动作完成信号

// 到位反馈命令定义
#define REACHED_CMD_BYTE1 0xFD
#define REACHED_CMD_BYTE2 0x9F
#define REACHED_CMD_BYTE3 0x6B


// 定义坐标系结构体
typedef struct
{
    float x;
    float y;
    float z;
} Coordinate;

void nvic_init(void);
void clock_init(void);
void board_init(void);
void usart2_init(void);

#endif
