#ifndef __MCF302CB_H
#define __MCF302CB_H

#include "can.h"

// 电机状态和错误标志位定义
typedef enum {
    MOTOR_STATE_RUNNING = 0x00,
    MOTOR_STATE_STOPPED = 0x10
} MotorState;

typedef enum {
    ERROR_VOLTAGE_LOW = (1 << 0),
    ERROR_TEMP_HIGH = (1 << 3)
} ErrorState;

// 抱闸器状态定义
typedef enum {
    BRAKE_ENGAGED = 0x00,   // 抱闸器断电，刹车启动
    BRAKE_DISENGAGED = 0x01 // 抱闸器通电，刹车释放
} BrakeState;

// 旋转方向定义
typedef enum {
    DIRECTION_CW = 0x00,    // 顺时针
    DIRECTION_CCW = 0x01    // 逆时针
} RotationDirection;

// 参数索引定义
typedef enum {
    PARAM_TORQUE_LIMIT = 10,
    PARAM_ACCEL_LIMIT = 12,
    PARAM_SPEED_LIMIT = 14
} ParamIndex;


// 基本控制命令
void CAN_MotorRun(uint8_t id);
void CAN_MotorStop(uint8_t id);
void CAN_MotorShutdown(uint8_t id);
void CAN_MotorBrakeControl(uint8_t id, BrakeState state);

// 开环和闭环控制
void CAN_MotorOpenLoopControl(uint8_t id, int16_t power);
void CAN_MotorTorqueControl(uint8_t id, int16_t torque);
void CAN_MotorSpeedControl1(uint8_t id, int32_t speed);
void CAN_MotorSpeedControl2(uint8_t id, int32_t speed, int16_t torque);
void CAN_MotorMultiTurnPosition1(uint8_t id, int32_t position);
void CAN_MotorMultiTurnPosition2(uint8_t id, int32_t position, uint16_t max_speed);
void CAN_MotorSingleTurnPosition1(uint8_t id, RotationDirection dir, uint32_t position);
void CAN_MotorSingleTurnPosition2(uint8_t id, RotationDirection dir, uint32_t position, uint16_t max_speed);
void CAN_MotorIncrementalPosition1(uint8_t id, int32_t increment);
void CAN_MotorIncrementalPosition2(uint8_t id, int32_t increment, uint16_t max_speed);

// 状态读取命令
void CAN_ReadMotorStatus1(uint8_t id);
void CAN_ReadMotorStatus2(uint8_t id);
void CAN_ReadMotorStatus3(uint8_t id);
void CAN_ReadEncoderData(uint8_t id);
void CAN_ReadMultiTurnAngle(uint8_t id);
void CAN_ReadSingleTurnAngle(uint8_t id);

// 参数配置命令
void CAN_ReadControlParameter(uint8_t id, ParamIndex index);
void CAN_SetControlParameter(uint8_t id, ParamIndex index, uint8_t data[6]);
void CAN_SetEncoderOffset(uint8_t id, uint16_t offset);
void CAN_SaveCurrentPositionAsZero(uint8_t id);
void CAN_SetMultiTurnAngle(uint8_t id, int32_t angle);


#endif
