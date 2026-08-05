#include "MCF302CB.h"

/**
  * @brief  发送电机运行命令
  * @param  id 电机ID (1-32)
  * @note   将电机从关闭状态切换到开启状态，LED由慢闪转为常亮
  */
void CAN_MotorRun(uint8_t id) {
    uint8_t data[8] = {0x88}; // 命令字节: 0x88 = 运行命令
    CAN_SendFrame(id, data);
}

/**
  * @brief  发送电机停止命令
  * @param  id 电机ID (1-32)
  * @note   停止电机但不改变运行状态，可再次发送指令控制
  */
void CAN_MotorStop(uint8_t id) {
    uint8_t data[8] = {0x81}; // 命令字节: 0x81 = 停止命令
    CAN_SendFrame(id, data);
}

/**
  * @brief  发送电机关闭命令
  * @param  id 电机ID (1-32)
  * @note   将电机切换到关闭状态，清除转动圈数及之前的控制指令
  */
void CAN_MotorShutdown(uint8_t id) {
    uint8_t data[8] = {0x80}; // 命令字节: 0x80 = 关闭命令
    CAN_SendFrame(id, data);
}

/**
  * @brief  控制抱闸器状态
  * @param  id 电机ID (1-32)
  * @param  state 抱闸器状态 (BRAKE_ENGAGED:抱闸, BRAKE_DISENGAGED:释放)
  */
void CAN_MotorBrakeControl(uint8_t id, BrakeState state) {
    uint8_t data[8] = {0x8C, (uint8_t)state}; // 命令字节: 0x8C = 抱闸控制
    CAN_SendFrame(id, data);
}

/**
  * @brief  开环控制命令（仅MS电机有效）
  * @param  id 电机ID (1-32)
  * @param  power 开环控制值 (-850~850)
  * @note   控制输出到电机的开环电压，数值范围-850~850
  */
void CAN_MotorOpenLoopControl(uint8_t id, int16_t power) {
    uint8_t data[8] = {0xA0, 0, 0, 0}; // 命令字节: 0xA0 = 开环控制
    data[4] = power & 0xFF;        // 功率值低字节
    data[5] = (power >> 8) & 0xFF; // 功率值高字节
    CAN_SendFrame(id, data);
}

/**
  * @brief  转矩闭环控制命令（仅MF、MH、MG电机有效）
  * @param  id 电机ID (1-32)
  * @param  torque 转矩电流值 (-2048~2048)
  * @note   -2048~2048对应MF电机实际转矩电流范围-16.5A~16.5A
  *         MG电机对应实际转矩电流范围-33A~33A
  */
void CAN_MotorTorqueControl(uint8_t id, int16_t torque) {
    uint8_t data[8] = {0xA1, 0, 0, 0}; // 命令字节: 0xA1 = 转矩控制
    data[4] = torque & 0xFF;        // 转矩值低字节
    data[5] = (torque >> 8) & 0xFF; // 转矩值高字节
    CAN_SendFrame(id, data);
}

/**
  * @brief  速度闭环控制命令1
  * @param  id 电机ID (1-32)
  * @param  speed 目标速度 (0.01dps/LSB, 36000=360度/秒)
  * @note   速度值由上位机中的Max Speed限制
  */
void CAN_MotorSpeedControl1(uint8_t id, int32_t speed) {
    uint8_t data[8] = {0xA2}; // 命令字节: 0xA2 = 速度控制1
    // 将32位速度值分解为4个字节 (小端序)
    data[4] = speed & 0xFF;         // 最低字节
    data[5] = (speed >> 8) & 0xFF;  // 次低字节
    data[6] = (speed >> 16) & 0xFF; // 次高字节
    data[7] = (speed >> 24) & 0xFF; // 最高字节
    CAN_SendFrame(id, data);
}

/**
  * @brief  速度闭环控制命令2（带转矩限制）
  * @param  id 电机ID (1-32)
  * @param  speed 目标速度 (0.01dps/LSB, 36000=360度/秒)
  * @param  torque 最大转矩电流值 (-2048~2048)
  * @note   速度值由上位机中的Max Speed限制，转矩值限制电流
  */
void CAN_MotorSpeedControl2(uint8_t id, int32_t speed, int16_t torque) {
    uint8_t data[8] = {0xAD}; // 命令字节: 0xAD = 速度控制2
    data[2] = torque & 0xFF;        // 转矩值低字节
    data[3] = (torque >> 8) & 0xFF; // 转矩值高字节
    // 将32位速度值分解为4个字节 (小端序)
    data[4] = speed & 0xFF;         // 最低字节
    data[5] = (speed >> 8) & 0xFF;  // 次低字节
    data[6] = (speed >> 16) & 0xFF; // 次高字节
    data[7] = (speed >> 24) & 0xFF; // 最高字节
    CAN_SendFrame(id, data);
}

/**
  * @brief  多圈位置闭环控制命令1
  * @param  id 电机ID (1-32)
  * @param  position 目标位置 (0.01度/LSB, 36000=360度)
  * @note   转动方向由目标位置和当前位置的差值决定
  */
void CAN_MotorMultiTurnPosition1(uint8_t id, int32_t position) {
    uint8_t data[8] = {0xA3}; // 命令字节: 0xA3 = 多圈位置控制1
    // 将32位位置值分解为4个字节 (小端序)
    data[4] = position & 0xFF;         // 最低字节
    data[5] = (position >> 8) & 0xFF;  // 次低字节
    data[6] = (position >> 16) & 0xFF; // 次高字节
    data[7] = (position >> 24) & 0xFF; // 最高字节
    CAN_SendFrame(id, data);
}

/**
  * @brief  多圈位置闭环控制命令2（带速度限制）
  * @param  id 电机ID (1-32)
  * @param  position 目标位置 (0.01度/LSB, 36000=360度)
  * @param  max_speed 最大速度 (1dps/LSB, 360=360度/秒)
  * @note   转动方向由目标位置和当前位置的差值决定
  */
void CAN_MotorMultiTurnPosition2(uint8_t id, int32_t position, uint16_t max_speed) {
    uint8_t data[8] = {0xA4}; // 命令字节: 0xA4 = 多圈位置控制2
    data[2] = max_speed & 0xFF;        // 速度限制低字节
    data[3] = (max_speed >> 8) & 0xFF; // 速度限制高字节
    // 将32位位置值分解为4个字节 (小端序)
    data[4] = position & 0xFF;         // 最低字节
    data[5] = (position >> 8) & 0xFF;  // 次低字节
    data[6] = (position >> 16) & 0xFF; // 次高字节
    data[7] = (position >> 24) & 0xFF; // 最高字节
    CAN_SendFrame(id, data);
}

/**
  * @brief  单圈位置闭环控制命令1
  * @param  id 电机ID (1-32)
  * @param  dir 旋转方向 (DIRECTION_CW:顺时针, DIRECTION_CCW:逆时针)
  * @param  position 目标位置 (0.01度/LSB, 36000=360度)
  * @note   位置值范围0-36000*减速比-1
  */
void CAN_MotorSingleTurnPosition1(uint8_t id, RotationDirection dir, uint32_t position) {
    uint8_t data[8] = {0xA5, (uint8_t)dir}; // 命令字节: 0xA5 = 单圈位置控制1
    // 将32位位置值分解为4个字节 (小端序)
    data[4] = position & 0xFF;         // 最低字节
    data[5] = (position >> 8) & 0xFF;  // 次低字节
    data[6] = (position >> 16) & 0xFF; // 次高字节
    data[7] = (position >> 24) & 0xFF; // 最高字节
    CAN_SendFrame(id, data);
}

/**
  * @brief  单圈位置闭环控制命令2（带速度限制）
  * @param  id 电机ID (1-32)
  * @param  dir 旋转方向 (DIRECTION_CW:顺时针, DIRECTION_CCW:逆时针)
  * @param  position 目标位置 (0.01度/LSB, 36000=360度)
  * @param  max_speed 最大速度 (1dps/LSB, 360=360度/秒)
  * @note   位置值范围0-36000*减速比-1
  */
void CAN_MotorSingleTurnPosition2(uint8_t id, RotationDirection dir, uint32_t position, uint16_t max_speed) {
    uint8_t data[8] = {0xA6, (uint8_t)dir}; // 命令字节: 0xA6 = 单圈位置控制2
    data[2] = max_speed & 0xFF;        // 速度限制低字节
    data[3] = (max_speed >> 8) & 0xFF; // 速度限制高字节
    // 将32位位置值分解为4个字节 (小端序)
    data[4] = position & 0xFF;         // 最低字节
    data[5] = (position >> 8) & 0xFF;  // 次低字节
    data[6] = (position >> 16) & 0xFF; // 次高字节
    data[7] = (position >> 24) & 0xFF; // 最高字节
    CAN_SendFrame(id, data);
}

/**
  * @brief  增量位置闭环控制命令1
  * @param  id 电机ID (1-32)
  * @param  increment 位置增量 (0.01度/LSB, 36000=360度)
  * @note   正值为顺时针增加，负值为逆时针减少
  */
void CAN_MotorIncrementalPosition1(uint8_t id, int32_t increment) {
    uint8_t data[8] = {0xA7}; // 命令字节: 0xA7 = 增量位置控制1
    // 将32位增量值分解为4个字节 (小端序)
    data[4] = increment & 0xFF;         // 最低字节
    data[5] = (increment >> 8) & 0xFF;  // 次低字节
    data[6] = (increment >> 16) & 0xFF; // 次高字节
    data[7] = (increment >> 24) & 0xFF; // 最高字节
    CAN_SendFrame(id, data);
}

/**
  * @brief  增量位置闭环控制命令2（带速度限制）
  * @param  id 电机ID (1-32)
  * @param  increment 位置增量 (0.01度/LSB, 36000=360度)
  * @param  max_speed 最大速度 (1dps/LSB, 360=360度/秒)
  * @note   正值为顺时针增加，负值为逆时针减少
  */
void CAN_MotorIncrementalPosition2(uint8_t id, int32_t increment, uint16_t max_speed) {
    uint8_t data[8] = {0xA8}; // 命令字节: 0xA8 = 增量位置控制2
    data[2] = max_speed & 0xFF;        // 速度限制低字节
    data[3] = (max_speed >> 8) & 0xFF; // 速度限制高字节
    // 将32位增量值分解为4个字节 (小端序)
    data[4] = increment & 0xFF;         // 最低字节
    data[5] = (increment >> 8) & 0xFF;  // 次低字节
    data[6] = (increment >> 16) & 0xFF; // 次高字节
    data[7] = (increment >> 24) & 0xFF; // 最高字节
    CAN_SendFrame(id, data);
}

/**
  * @brief  读取电机状态1和错误标志
  * @param  id 电机ID (1-32)
  * @note   读取温度、电压、电流、状态和错误标志
  */
void CAN_ReadMotorStatus1(uint8_t id) {
    uint8_t data[8] = {0x9A}; // 命令字节: 0x9A = 读取状态1
    CAN_SendFrame(id, data);
}

/**
  * @brief  读取电机状态2
  * @param  id 电机ID (1-32)
  * @note   读取温度、转矩电流/功率、转速、编码器位置
  */
void CAN_ReadMotorStatus2(uint8_t id) {
    uint8_t data[8] = {0x9C}; // 命令字节: 0x9C = 读取状态2
    CAN_SendFrame(id, data);
}

/**
  * @brief  读取电机状态3
  * @param  id 电机ID (1-32)
  * @note   读取温度和三相电流数据（仅MF、MH、MG电机有效）
  */
void CAN_ReadMotorStatus3(uint8_t id) {
    uint8_t data[8] = {0x9D}; // 命令字节: 0x9D = 读取状态3
    CAN_SendFrame(id, data);
}

/**
  * @brief  读取编码器数据
  * @param  id 电机ID (1-32)
  * @note   读取编码器位置、原始位置和零偏值
  */
void CAN_ReadEncoderData(uint8_t id) {
    uint8_t data[8] = {0x90}; // 命令字节: 0x90 = 读取编码器
    CAN_SendFrame(id, data);
}

/**
  * @brief  读取多圈角度
  * @param  id 电机ID (1-32)
  * @note   读取绝对角度值（0.01度/LSB）
  */
void CAN_ReadMultiTurnAngle(uint8_t id) {
    uint8_t data[8] = {0x92}; // 命令字节: 0x92 = 读取多圈角度
    CAN_SendFrame(id, data);
}

/**
  * @brief  读取单圈角度
  * @param  id 电机ID (1-32)
  * @note   读取单圈角度值（0.01度/LSB）
  */
void CAN_ReadSingleTurnAngle(uint8_t id) {
    uint8_t data[8] = {0x94}; // 命令字节: 0x94 = 读取单圈角度
    CAN_SendFrame(id, data);
}

/**
  * @brief  读取控制参数
  * @param  id 电机ID (1-32)
  * @param  index 参数索引 (参考ParamIndex枚举)
  */
void CAN_ReadControlParameter(uint8_t id, ParamIndex index) {
    uint8_t data[8] = {0xB0, (uint8_t)index}; // 命令字节: 0xB0 = 读取参数
    CAN_SendFrame(id, data);
}

/**
  * @brief  设置控制参数
  * @param  id 电机ID (1-32)
  * @param  index 参数索引 (参考ParamIndex枚举)
  * @param  param_data 参数数据 (6字节数组)
  * @note   断电后设置的参数失效
  */
void CAN_SetControlParameter(uint8_t id, ParamIndex index, uint8_t param_data[6]) {
    uint8_t data[8] = {0xB1, (uint8_t)index}; // 命令字节: 0xB1 = 设置参数
    for(int i = 0; i < 6; i++) {
        data[2+i] = param_data[i]; // 填充6字节参数数据
    }
    CAN_SendFrame(id, data);
}

/**
  * @brief  设置编码器零偏值
  * @param  id 电机ID (1-32)
  * @param  offset 编码器零偏值 (14bit:0-16383)
  */
void CAN_SetEncoderOffset(uint8_t id, uint16_t offset) {
    uint8_t data[8] = {0x91}; // 命令字节: 0x91 = 设置编码器零偏
    data[6] = offset & 0xFF;        // 零偏低字节
    data[7] = (offset >> 8) & 0xFF; // 零偏高字节
    CAN_SendFrame(id, data);
}

/**
  * @brief  保存当前位置为零点
  * @param  id 电机ID (1-32)
  * @note   需要重新上电后才能生效，频繁使用影响芯片寿命
  */
void CAN_SaveCurrentPositionAsZero(uint8_t id) {
    uint8_t data[8] = {0x19}; // 命令字节: 0x19 = 保存零点
    CAN_SendFrame(id, data);
}

/**
  * @brief  设置多圈角度到当前位置
  * @param  id 电机ID (1-32)
  * @param  angle 多圈角度值 (0.01度/LSB)
  * @note   写入RAM，断电后失效
  */
void CAN_SetMultiTurnAngle(uint8_t id, int32_t angle) {
    uint8_t data[8] = {0x95}; // 命令字节: 0x95 = 设置多圈角度
    // 将32位角度值分解为4个字节 (小端序)
    data[4] = angle & 0xFF;        // 最低字节
    data[5] = (angle >> 8) & 0xFF; // 次低字节
    data[6] = (angle >> 16) & 0xFF;// 次高字节
    data[7] = (angle >> 24) & 0xFF;// 最高字节
    CAN_SendFrame(id, data);
}
