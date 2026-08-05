#ifndef __EMM_V5_H
#define __EMM_V5_H

#include "can.h" // 包含CAN通信库

// 绝对值计算宏
#define ABS(x) ((x) > 0 ? (x) : -(x))

/**
 * @brief 系统参数读取枚举
 * @note 用于选择要读取的系统参数
 */
typedef enum
{
	S_VER = 0,	  // 读取固件版本和对应硬件版本
	S_RL = 1,	  // 读取电机额定参数（额定电流、电阻等）
	S_PID = 2,	  // 读取PID控制参数
	S_VBUS = 3,	  // 读取母线电压
	S_CPHA = 5,	  // 读取相电流
	S_ENCL = 7,	  // 读取编码器值（未校准时的原始值）
	S_TPOS = 8,	  // 读取目标位置角度
	S_VEL = 9,	  // 读取实时转速
	S_CPOS = 10,  // 读取实时位置角度
	S_PERR = 11,  // 读取位置跟踪误差角度
	S_FLAG = 13,  // 读取使能/原点/报警状态标志位
	S_Conf = 14,  // 读取配置参数
	S_State = 15, // 读取系统状态信息
	S_ORG = 16,	  // 读取原点状态/回零失败标志位
} SysParams_t;

/**
 * @brief 将当前位置重置为零点
 * @param addr 驱动器地址 (设备ID)
 */
void Emm_V5_Reset_CurPos_To_Zero(uint8_t addr);

/**
 * @brief 重置堵转保护计数器
 * @param addr 驱动器地址
 */
void Emm_V5_Reset_Clog_Pro(uint8_t addr);

/**
 * @brief 读取系统参数
 * @param addr 驱动器地址
 * @param s 要读取的系统参数类型 (枚举值)
 */
void Emm_V5_Read_Sys_Params(uint8_t addr, SysParams_t s);

/**
 * @brief 修改控制模式
 * @param addr 驱动器地址
 * @param svF 是否保存到Flash：true保存，false不保存
 * @param ctrl_mode 控制模式：
 *        0 = 关闭控制（自由状态）
 *        1 = 开环模式
 *        2 = 闭环模式
 *        3 = 特殊模式（En引脚作为电机使能，Dir引脚作为原点信号）
 */
void Emm_V5_Modify_Ctrl_Mode(uint8_t addr, bool svF, uint8_t ctrl_mode);

/**
 * @brief 电机使能控制
 * @param addr 驱动器地址
 * @param state 使能状态：true使能，false失能
 * @param snF 同步运动标志：true同步执行，false立即执行
 */
void Emm_V5_En_Control(uint8_t addr, bool state, bool snF);

/**
 * @brief 速度模式控制
 * @param addr 驱动器地址
 * @param dir 旋转方向：0=CW(顺时针)，非0=CCW(逆时针)
 * @param vel 目标速度：范围0-5000 RPM
 * @param acc 加速度：范围0-255（0表示瞬时加速）
 * @param snF 同步运动标志：true同步执行，false立即执行
 */
void Emm_V5_Vel_Control(uint8_t addr, uint8_t dir, uint16_t vel, uint8_t acc, bool snF);

/**
 * @brief 位置模式控制（核心运动函数）
 * @param addr 驱动器地址
 * @param dir 旋转方向：0=CW(顺时针)，非0=CCW(逆时针)
 * @param vel 运行速度：范围0-5000 RPM
 * @param acc 加速度：范围0-255（0表示瞬时加速）
 * @param clk 脉冲数量：0 - (2^32 - 1) 决定电机旋转角度
 * @param raF 运动模式：
 *        false = 相对运动（基于当前位置）
 *        true = 绝对运动（基于零点位置）
 * @param snF 同步运动标志：true同步执行，false立即执行
 */
void Emm_V5_Pos_Control(uint8_t addr, uint8_t dir, uint16_t vel, uint8_t acc, uint32_t clk, bool raF, bool snF);

/**
 * @brief 立即停止运动
 * @param addr 驱动器地址
 * @param snF 同步运动标志：true同步停止，false立即停止
 */
void Emm_V5_Stop_Now(uint8_t addr, bool snF);

/**
 * @brief 触发同步运动（执行所有挂起的同步命令）
 * @param addr 驱动器地址（0x00表示广播地址）
 */
void Emm_V5_Synchronous_motion(uint8_t addr);

/**
 * @brief 将当前位置设置为原点
 * @param addr 驱动器地址
 * @param svF 是否保存到Flash：true保存，false不保存
 */
void Emm_V5_Origin_Set_O(uint8_t addr, bool svF);

/**
 * @brief 修改原点搜索参数
 * @param addr 驱动器地址
 * @param svF 是否保存到Flash
 * @param o_mode 原点模式：
 *        0 = 限位开关触发停止
 *        1 = 编码器Z脉冲停止
 *        2 = 限位开关碰撞停止
 *        3 = 限位开关返回停止
 * @param o_dir 搜索方向：0=CW，非0=CCW
 * @param o_vel 搜索速度 (RPM)
 * @param o_tm 超时时间 (毫秒)
 * @param sl_vel 碰撞检测速度 (RPM)
 * @param sl_ma 碰撞检测电流 (mA)
 * @param sl_ms 碰撞检测时间 (毫秒)
 * @param potF 上电自动回原点使能：true启用，false禁用
 */
void Emm_V5_Origin_Modify_Params(uint8_t addr, bool svF, uint8_t o_mode, uint8_t o_dir, uint16_t o_vel, uint32_t o_tm, uint16_t sl_vel, uint16_t sl_ma, uint16_t sl_ms, bool potF);

/**
 * @brief 触发原点搜索
 * @param addr 驱动器地址
 * @param o_mode 原点模式（同Origin_Modify_Params）
 * @param snF 同步运动标志：true同步执行，false立即执行
 */
void Emm_V5_Origin_Trigger_Return(uint8_t addr, uint8_t o_mode, bool snF);

/**
 * @brief 中断原点搜索过程
 * @param addr 驱动器地址
 */
void Emm_V5_Origin_Interrupt(uint8_t addr);

#endif
