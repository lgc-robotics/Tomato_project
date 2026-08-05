#include "Emm_V5.h"

/**
 * @brief    将当前位置重置为零点
 * @param    addr  驱动器地址 (设备ID)
 */
void Emm_V5_Reset_CurPos_To_Zero(uint8_t addr)
{
  uint8_t cmd[16] = {0};

  // 构建CAN命令
  cmd[0] = addr; // 设备地址
  cmd[1] = 0x0A; // 功能码：复位当前位置
  cmd[2] = 0x6D; // 操作码
  cmd[3] = 0x6B; // 固定校验字节

  can_SendCmd(cmd, 4); // 发送CAN命令（4字节）
}

/**
 * @brief    重置堵转保护计数器
 * @param    addr  驱动器地址 (设备ID)
 */
void Emm_V5_Reset_Clog_Pro(uint8_t addr)
{
  uint8_t cmd[16] = {0};

  cmd[0] = addr; // 设备地址
  cmd[1] = 0x0E; // 功能码：重置堵转保护
  cmd[2] = 0x52; // 操作码
  cmd[3] = 0x6B; // 固定校验字节

  can_SendCmd(cmd, 4);
}

/**
 * @brief    读取系统参数
 * @param    addr  驱动器地址
 * @param    s     要读取的系统参数类型 (枚举值)
 */
void Emm_V5_Read_Sys_Params(uint8_t addr, SysParams_t s)
{
  uint8_t i = 0;
  uint8_t cmd[16] = {0};

  cmd[i] = addr;
  ++i; // 设备地址

  // 根据参数类型设置功能码
  switch (s)
  {
  case S_VER:
    cmd[i] = 0x1F;
    ++i;
    break; // 固件版本
  case S_RL:
    cmd[i] = 0x20;
    ++i;
    break; // 电机额定参数
  case S_PID:
    cmd[i] = 0x21;
    ++i;
    break; // PID参数
  case S_VBUS:
    cmd[i] = 0x24;
    ++i;
    break; // 母线电压
  case S_CPHA:
    cmd[i] = 0x27;
    ++i;
    break; // 相电流
  case S_ENCL:
    cmd[i] = 0x31;
    ++i;
    break; // 编码器位置
  case S_TPOS:
    cmd[i] = 0x33;
    ++i;
    break; // 目标位置
  case S_VEL:
    cmd[i] = 0x35;
    ++i;
    break; // 当前速度
  case S_CPOS:
    cmd[i] = 0x36;
    ++i;
    break; // 当前位置
  case S_PERR:
    cmd[i] = 0x37;
    ++i;
    break; // 位置误差
  case S_FLAG:
    cmd[i] = 0x3A;
    ++i;
    break; // 状态标志
  case S_ORG:
    cmd[i] = 0x3B;
    ++i;
    break; // 原点状态
  case S_Conf:
    cmd[i] = 0x42;
    ++i;
    cmd[i] = 0x6C;
    ++i;
    break; // 配置参数
  case S_State:
    cmd[i] = 0x43;
    ++i;
    cmd[i] = 0x7A;
    ++i;
    break; // 系统状态
  default:
    break;
  }

  cmd[i] = 0x6B;
  ++i;                 // 固定校验字节
  can_SendCmd(cmd, i); // 发送命令
}

/**
 * @brief    修改控制模式
 * @param    addr       驱动器地址
 * @param    svF        是否保存到Flash：true保存，false不保存
 * @param    ctrl_mode  控制模式：
 *                      0 = 关闭控制（自由状态）
 *                      1 = 开环模式
 *                      2 = 闭环模式
 *                      3 = 特殊模式（En引脚作为电机使能，Dir引脚作为原点信号）
 */
void Emm_V5_Modify_Ctrl_Mode(uint8_t addr, bool svF, uint8_t ctrl_mode)
{
  uint8_t cmd[16] = {0};

  cmd[0] = addr;      // 设备地址
  cmd[1] = 0x46;      // 功能码：修改控制模式
  cmd[2] = 0x69;      // 操作码
  cmd[3] = svF;       // 保存标志
  cmd[4] = ctrl_mode; // 控制模式
  cmd[5] = 0x6B;      // 固定校验字节

  can_SendCmd(cmd, 6);
}

/**
 * @brief    使能控制
 * @param    addr   驱动器地址
 * @param    state  使能状态：true使能，false失能
 * @param    snF    同步运动标志：true同步执行，false立即执行
 */
void Emm_V5_En_Control(uint8_t addr, bool state, bool snF)
{
  uint8_t cmd[16] = {0};

  cmd[0] = addr;           // 设备地址
  cmd[1] = 0xF3;           // 功能码：使能控制
  cmd[2] = 0xAB;           // 操作码
  cmd[3] = (uint8_t)state; // 使能状态
  cmd[4] = snF;            // 同步标志
  cmd[5] = 0x6B;           // 固定校验字节

  can_SendCmd(cmd, 6);
}

/**
 * @brief    速度模式控制
 * @param    addr  驱动器地址
 * @param    dir   旋转方向：0=CW(顺时针)，非0=CCW(逆时针)
 * @param    vel   目标速度：范围0-5000 RPM
 * @param    acc   加速度：范围0-255（0表示瞬时加速）
 * @param    snF   同步运动标志：true同步执行，false立即执行
 */
void Emm_V5_Vel_Control(uint8_t addr, uint8_t dir, uint16_t vel, uint8_t acc, bool snF)
{
  uint8_t cmd[16] = {0};

  cmd[0] = addr;                // 设备地址
  cmd[1] = 0xF6;                // 功能码：速度控制
  cmd[2] = dir;                 // 方向
  cmd[3] = (uint8_t)(vel >> 8); // 速度高字节
  cmd[4] = (uint8_t)(vel >> 0); // 速度低字节
  cmd[5] = acc;                 // 加速度
  cmd[6] = snF;                 // 同步标志
  cmd[7] = 0x6B;                // 固定校验字节

  can_SendCmd(cmd, 8);
}

/**
 * @brief    位置模式控制
 * @param    addr  驱动器地址
 * @param    dir   旋转方向：0=CW(顺时针)，非0=CCW(逆时针)
 * @param    vel   运行速度：范围0-5000 RPM
 * @param    acc   加速度：范围0-255（0表示瞬时加速）
 * @param    clk   脉冲数量：0 - (2^32 - 1)
 * @param    raF   运动模式：
 *                 false = 相对运动（基于当前位置）
 *                 true = 绝对运动（基于零点位置）
 * @param    snF   同步运动标志：true同步执行，false立即执行
 */
void Emm_V5_Pos_Control(uint8_t addr, uint8_t dir, uint16_t vel, uint8_t acc, uint32_t clk, bool raF, bool snF)
{
  uint8_t cmd[16] = {0};

  cmd[0] = addr;                 // 设备地址
  cmd[1] = 0xFD;                 // 功能码：位置控制
  cmd[2] = dir;                  // 方向
  cmd[3] = (uint8_t)(vel >> 8);  // 速度高字节
  cmd[4] = (uint8_t)(vel >> 0);  // 速度低字节
  cmd[5] = acc;                  // 加速度
  cmd[6] = (uint8_t)(clk >> 24); // 脉冲数最高字节
  cmd[7] = (uint8_t)(clk >> 16); // 脉冲数次高字节
  cmd[8] = (uint8_t)(clk >> 8);  // 脉冲数次低字节
  cmd[9] = (uint8_t)(clk >> 0);  // 脉冲数最低字节
  cmd[10] = raF;                 // 运动模式标志
  cmd[11] = snF;                 // 同步标志
  cmd[12] = 0x6B;                // 固定校验字节

  can_SendCmd(cmd, 13);
}

/**
 * @brief    立即停止运动
 * @param    addr  驱动器地址
 * @param    snF   同步运动标志：true同步停止，false立即停止
 */
void Emm_V5_Stop_Now(uint8_t addr, bool snF)
{
  uint8_t cmd[16] = {0};

  cmd[0] = addr; // 设备地址
  cmd[1] = 0xFE; // 功能码：立即停止
  cmd[2] = 0x98; // 操作码
  cmd[3] = snF;  // 同步标志
  cmd[4] = 0x6B; // 固定校验字节

  can_SendCmd(cmd, 5);
}

/**
 * @brief    触发同步运动（执行所有挂起的同步命令）
 * @param    addr  驱动器地址（0x00表示广播地址）
 */
void Emm_V5_Synchronous_motion(uint8_t addr)
{
  uint8_t cmd[16] = {0};

  cmd[0] = addr; // 设备地址（0=广播）
  cmd[1] = 0xFF; // 功能码：同步执行
  cmd[2] = 0x66; // 操作码
  cmd[3] = 0x6B; // 固定校验字节

  can_SendCmd(cmd, 4);
}

/**
 * @brief    将当前位置设置为原点
 * @param    addr  驱动器地址
 * @param    svF   是否保存到Flash：true保存，false不保存
 */
void Emm_V5_Origin_Set_O(uint8_t addr, bool svF)
{
  uint8_t cmd[16] = {0};

  cmd[0] = addr; // 设备地址
  cmd[1] = 0x93; // 功能码：设置原点
  cmd[2] = 0x88; // 操作码
  cmd[3] = svF;  // 保存标志
  cmd[4] = 0x6B; // 固定校验字节

  can_SendCmd(cmd, 5);
}

/**
 * @brief    修改原点搜索参数
 * @param    addr    驱动器地址
 * @param    svF     是否保存到Flash
 * @param    o_mode  原点模式：
 *                   0 = 限位开关触发停止
 *                   1 = 编码器Z脉冲停止
 *                   2 = 限位开关碰撞停止
 *                   3 = 限位开关返回停止
 * @param    o_dir   搜索方向：0=CW，非0=CCW
 * @param    o_vel   搜索速度 (RPM)
 * @param    o_tm    超时时间 (毫秒)
 * @param    sl_vel  碰撞检测速度 (RPM)
 * @param    sl_ma   碰撞检测电流 (mA)
 * @param    sl_ms   碰撞检测时间 (毫秒)
 * @param    potF    上电自动回原点使能：true启用，false禁用
 */
void Emm_V5_Origin_Modify_Params(uint8_t addr, bool svF, uint8_t o_mode, uint8_t o_dir, uint16_t o_vel, uint32_t o_tm, uint16_t sl_vel, uint16_t sl_ma, uint16_t sl_ms, bool potF)
{
  uint8_t cmd[32] = {0};

  cmd[0] = addr;                    // 设备地址
  cmd[1] = 0x4C;                    // 功能码：修改原点参数
  cmd[2] = 0xAE;                    // 操作码
  cmd[3] = svF;                     // 保存标志
  cmd[4] = o_mode;                  // 原点模式
  cmd[5] = o_dir;                   // 搜索方向
  cmd[6] = (uint8_t)(o_vel >> 8);   // 搜索速度高字节
  cmd[7] = (uint8_t)(o_vel >> 0);   // 搜索速度低字节
  cmd[8] = (uint8_t)(o_tm >> 24);   // 超时时间最高字节
  cmd[9] = (uint8_t)(o_tm >> 16);   // 超时时间次高字节
  cmd[10] = (uint8_t)(o_tm >> 8);   // 超时时间次低字节
  cmd[11] = (uint8_t)(o_tm >> 0);   // 超时时间最低字节
  cmd[12] = (uint8_t)(sl_vel >> 8); // 碰撞速度高字节
  cmd[13] = (uint8_t)(sl_vel >> 0); // 碰撞速度低字节
  cmd[14] = (uint8_t)(sl_ma >> 8);  // 碰撞电流高字节
  cmd[15] = (uint8_t)(sl_ma >> 0);  // 碰撞电流低字节
  cmd[16] = (uint8_t)(sl_ms >> 8);  // 碰撞时间高字节
  cmd[17] = (uint8_t)(sl_ms >> 0);  // 碰撞时间低字节
  cmd[18] = potF;                   // 上电回零使能
  cmd[19] = 0x6B;                   // 固定校验字节

  can_SendCmd(cmd, 20);
}

/**
 * @brief    触发原点搜索
 * @param    addr    驱动器地址
 * @param    o_mode  原点模式（同Origin_Modify_Params）
 * @param    snF     同步运动标志：true同步执行，false立即执行
 */
void Emm_V5_Origin_Trigger_Return(uint8_t addr, uint8_t o_mode, bool snF)
{
  uint8_t cmd[16] = {0};

  cmd[0] = addr;   // 设备地址
  cmd[1] = 0x9A;   // 功能码：触发原点搜索
  cmd[2] = o_mode; // 原点模式
  cmd[3] = snF;    // 同步标志
  cmd[4] = 0x6B;   // 固定校验字节

  can_SendCmd(cmd, 5);
}

/**
 * @brief    中断原点搜索过程
 * @param    addr  驱动器地址
 */
void Emm_V5_Origin_Interrupt(uint8_t addr)
{
  uint8_t cmd[16] = {0};

  cmd[0] = addr; // 设备地址
  cmd[1] = 0x9C; // 功能码：中断原点搜索
  cmd[2] = 0x48; // 操作码
  cmd[3] = 0x6B; // 固定校验字节

  can_SendCmd(cmd, 4);
}
