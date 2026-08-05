#include <REGX52.H>

/**
  * @brief  定时器0初始化@11.0952.000MHz
  * @param  无
  * @retval 无
  */
void InitTimer0(void)
{
    TMOD &= 0xF0;       // 清除T0的控制位
    TMOD |= 0x01;       // 设置定时器模式1（16位定时器）
    
    // 设置定时初值为0x1389（5001），对应约65.536ms
    TL0 = 0x89;         // 低8位 = 0x89
    TH0 = 0x13;         // 高8位 = 0x13
    
    TF0 = 0;            // 清除TF0标志
    TR0 = 1;            // 定时器0开始计时
    ET0 = 1;            // 开定时器0中断
}
