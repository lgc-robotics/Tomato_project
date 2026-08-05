# STM32 机械臂与底盘控制固件

## 目标平台

- MCU：STM32F103C8，Cortex-M3。
- 工具链：Keil MDK。
- 外设库：STM32F10x Standard Peripheral Library 3.5.0。
- 工程文件：`PRJ/STM32_CAN_CMD.uvprojx`。

## 职责

- USART1 接收上位机 15 字节命令帧，波特率 115200。
- 解析 `0x04` 三轴坐标、`0x05` 旋转角、`0x06` 末端动作和 `0x07` 底盘误差。
- 通过 CAN 控制三轴闭环步进驱动器并识别到位帧。
- 通过 CAN 控制末端旋转关节和底盘。
- 通过 USART2/RS485 向 AT89C52 发送 `0x01/0x02`。
- 维护机械臂到位看门狗、底盘定时移动和空闲保活。

`APP/main.c` 是主入口；`BSP/board.c` 负责 CAN/USART 初始化与中断；`BSP/Timer.c` 负责看门狗、底盘定时状态和保活协调。

## 编译

1. 使用 Keil MDK 打开 `PRJ/STM32_CAN_CMD.uvprojx`。
2. 确认目标器件为 STM32F103C8。
3. Build Target。
4. 烧录并复位。

`prebuilt/Template.hex` 是整理本快照时保留的预编译文件。若源码有任何修改，应重新编译，不要继续使用旧 HEX。

## 关键说明

机械臂位置分片和同步 CAN 帧发送期间，TIM4 不直接抢占发送底盘保活，而是置待发送标志；机械臂突发结束后立即补发。不要删除这段协调逻辑，否则可能重新出现机械臂 CAN 命令偶发丢失或底盘退出自动模式的问题。
