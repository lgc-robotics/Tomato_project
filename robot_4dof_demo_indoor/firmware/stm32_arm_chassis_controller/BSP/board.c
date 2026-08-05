#include "board.h"

/**
 * @brief  NVIC中断控制器初始化
 * @param  无
 * @retval 无
 */
void nvic_init(void)
{
  // 设置优先级分组（4位抢占优先级）
  NVIC_PriorityGroupConfig(NVIC_PriorityGroup_4);

  // 初始化CAN中断（优先级0 - 最高）
  NVIC_InitTypeDef NVIC_InitStructure;

  // CAN中断配置
  NVIC_InitStructure.NVIC_IRQChannel = USB_LP_CAN1_RX0_IRQn;
  NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 0; // 高优先级
  NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0;
  NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
  NVIC_Init(&NVIC_InitStructure);

  // USART中断配置（优先级1 - 次高优先级）
  NVIC_InitStructure.NVIC_IRQChannel = USART1_IRQn;
  NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 1; // 比CAN优先级低
  NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0;
  NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
  NVIC_Init(&NVIC_InitStructure);

  // 配置USART2中断
  NVIC_InitStructure.NVIC_IRQChannel = USART2_IRQn;
  NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 2;
  NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0;
  NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
  NVIC_Init(&NVIC_InitStructure);
}

/**
 * @brief  系统时钟初始化
 * @param  无
 * @retval 无
 */
void clock_init(void)
{
  // 启用GPIO和AFIO时钟
  RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOB |RCC_APB2Periph_AFIO,ENABLE);
 
  // 启用外设时钟（同时支持CAN和USART）
  RCC_APB1PeriphClockCmd(RCC_APB1Periph_CAN1, ENABLE);

  // 禁用JTAG释放PB3/PB4引脚
  GPIO_PinRemapConfig(GPIO_Remap_SWJ_JTAGDisable, ENABLE);
}


/**
  * @brief  CAN 初始化，配置两个 FIFO 的过滤规则
  * @note   标准帧 ID 0x502 存入 FIFO1（无中断）
  *         扩展帧中 addr (ID[15:8]) 为 0x01/0x02/0x03/0x04 的存入 FIFO0（产生中断）
  */
void can_init(void)
{
    // ---------- GPIO 初始化 ----------
    GPIO_InitTypeDef GPIO_InitStructure;
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_12;
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
    GPIO_Init(GPIOA, &GPIO_InitStructure);

    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_11;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IPU;
    GPIO_Init(GPIOA, &GPIO_InitStructure);

    // ---------- CAN 基本参数 ----------
    CAN_InitTypeDef CAN_InitStructure;
    CAN_StructInit(&CAN_InitStructure);
    CAN_InitStructure.CAN_Mode = CAN_Mode_Normal;
    CAN_InitStructure.CAN_SJW = CAN_SJW_1tq;
    CAN_InitStructure.CAN_BS1 = CAN_BS1_9tq;
    CAN_InitStructure.CAN_BS2 = CAN_BS2_2tq;
    CAN_InitStructure.CAN_Prescaler = 6;   // 500kbps
    CAN_Init(CAN1, &CAN_InitStructure);

    // ---------- 过滤器配置 ----------
    CAN_FilterInitTypeDef CAN_FilterInitStructure;
    uint32_t filter_id, mask;

    // ========== 1. 标准帧 0x502 → FIFO1 ==========
    CAN_FilterInitStructure.CAN_FilterNumber = 0;           // 使用过滤器组0
    CAN_FilterInitStructure.CAN_FilterMode = CAN_FilterMode_IdMask;
    CAN_FilterInitStructure.CAN_FilterScale = CAN_FilterScale_32bit;

    // 构造32位过滤ID（标准帧格式）
    // 标准ID左移21位（对齐到[31:21]），IDE=0（标准帧），RTR=0（数据帧）
	filter_id = ((uint32_t)0x502 << 21) | (0 << 2) | 0;
	mask = ((uint32_t)0x7FF << 21) | ((uint32_t)1 << 2);

    CAN_FilterInitStructure.CAN_FilterIdHigh = (uint16_t)(filter_id >> 16);
    CAN_FilterInitStructure.CAN_FilterIdLow  = (uint16_t)(filter_id & 0xFFFF);
    CAN_FilterInitStructure.CAN_FilterMaskIdHigh = (uint16_t)(mask >> 16);
    CAN_FilterInitStructure.CAN_FilterMaskIdLow  = (uint16_t)(mask & 0xFFFF);
    CAN_FilterInitStructure.CAN_FilterFIFOAssignment = CAN_FIFO1;
    CAN_FilterInitStructure.CAN_FilterActivation = ENABLE;
    CAN_FilterInit(&CAN_FilterInitStructure);

    // ========== 2. 扩展帧 addr=0x01~0x04 → FIFO0 ==========
    // 需要4个过滤器组，分别匹配每个addr值（因为掩码模式无法同时匹配多个不连续的addr）
    uint8_t addr_list[4] = {0x01, 0x02, 0x03, 0x04};
    for (int i = 0; i < 4; i++)
    {
        CAN_FilterInitStructure.CAN_FilterNumber = i + 1;   // 使用组1,2,3,4
        CAN_FilterInitStructure.CAN_FilterMode = CAN_FilterMode_IdMask;
        CAN_FilterInitStructure.CAN_FilterScale = CAN_FilterScale_32bit;

        // 构造32位过滤ID（扩展帧格式）
        // 扩展ID[28:0]左移3位（对齐到[31:3]），IDE=1（扩展帧），RTR=0
        // 我们的扩展ID格式：高8位(comm_type) | 中间8位(addr) | 低8位(其他)
        // 要匹配 addr 字段，只需要关心 bit15~8（即扩展ID的第15-8位，对应于32位寄存器中的位置？）
        // 扩展ID在32位寄存器中占据 [31:3]，所以原来的 bit15~8 会移动到 (15+3)=18 到 (8+3)=11 位。
        // 更简单的方法：直接构造一个只包含addr的扩展ID基准值，掩码只覆盖addr所在的位段。
        // 假设我们只关心 addr 字段（bit15-8），其他位一律视为无关。
        uint32_t expected_id = ((uint32_t)addr_list[i] << 8);   // 原扩展ID中 addr 在 bit15-8
        // 将这个期望值左移3位，得到寄存器中的对齐值
		filter_id = (expected_id << 3) | ((uint32_t)1 << 2);
		mask = ((uint32_t)0xFF << 11) | ((uint32_t)1 << 2);

        CAN_FilterInitStructure.CAN_FilterIdHigh = (uint16_t)(filter_id >> 16);
        CAN_FilterInitStructure.CAN_FilterIdLow  = (uint16_t)(filter_id & 0xFFFF);
        CAN_FilterInitStructure.CAN_FilterMaskIdHigh = (uint16_t)(mask >> 16);
        CAN_FilterInitStructure.CAN_FilterMaskIdLow  = (uint16_t)(mask & 0xFFFF);
        CAN_FilterInitStructure.CAN_FilterFIFOAssignment = CAN_FIFO0;
        CAN_FilterInitStructure.CAN_FilterActivation = ENABLE;
        CAN_FilterInit(&CAN_FilterInitStructure);
    }

    // 启用 FIFO0 接收中断（FIFO1 无中断）
    CAN_ITConfig(CAN1, CAN_IT_FMP0, ENABLE);
    // FIFO1 的中断不使能
}

/**
  * @brief  初始化USART2串口
  * @param  无
  * @retval None
  */
void usart2_init(void)
{
    GPIO_InitTypeDef GPIO_InitStructure;
    USART_InitTypeDef USART_InitStructure;
    
    // 1. 使能时钟
    RCC_APB1PeriphClockCmd(RCC_APB1Periph_USART2, ENABLE);
    RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA | RCC_APB2Periph_GPIOD | 
                           RCC_APB2Periph_AFIO, ENABLE);
    
    // 2. 配置USART2 TX引脚(PA2) - 复用推挽输出
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_2;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_Init(GPIOA, &GPIO_InitStructure);
    
    // 3. 配置USART2 RX引脚(PA3) - 浮空输入
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_3;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IN_FLOATING;
    GPIO_Init(GPIOA, &GPIO_InitStructure);
    
    // 4. 配置RS485收发控制引脚(PD7)
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_7;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_Init(GPIOD, &GPIO_InitStructure);
    GPIO_ResetBits(GPIOD, GPIO_Pin_7);  // 默认接收模式
    
    // 5. 配置USART2
    USART_InitStructure.USART_BaudRate = 9600;
    USART_InitStructure.USART_WordLength = USART_WordLength_8b;
    USART_InitStructure.USART_StopBits = USART_StopBits_1;
    USART_InitStructure.USART_Parity = USART_Parity_No;
    USART_InitStructure.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
    USART_InitStructure.USART_Mode = USART_Mode_Tx | USART_Mode_Rx;
    USART_Init(USART2, &USART_InitStructure);
    
    // 6. 使能接收中断
    USART_ITConfig(USART2, USART_IT_RXNE, ENABLE);
    
    // 8. 使能USART2
    USART_Cmd(USART2, ENABLE);  

	NVIC_EnableIRQ(USART2_IRQn);
}

/**
 * @brief  开发板全局初始化
 * @param  无
 * @retval 无
 */
void board_init(void)
{
  nvic_init();   // 中断配置（包含优先级）
  clock_init();  // 时钟配置
  can_init();    // CAN初始化
  usart2_init(); // USART2初始化
}
