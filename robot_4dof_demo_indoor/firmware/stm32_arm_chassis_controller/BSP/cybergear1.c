#include "stm32f10x.h"                  // Device header
#include "cybergear1.h"
#include "Miparam.h"
#include "Mycan.h"
#include <string.h>
#include "Usart1.h"
#include "delay.h"
#include "stm32f10x_usart.h"
#include <stdio.h>
#include <stdlib.h>

//小米电机函数库
MI_Motor cyber;
//分别定义一个CAN发送和接收的结构体
CanRxMsg rxMsg;//接收结构体
CanTxMsg txMsg;//发送结构体
uint8_t rx_data[8];       //接收数据存放的位置
uint32_t Motor_Can_ID;    //接收报文的电机ID
uint8_t byte[4];          //转换临时数据
uint32_t send_mail_box = {0};//NONE，发送邮箱？

extern volatile uint16_t raw_angle;

/*  **************************************************************/
/** @brief          小米电机初始化
  * @param[in]      Motor:  电机结构体
  * @param[in]      Can_Id: 小米电机ID(默认0x7F)
  * @param[in]      mode: 电机工作模式（0.运动模式Motion_mode  1. 位置模式Position_mode 
  *                                   2. 速度模式Speed_mode  3. 电流模式Current_mode）
  * @retval         none
  *  *************************************************************/
void init_cybergear(MI_Motor *Motor, uint8_t Can_Id, uint8_t mode)
{
    txMsg.StdId = 0;            //配置CAN发送：标准帧清零 
    txMsg.ExtId = 0;            //配置CAN发送：扩展帧清零  	
    txMsg.IDE = CAN_ID_EXT;     //配置CAN发送：扩展帧
    txMsg.RTR = CAN_RTR_DATA;   //配置CAN发送：数据帧
    txMsg.DLC = 0x08;           //配置CAN发送：数据长度为8个字节
	Motor->CAN_ID=Can_Id;       //ID设置 
	Motor->mode=mode;           //模式设置
	Set_Mode_Cybergear(Motor,mode);//设置电机的运动模式
	start_cybergear(Motor->CAN_ID);//使能小米电机，通信类型为3
	set_zeropos_cybergear(Motor);//设置电机零点,通信类型为6
}
	
/*  **************************************************************/
/** @brief          小米电机回文16位数据转浮点
  * @param[in]      x:16位回文            x_min:对应参数下限     
                    x_max:对应参数上限    bits:参数位数
  * @retval         返回浮点值
  **  *************************************************************/
float uint16_to_float(uint16_t x,float x_min,float x_max,int bits)
{
    uint32_t span = (1 << bits) - 1;//二进制数的范围2^bits-1
    float offset = x_max - x_min;//实际范围
	float s;//转换后的浮点数
	s=offset*(x/span)+x_min;//实现映射变换
	return s;//转换为浮点数
}
/*****************************************************************/
/** @brief         小米电机发送浮点转16位数据，每个数据都是2个字节也就是16位二进制数
  * @param[in]      x:输入的参数数据，浮点
  * @param[in]      x_min:对应参数下限
  * @param[in]      x_max:对应参数上限
  * @param[in]      bits:参数位数
  * @retval         返回浮点值
  * static 表示其作用域仅限于这个文件
  ***************************************************************/
int float_to_uint(float x, float x_min, float x_max, int bits)
{
	float span = x_max - x_min;//浮点数的范围
	float offset = x_min; 
	//限制X的范围
	if(x > x_max) x=x_max;
	else if(x < x_min) x= x_min;
	return (int) ((x-offset)*((float)((1<<bits)-1))/span);//（x-下限）/（（2^bits-1）/范围）
}
 /****************************************************************/
/** @brief          小米电机ID检查，通信类型为0
  * @param[in]      id:  控制电机CAN_ID【出厂默认0x7F】
  ******************************************************************/
uint32_t check_cybergear(uint8_t ID)
{
    uint8_t tx_data[8] = {0};//没有数据
    txMsg.ExtId = Communication_Type_GetID<<24|Master_CAN_ID<<8|ID;//扩展ID的组合，依旧是3个部分
	//txMsg.ExtId = 0x00<<24|Master_CAN_ID<<8|ID;//通讯类型为0x00
	MyCAN_Transmit(txMsg.ExtId,txMsg.DLC,tx_data);//写入指令
	return txMsg.ExtId;
}
/******************************************************************/
/** @brief          小米运控模式指令，通信类型：1
  * @param[in]      Motor:  目标电机结构体
  * @param[in]      torque: 力矩设置[-12,12] N*M
  * @param[in]      MechPosition: 位置设置[-12.5,12.5] rad
  * @param[in]      speed: 速度设置[-30,30] rpm
  * @param[in]      kp: 比例参数设置
  * @param[in]      kd: 微分参数设置
  * @retval         none
  ******************************************************************/
void motor_controlmode(MI_Motor *Motor,float torque, float MechPosition, float speed, float kp, float kd)
{   
    uint8_t tx_data[8]={0};//发送数据初始化
	//如果不是运控模式就改成运控模式
	if (Motor->mode != 0x00)
	{
	  uint8_t mode=Motion_mode;//控制模式为运控控制
      Motor->mode=mode;
	  Set_Mode_Cybergear(Motor,mode);//设置电机的运动模式
	  start_cybergear(Motor->CAN_ID);//使能小米电机，通信类型为3
	}
    //装填发送数据
    tx_data[0]=float_to_uint(MechPosition,P_MIN,P_MAX,16)>>8;  //目标角度：取得是高位的结果，即0x12
    tx_data[1]=float_to_uint(MechPosition,P_MIN,P_MAX,16);  //目标角度：取得是低位的结果，即0x34
    tx_data[2]=float_to_uint(speed,V_MIN,V_MAX,16)>>8;//目标速度 ，高位 
    tx_data[3]=float_to_uint(speed,V_MIN,V_MAX,16);  //目标速度，低位
    tx_data[4]=float_to_uint(kp,KP_MIN,KP_MAX,16)>>8;  	  //目标KP
    tx_data[5]=float_to_uint(kp,KP_MIN,KP_MAX,16);  	  //目标KP
    tx_data[6]=float_to_uint(kd,KD_MIN,KD_MAX,16)>>8;  	  //目标KD
    tx_data[7]=float_to_uint(kd,KD_MIN,KD_MAX,16); 	  //目标KD
    txMsg.ExtId = Communication_Type_MotionControl<<24|float_to_uint(torque,T_MIN,T_MAX,16)<<8|Motor->CAN_ID;//装填扩展帧数据	
	MyCAN_Transmit(txMsg.ExtId,txMsg.DLC,tx_data);//写入指令	
}
/******************************************************************/
/** @brief         电机反馈数据，通信类型2
  * @param[in]     信息存放的地址
  * @retval         none
  *****************************************************************/
void Rx_Fifo0_Msg(MI_Motor *Motor)
{
    uint8_t tx_data[8]={0};

    txMsg.ExtId =
        Communication_Type_MotorRequest<<24 |
        Master_CAN_ID<<8 |
        Motor->CAN_ID;

    MyCAN_Transmit(
        txMsg.ExtId,
        txMsg.DLC,
        tx_data);
}
//void Rx_Fifo0_Msg(MI_Motor *Motor)
//{
//	//写入电机信息反馈通讯指令
//	uint8_t tx_data[8]={0};//发送数据初始化
//	txMsg.ExtId = Communication_Type_MotorRequest<<24|Master_CAN_ID<<8|Motor->CAN_ID;//装填扩展帧数据
//	MyCAN_Transmit(txMsg.ExtId,txMsg.DLC,tx_data);//写入指令，以获取位置反馈
//	if (MyCAN_ReceiveFlag())//判断是否接收到报文信息
//	{
//		uint32_t RxID;
//		uint8_t RxLength;
//		uint8_t RxData[8];
//		//接收信息放到对应的接收报文之中
//		MyCAN_Receive(&RxID, &RxLength, RxData);
//		
//		//显示接收的数据	
////		OLED_ShowString(1, 1, "ID:");//第3行第1列显示
////		OLED_ShowHexNum(2, 1, RxID, 8);//显示ID
////		OLED_ShowString(3, 1, "DATA");//显示接收到的数组长度
////		for(int i=0;i<8;i++){
////			OLED_ShowHexNum(4, 2*i+1, RxData[i], 2);//显示数据
////		}
//		
//		//数据转化，如果通讯类型是02，则将数据进行转换
//		if(RxID>>24==2)
//		{
//			raw_angle = (RxData[0] << 8) | RxData[1];
//			Motor->CAN_ID=(RxID&0xFFFF)>>8;//获取接收数据的ID：保留低16位，其余全变成零，再右移8位，则获得了bit8~bit15的canid
//			Motor->Angle=uint16_to_float(RxData[0]<<8|RxData[1],P_MIN,P_MAX,16);//将字节0~1转化位浮点数，即为当前角度
//			Motor->Speed=uint16_to_float(RxData[2]<<8|RxData[3],V_MIN,V_MAX,16);//将字节2~3转化位浮点数，即为当前速度			
//			Motor->Torque=uint16_to_float(RxData[4]<<8|RxData[5],T_MIN,T_MAX,16);//将字节4~5转化位浮点数，即为当前角度				
//			Motor->Temp=(RxData[6]<<8|RxData[7])*Temp_Gain;//将字节4~5转化为当前温度	
//			Motor->error_code=(RxID&0x1F0000)>>16;	
//		}
//		else if(RxID>>24==0)
//		{
//			Motor->CAN_ID=RxID>>8;//获取接收数据的ID：保留低16位，其余全变成零，再右移8位，则获得了bit8~bit15的canid
//		}
//		USART_SendByte(0x05);
//	}
//}
/**********************************************************************/
/** @brief          使能小米电机，通信类型为3
  * @param[in]      Motor:对应控制电机结构体   
  * @retval         none
  *********************************************************************/
void start_cybergear(uint8_t can_id)
{
    uint8_t tx_data[8] = {0}; 
    txMsg.ExtId = Communication_Type_MotorEnable<<24|Master_CAN_ID<<8|can_id;
    MyCAN_Transmit(txMsg.ExtId,txMsg.DLC,tx_data);//写入指令	
}
/************************************************************************/
/** @brief          停止电机，通信类型为4
  * @param[in]      Motor:对应控制电机结构体   
  * @param[in]      clear_error:清除错误位（0 不清除 1清除）
  * @retval         None
  ************************************************************************/
void stop_cybergear(MI_Motor *Motor,uint8_t clear_error)
{
	uint8_t tx_data[8]={0};
	tx_data[0]=clear_error;//清除错误位设置
	txMsg.ExtId = Communication_Type_MotorStop<<24|Master_CAN_ID<<8|Motor->CAN_ID;
  	MyCAN_Transmit(txMsg.ExtId,txMsg.DLC,tx_data);//写入指令	
}
/************************************************************************/
/** @brief          设置电机零点,通信类型为6
  * @param[in]      Motor:  电机结构体
  * @retval         none
  ************************************************************************/
void set_zeropos_cybergear(MI_Motor *Motor)
{
	uint8_t tx_data[8]={0};
	tx_data[0] = 1;//数据区为1
	txMsg.ExtId = Communication_Type_SetPosZero<<24|Master_CAN_ID<<8|Motor->CAN_ID;//扩展帧格式
	MyCAN_Transmit(txMsg.ExtId,txMsg.DLC,tx_data);//写入指令	
}
/************************************************************************/
/** @brief          设置电机CANID，通信类型为7
  * @param[in]      Motor:  电机结构体
  * @param[in]      Motor:  设置新ID
  * @retval         none
  */
void set_CANID_cybergear(MI_Motor *Motor,uint8_t CAN_ID)
{
	uint8_t tx_data[8]={0};
	txMsg.ExtId = Communication_Type_CanID<<24|CAN_ID<<16|Master_CAN_ID<<8|Motor->CAN_ID;
  	Motor->CAN_ID = CAN_ID;//将新的ID导入电机结构体
  	MyCAN_Transmit(txMsg.ExtId,txMsg.DLC,tx_data);//写入指令	
}

/*************************************************************************/
/** @brief         电机参数读取，通信类型17
  * @param[in]     ID,电机的ID号
**************************************************************************/
void check_param_cybergear(MI_Motor *Motor, uint16_t index)
{
	uint8_t tx_data[8]={0};//写入的数据
    //txMsg.ExtId = Communication_Type_GetSingleParameter<<24|Master_CAN_ID<<8|Motor->CAN_ID;//装填扩展帧数据
	txMsg.ExtId = 0x11<<24|Master_CAN_ID<<8|Motor->CAN_ID;//装填扩展帧数据
    //tx_data[0]=index;//低8位
	//tx_data[1]=index>>8;//高8位
	memcpy(&tx_data[0],&index,2);
    MyCAN_Transmit(txMsg.ExtId,txMsg.DLC,tx_data);//写入指令
	if (MyCAN_ReceiveFlag())//判断是否接收到报文信息
	{
		uint32_t RxID;//用于存放ID
		uint8_t RxLength;//用于存放数据长度
		uint8_t RxData[8];//用于存放数据
		//接收信息放到对应的接收报文之中
		MyCAN_Receive(&RxID, &RxLength, RxData);
		//检验用
		/*
		OLED_ShowString(1, 1, "ID:");//第3行第1列显示
		OLED_ShowHexNum(2, 1, RxID, 8);//显示ID
		OLED_ShowString(3, 1, "DATA");//显示接收到的数组长度
		for(int i=0;i<8;i++){
			OLED_ShowHexNum(4, 2*i+1, RxData[i], 2);//显示数据
		}
		*/
	}
}

/****************************************************************************/
/** @brief          写入电机参数，对应说明书的单个参数写入18
  * @param[in]      Motor:对应控制电机结构体
  * @param[in]      Index:写入参数对应地址,可以找到相应的枚举类型为Index;
  * @param[in]      Ref:写入参数值
  * @retval         none
  **************************************************************************/
void Set_Motor_Parameter(MI_Motor *Motor,uint16_t Index,float Ref)
{
	uint8_t tx_data[8]={0};//写入的数据
	//扩展ID，包括三个部分：通信类型0x12、主ID、电机ID
	txMsg.ExtId = Communication_Type_SetSingleParameter<<24|Master_CAN_ID<<8|Motor->CAN_ID;
	txMsg.DLC = 8;
	//txMsg.ExtId = 0x12<<24|Master_CAN_ID<<8|Motor->CAN_ID;
	//参见说明书的通信类型18里面
	memcpy(&tx_data[0],&Index, 2);//将Index中的数据复制到tx_data中，复制的长度为2个字节。
	memcpy(&tx_data[4],&Ref,4);//将Ref数据复制到tx_data中，复制的长度为4字节。
    MyCAN_Transmit(txMsg.ExtId,txMsg.DLC,tx_data);//写入指令
//	USART_SendByte(0x05);
}

/*************************设置电机模式，通信类型为18*********************************/
/** @brief          设置电机模式(必须停止时调整！)
  * @param[in]      Mode为电机模式
*************************************************************************/
void Set_Mode_Cybergear(MI_Motor *Motor,uint8_t Mode)
{	
	stop_cybergear(Motor,1);//先把电机停下来，需要停机才能够进行类型修改
	uint8_t tx_data[8]={0};//写入的数据
	//扩展ID，采用的类型是0x12==>写入参数的指令
	txMsg.ExtId = Communication_Type_SetSingleParameter<<24|Master_CAN_ID<<8|Motor->CAN_ID;
	//参见说明书的通信类型18里面，对于run_mode其数据只有1个字节，对应的copy关系如下
	uint16_t Index=run_mode_idx;
	memcpy(&tx_data[0],&Index, 2);//将Index中的数据复制到tx_data中，复制的长度为2个字节。
	memcpy(&tx_data[4],&Mode,1);//将Ref数据复制到tx_data中，复制的长度为1字节。
    MyCAN_Transmit(txMsg.ExtId,txMsg.DLC,tx_data);//写入指令
}


/*************************位置模式指令写入*********************************/
/** @brief          位置模式指令写入，如果原来不是位置模式，就修改为位置模式，如果是位置模式就不改了
  * @param[in]      Motor       电机结构体
  * @param[in]      speed_max   最大速度
  * @param[in]      pos         指定位置
*************************************************************************/
void pos_mode(MI_Motor *Motor, float speed_max,float pos)
{
	//如果不是位置模式就改成位置模式
	if (Motor->mode != 0x01)
	{
	  uint8_t mode=Position_mode;//控制模式为位置控制
      Motor->mode=mode;
	  Set_Mode_Cybergear(Motor,mode);//设置电机的运动模式
	  delay_ms(10);
	  start_cybergear(Motor->CAN_ID);//使能小米电机，通信类型为3
	  delay_ms(10);
//	  OLED_ShowString(1, 4, "Position_mode");//第3行第1列显示	
	}
	Set_Motor_Parameter(Motor,LimitSpd_idx,speed_max);//速度最大值
	Set_Motor_Parameter(Motor,LocRef_idx ,pos);//位置设置
	start_cybergear(Motor->CAN_ID);
}

/*************************速度模式指令写入*********************************/
/** @brief          速度模式指令写入
  * @param[in]      Motor       电机结构体
  * @param[in]      Cur_max     最大电流
  * @param[in]      Spd         指定速度
*************************************************************************/
void speed_mode(MI_Motor *Motor, float Cur_max,float Spd)
{
	//如果不是速度模式就改成速度模式
	if (Motor->mode != 0x02)
	{
	  uint8_t mode=Speed_mode;//控制模式为位置控制
      Motor->mode=mode;
	  Set_Mode_Cybergear(Motor,mode);//设置电机的运动模式
	  start_cybergear(Motor->CAN_ID);//使能小米电机，通信类型为3	
//	  OLED_ShowString(1, 4, "Speed_mode");//第3行第1列显示
	}
	Set_Motor_Parameter(Motor,LimitCur_idx ,Cur_max);//最大电流
	Set_Motor_Parameter(Motor,SpdRef_idx ,Spd);//对应速度
}


