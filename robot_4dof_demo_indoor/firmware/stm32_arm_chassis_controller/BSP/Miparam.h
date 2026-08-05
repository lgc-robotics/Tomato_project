#ifndef _MIPARAM_H
#define _MIPARAM_H
/********************************************
说明：这个文件里面的参数不做修改，为小米电机的相关参数宏定义
*/
/*****************控制参数最值，谨慎更改*********************/
#define P_MIN -12.566370616f
#define P_MAX 12.566370616f
#define V_MIN -30.0f
#define V_MAX 30.0f
#define KP_MIN 0.0f
#define KP_MAX 500.0f
#define KD_MIN 0.0f
#define KD_MAX 5.0f
#define T_MIN -12.0f
#define T_MAX 12.0f
#define MAX_P 720
#define MIN_P -720

/*****************控制命令宏定义***************************/
#define Communication_Type_GetID 0x00           //获取设备的ID和64位MCU唯一标识符
#define Communication_Type_MotionControl 0x01 	//用来向主机发送控制指令
#define Communication_Type_MotorRequest 0x02	//用来向主机反馈电机运行状态
#define Communication_Type_MotorEnable 0x03	    //电机使能运行
#define Communication_Type_MotorStop 0x04	    //电机停止运行
#define Communication_Type_SetPosZero 0x06	    //设置电机机械零位
#define Communication_Type_CanID 0x07	        //更改当前电机CAN_ID
#define Communication_Type_Control_Mode 0x12
#define Communication_Type_GetSingleParameter 0x11	//读取单个参数
#define Communication_Type_SetSingleParameter 0x12	//设定单个参数
#define Communication_Type_ErrorFeedback 0x15	    //故障反馈帧

typedef enum {
    run_mode_idx = 0x7005,//运动模式
    IqRef_idx = 0x7006,//电流模式Iq指令
    SpdRef_idx = 0x700A,//转速模式转速指令
    LimitTorque_idx = 0x700B,//转矩限制
    CurKp_idx = 0x7010,//电流的kp
    CurKi_idx = 0x7011,//电流的ki
    CurFiltGain_idx = 0x7014,//电流滤波系数
    LocRef_idx = 0x7016,//位置模式角度指令
    LimitSpd_idx = 0x7017,//位置模式速度指令
	LimitCur_idx = 0x7018,//速度位置模式电流限制
    MechPos_idx = 0x7019,//负载端计圈机械角度
    IqFilt_idx = 0x701A,//iq滤波值
    MechVel_idx = 0x701B,//负载端转速
    Vbus_idx = 0x701C,//母线电压
    Rotation_idx = 0x701D,//圈数
    LocKp_idx = 0x701E,//位置的kp
    SpdKp_idx = 0x701F,//速度的kp
    SpdKi_idx = 0x7020//速度的ki
} Index;
 

#define Motion_mode 0//运控模式  
#define Position_mode 1  //位置模式
#define Speed_mode 2     //速度模式  
#define Current_mode 3    //电流模式


typedef enum      //错误回传对照
{
    OK                 = 0,//无故障
    BAT_LOW_ERR        = 1,//欠压故障
    OVER_CURRENT_ERR   = 2,//过流
    OVER_TEMP_ERR      = 3,//过温
    MAGNETIC_ERR       = 4,//磁编码故障
    HALL_ERR_ERR       = 5,//HALL编码故障
    NO_CALIBRATION_ERR = 6//未标定
}ERROR_TAG ;

#endif
