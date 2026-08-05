#include "PID.h"
#include <stdint.h>

static float Kp;
static float Kd;

static int16_t LastError = 0;

void PID_Init(void)
{
		//初始参数
    Kp = 1.2f;
    Kd = 0.8f;

    PID_Reset();
}

void PID_Reset(void)
{
    LastError = 0;
}


int16_t PID_Calc(int16_t Error)
{
    float output;
    float P,D;

		//P项
    P = Kp * Error;

		//D项
    D = Kd * (Error - LastError);

		//总输出
    output = P + D;

    LastError = Error;	

    /*
    限幅
    底盘最大正负400°
    */
    if(output > 400)
        output = 400;

    if(output < -400)
        output = -400;

    return (int16_t)output;
}
