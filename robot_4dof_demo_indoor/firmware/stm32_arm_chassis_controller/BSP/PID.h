#ifndef __PID_H
#define __PID_H

#include <stdint.h>

void PID_Init(void);
void PID_Reset(void);
int16_t PID_Calc(int16_t error);

#endif
