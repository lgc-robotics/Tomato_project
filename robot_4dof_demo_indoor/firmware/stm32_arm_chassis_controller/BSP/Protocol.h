#ifndef __Protocol_H
#define __Protocol_H
#include "stdint.h"

#define FRAME_SIZE 15
#define CAR_FRAME_SIZE 12
extern uint8_t frameReady;
extern uint8_t frame[FRAME_SIZE];

void Protocol_Input(uint8_t byte);

#endif
