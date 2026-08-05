#include <REGX52.H>
#include "Delay.h"

unsigned char Key()
{
	unsigned char KeyNumber = 0;
		
	if(P2_2 == 0)
	{	
		delay_10us(20);
		while(P2_2==0);
		delay_10us(20);
		KeyNumber=0x01;
	}
		
	if(P2_3==0)
	{
		delay_10us(20);
		while(P2_3==0);
		delay_10us(20);
		KeyNumber=0x02;
	}	
	
	if(P2_4 == 0)
	{	
		delay_10us(20);
		while(P2_4==0);
		delay_10us(20);
		KeyNumber=0x03;
	}
	
	if(P2_5 == 0)
	{	
		delay_10us(20);
		while(P2_5==0);
		delay_10us(20);
		KeyNumber=0x04;
	}
	
	return KeyNumber;
	
}