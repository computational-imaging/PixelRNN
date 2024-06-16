//Author: Laurie Bose
//Date: 2021

#include "MISC_FUNCTIONS.hpp"

unsigned char reverse_byte(unsigned char x)
{
    static const unsigned char table[] = {
        0x00, 0x80, 0x40, 0xc0, 0x20, 0xa0, 0x60, 0xe0,
        0x10, 0x90, 0x50, 0xd0, 0x30, 0xb0, 0x70, 0xf0,
        0x08, 0x88, 0x48, 0xc8, 0x28, 0xa8, 0x68, 0xe8,
        0x18, 0x98, 0x58, 0xd8, 0x38, 0xb8, 0x78, 0xf8,
        0x04, 0x84, 0x44, 0xc4, 0x24, 0xa4, 0x64, 0xe4,
        0x14, 0x94, 0x54, 0xd4, 0x34, 0xb4, 0x74, 0xf4,
        0x0c, 0x8c, 0x4c, 0xcc, 0x2c, 0xac, 0x6c, 0xec,
        0x1c, 0x9c, 0x5c, 0xdc, 0x3c, 0xbc, 0x7c, 0xfc,
        0x02, 0x82, 0x42, 0xc2, 0x22, 0xa2, 0x62, 0xe2,
        0x12, 0x92, 0x52, 0xd2, 0x32, 0xb2, 0x72, 0xf2,
        0x0a, 0x8a, 0x4a, 0xca, 0x2a, 0xaa, 0x6a, 0xea,
        0x1a, 0x9a, 0x5a, 0xda, 0x3a, 0xba, 0x7a, 0xfa,
        0x06, 0x86, 0x46, 0xc6, 0x26, 0xa6, 0x66, 0xe6,
        0x16, 0x96, 0x56, 0xd6, 0x36, 0xb6, 0x76, 0xf6,
        0x0e, 0x8e, 0x4e, 0xce, 0x2e, 0xae, 0x6e, 0xee,
        0x1e, 0x9e, 0x5e, 0xde, 0x3e, 0xbe, 0x7e, 0xfe,
        0x01, 0x81, 0x41, 0xc1, 0x21, 0xa1, 0x61, 0xe1,
        0x11, 0x91, 0x51, 0xd1, 0x31, 0xb1, 0x71, 0xf1,
        0x09, 0x89, 0x49, 0xc9, 0x29, 0xa9, 0x69, 0xe9,
        0x19, 0x99, 0x59, 0xd9, 0x39, 0xb9, 0x79, 0xf9,
        0x05, 0x85, 0x45, 0xc5, 0x25, 0xa5, 0x65, 0xe5,
        0x15, 0x95, 0x55, 0xd5, 0x35, 0xb5, 0x75, 0xf5,
        0x0d, 0x8d, 0x4d, 0xcd, 0x2d, 0xad, 0x6d, 0xed,
        0x1d, 0x9d, 0x5d, 0xdd, 0x3d, 0xbd, 0x7d, 0xfd,
        0x03, 0x83, 0x43, 0xc3, 0x23, 0xa3, 0x63, 0xe3,
        0x13, 0x93, 0x53, 0xd3, 0x33, 0xb3, 0x73, 0xf3,
        0x0b, 0x8b, 0x4b, 0xcb, 0x2b, 0xab, 0x6b, 0xeb,
        0x1b, 0x9b, 0x5b, 0xdb, 0x3b, 0xbb, 0x7b, 0xfb,
        0x07, 0x87, 0x47, 0xc7, 0x27, 0xa7, 0x67, 0xe7,
        0x17, 0x97, 0x57, 0xd7, 0x37, 0xb7, 0x77, 0xf7,
        0x0f, 0x8f, 0x4f, 0xcf, 0x2f, 0xaf, 0x6f, 0xef,
        0x1f, 0x9f, 0x5f, 0xdf, 0x3f, 0xbf, 0x7f, 0xff,
    };
    return table[x];
}

double sin_approx3(double angle)
{
	return angle-(angle*angle*angle)/6.0+(angle*angle*angle*angle*angle)/120.0;
}

double cos_approx3(double angle)
{
	return 1.0-(angle*angle/2.0)+(angle*angle*angle*angle/24.0);
}

double acos_approx3(double value)
{
	return M_PI*0.5-value-(value*value*value)/6.0-(value*value*value*value*value)*3.0/40.0;
}

double tan_approx3(double angle)
{
	return angle+(angle*angle*angle/3.0)+(angle*angle*angle*angle*angle*2.0/15.0);
}

void SETUP_SHIFT_R11_USING_AREG_EF(bool black_boundaries)
{
	//USES AREG F AND E
	if(black_boundaries)
	{
		scamp5_load_in(E,20);
	}
	else
	{
		scamp5_load_in(E,-20);
	}
	scamp5_load_in(100);
}

void SHIFTx1_R11_USING_AREG_EF(int shift_x, int shift_y, bool perform_shift_setup, bool setup_shift_for_black_borders)
{
	if(perform_shift_setup)
	{
		SETUP_SHIFT_R11_USING_AREG_EF(setup_shift_for_black_borders);
	}
	if(shift_x > 0)
	{
		for(int n = 0 ; n < shift_x;n++)
		{
			SHIFTx1_R11_USING_AREG_EF_WEST();
		}
	}
	else
	{
		if(shift_x < 0)
		{
			for(int n = 0 ; n < -shift_x;n++)
			{
				SHIFTx1_R11_USING_AREG_EF_EAST();
			}
		}
	}

	if(shift_y > 0)
	{
		for(int n = 0 ; n < shift_y;n++)
		{
			SHIFTx1_R11_USING_AREG_EF_SOUTH();
		}
	}
	else
	{
		if(shift_y < 0)
		{
			for(int n = 0 ; n < -shift_y;n++)
			{
				SHIFTx1_R11_USING_AREG_EF_NORTH();
			}
		}
	}
}

void SHIFTx1_R11_USING_AREG_EF_NORTH()
{
	//REQUIRES SETUP CALL SETUP_SHIFT_R11_USING_AREG_EF
	scamp5_kernel_begin();
		mov(F,IN);
		WHERE(R11);
			sub(F,F,IN);
			sub(F,F,IN);
		all();
		bus(NEWS,F);
		bus(F,XN);
		add(F,F,E);
		bus(NEWS,F);
		where(NEWS);
			MOV(R11,FLAG);
		all();
	scamp5_kernel_end();
}

void SHIFTx1_R11_USING_AREG_EF_SOUTH()
{
	//REQUIRES SETUP CALL SETUP_SHIFT_R11_USING_AREG_EF
	scamp5_kernel_begin();
		mov(F,IN);
		WHERE(R11);
			sub(F,F,IN);
			sub(F,F,IN);
		all();
		bus(NEWS,F);
		bus(F,XS);
		add(F,F,E);
		bus(NEWS,F);
		where(NEWS);
			MOV(R11,FLAG);
		all();
	scamp5_kernel_end();
}

void SHIFTx1_R11_USING_AREG_EF_EAST()
{
	//REQUIRES SETUP CALL SETUP_SHIFT_R11_USING_AREG_EF
	scamp5_kernel_begin();
		mov(F,IN);
		WHERE(R11);
			sub(F,F,IN);
			sub(F,F,IN);
		all();
		bus(NEWS,F);
		bus(F,XE);
		add(F,F,E);
		bus(NEWS,F);
		where(NEWS);
			MOV(R11,FLAG);
		all();
	scamp5_kernel_end();
}

void SHIFTx1_R11_USING_AREG_EF_WEST()
{
	//REQUIRES SETUP CALL SETUP_SHIFT_R11_USING_AREG_EF
	scamp5_kernel_begin();
		mov(F,IN);
		WHERE(R11);
			sub(F,F,IN);
			sub(F,F,IN);
		all();
		bus(NEWS,F);
		bus(F,XW);
		add(F,F,E);
		bus(NEWS,F);
		where(NEWS);
			MOV(R11,FLAG);
		all();
	scamp5_kernel_end();
}

void SHIFTx4_R11_USING_AREG_EF(int shift_x, int shift_y, bool perform_shift_setup, bool setup_shift_for_black_borders)
{
	if(perform_shift_setup)
	{
		SETUP_SHIFT_R11_USING_AREG_EF(setup_shift_for_black_borders);
	}
	if(shift_x > 0)
	{
		for(int n = 0 ; n < shift_x;n++)
		{
			SHIFTx4_R11_USING_AREG_EF_WEST();
		}
	}
	else
	{
		if(shift_x < 0)
		{
			for(int n = 0 ; n < -shift_x;n++)
			{
				SHIFTx4_R11_USING_AREG_EF_EAST();
			}
		}
	}

	if(shift_y > 0)
	{
		for(int n = 0 ; n < shift_y;n++)
		{
			SHIFTx4_R11_USING_AREG_EF_SOUTH();
		}
	}
	else
	{
		if(shift_y < 0)
		{
			for(int n = 0 ; n < -shift_y;n++)
			{
				SHIFTx4_R11_USING_AREG_EF_NORTH();
			}
		}
	}
}

void SHIFTx4_R11_USING_AREG_EF_NORTH()
{
	//REQUIRES SETUP CALL SETUP_SHIFT_R11_USING_AREG_EF
	scamp5_kernel_begin();
		mov(F,IN);
		WHERE(R11);
			sub(F,F,IN);
			sub(F,F,IN);
		all();
		bus(NEWS,F);
		bus(F,XN);
		bus(NEWS,F);
		bus(F,XN);
		bus(NEWS,F);
		bus(F,XN);
		bus(NEWS,F);
		bus(F,XN);
		add(F,F,E);
		bus(NEWS,F);
		where(NEWS);
			MOV(R11,FLAG);
		all();
	scamp5_kernel_end();
}

void SHIFTx4_R11_USING_AREG_EF_SOUTH()
{
	//REQUIRES SETUP CALL SETUP_SHIFT_R11_USING_AREG_EF
	scamp5_kernel_begin();
		mov(F,IN);
		WHERE(R11);
			sub(F,F,IN);
			sub(F,F,IN);
		all();
		bus(NEWS,F);
		bus(F,XS);
		bus(NEWS,F);
		bus(F,XS);
		bus(NEWS,F);
		bus(F,XS);
		bus(NEWS,F);
		bus(F,XS);
		add(F,F,E);
		bus(NEWS,F);
		where(NEWS);
			MOV(R11,FLAG);
		all();
	scamp5_kernel_end();
}

void SHIFTx4_R11_USING_AREG_EF_EAST()
{
	//REQUIRES SETUP CALL SETUP_SHIFT_R11_USING_AREG_EF
	scamp5_kernel_begin();
		mov(F,IN);
		WHERE(R11);
			sub(F,F,IN);
			sub(F,F,IN);
		all();
		bus(NEWS,F);
		bus(F,XE);
		bus(NEWS,F);
		bus(F,XE);
		bus(NEWS,F);
		bus(F,XE);
		bus(NEWS,F);
		bus(F,XE);
		add(F,F,E);
		bus(NEWS,F);
		where(NEWS);
			MOV(R11,FLAG);
		all();
	scamp5_kernel_end();
}

void SHIFTx4_R11_USING_AREG_EF_WEST()
{
	//REQUIRES SETUP CALL SETUP_SHIFT_R11_USING_AREG_EF
	scamp5_kernel_begin();
		mov(F,IN);
		WHERE(R11);
			sub(F,F,IN);
			sub(F,F,IN);
		all();
		bus(NEWS,F);
		bus(F,XW);
		bus(NEWS,F);
		bus(F,XW);
		bus(NEWS,F);
		bus(F,XW);
		bus(NEWS,F);
		bus(F,XW);
		add(F,F,E);
		bus(NEWS,F);
		where(NEWS);
			MOV(R11,FLAG);
		all();
	scamp5_kernel_end();
}


void copy_dreg(DENUM target,DENUM source)
{
	switch(source)
	{
		case DENUM::R0:
			scamp5_kernel_begin();
				WHERE(R0);
			scamp5_kernel_end();
			break;
		case DENUM::R1:
			scamp5_kernel_begin();
				WHERE(R1);
			scamp5_kernel_end();
			break;
		case DENUM::R2:
			scamp5_kernel_begin();
				WHERE(R2);
			scamp5_kernel_end();
			break;
		case DENUM::R3:
			scamp5_kernel_begin();
				WHERE(R3);
			scamp5_kernel_end();
			break;
		case DENUM::R4:
			scamp5_kernel_begin();
				WHERE(R4);
			scamp5_kernel_end();
			break;
		case DENUM::R5:
			scamp5_kernel_begin();
				WHERE(R5);
			scamp5_kernel_end();
			break;
		case DENUM::R6:
			scamp5_kernel_begin();
				WHERE(R6);
			scamp5_kernel_end();
			break;
		case DENUM::R7:
			scamp5_kernel_begin();
				WHERE(R7);
			scamp5_kernel_end();
			break;
		case DENUM::R8:
			scamp5_kernel_begin();
				WHERE(R8);
			scamp5_kernel_end();
			break;
		case DENUM::R9:
			scamp5_kernel_begin();
				WHERE(R9);
			scamp5_kernel_end();
			break;
		case DENUM::R10:
			scamp5_kernel_begin();
				WHERE(R10);
			scamp5_kernel_end();
			break;
		case DENUM::R11:
			scamp5_kernel_begin();
				WHERE(R11);
			scamp5_kernel_end();
			break;
		case DENUM::R12:
			scamp5_kernel_begin();
				WHERE(R12);
			scamp5_kernel_end();
			break;
	}

	switch(target)
	{
		case DENUM::R0:
			scamp5_kernel_begin();
				MOV(R0,FLAG);
			scamp5_kernel_end();
			break;
		case DENUM::R1:
			scamp5_kernel_begin();
				MOV(R1,FLAG);
			scamp5_kernel_end();
			break;
		case DENUM::R2:
			scamp5_kernel_begin();
				MOV(R2,FLAG);
			scamp5_kernel_end();
			break;
		case DENUM::R3:
			scamp5_kernel_begin();
				MOV(R3,FLAG);
			scamp5_kernel_end();
			break;
		case DENUM::R4:
			scamp5_kernel_begin();
				MOV(R4,FLAG);
			scamp5_kernel_end();
			break;
		case DENUM::R5:
			scamp5_kernel_begin();
				MOV(R5,FLAG);
			scamp5_kernel_end();
			break;
		case DENUM::R6:
			scamp5_kernel_begin();
				MOV(R6,FLAG);
			scamp5_kernel_end();
			break;
		case DENUM::R7:
			scamp5_kernel_begin();
				MOV(R7,FLAG);
			scamp5_kernel_end();
			break;
		case DENUM::R8:
			scamp5_kernel_begin();
				MOV(R8,FLAG);
			scamp5_kernel_end();
			break;
		case DENUM::R9:
			scamp5_kernel_begin();
				MOV(R9,FLAG);
			scamp5_kernel_end();
			break;
		case DENUM::R10:
			scamp5_kernel_begin();
				MOV(R10,FLAG);
			scamp5_kernel_end();
			break;
		case DENUM::R11:
			scamp5_kernel_begin();
				MOV(R11,FLAG);
			scamp5_kernel_end();
			break;
		case DENUM::R12:
			scamp5_kernel_begin();
				SET(R0);
				MOV(R12,FLAG);
			scamp5_kernel_end();
			break;
	}

	scamp5_kernel_begin();
		ALL();
	scamp5_kernel_end();
}

void copy_dreg_into_R11(DENUM reg)
{
	switch(reg)
	{
		case DENUM::R0:
			scamp5_kernel_begin();
				MOV(R11,R0);
			scamp5_kernel_end();
			break;
		case DENUM::R1:
			scamp5_kernel_begin();
				MOV(R11,R1);
			scamp5_kernel_end();
			break;
		case DENUM::R2:
			scamp5_kernel_begin();
				MOV(R11,R2);
			scamp5_kernel_end();
			break;
		case DENUM::R3:
			scamp5_kernel_begin();
				MOV(R11,R3);
			scamp5_kernel_end();
			break;
		case DENUM::R4:
			scamp5_kernel_begin();
				MOV(R11,R4);
			scamp5_kernel_end();
			break;
		case DENUM::R5:
			scamp5_kernel_begin();
				MOV(R11,R5);
			scamp5_kernel_end();
			break;
		case DENUM::R6:
			scamp5_kernel_begin();
				MOV(R11,R6);
			scamp5_kernel_end();
			break;
		case DENUM::R7:
			scamp5_kernel_begin();
				MOV(R11,R7);
			scamp5_kernel_end();
			break;
		case DENUM::R8:
			scamp5_kernel_begin();
				MOV(R11,R8);
			scamp5_kernel_end();
			break;
		case DENUM::R9:
			scamp5_kernel_begin();
				MOV(R11,R9);
			scamp5_kernel_end();
			break;
		case DENUM::R10:
			scamp5_kernel_begin();
				MOV(R11,R10);
			scamp5_kernel_end();
			break;
		case DENUM::R11:
			break;
		case DENUM::R12:
			scamp5_kernel_begin();
				MOV(R11,R12);
			scamp5_kernel_end();
			break;
	}
}

void copy_R11_into_dreg(DENUM reg)
{
	switch(reg)
	{
		case DENUM::R0:
			scamp5_kernel_begin();
				MOV(R0,R11);
			scamp5_kernel_end();
			break;
		case DENUM::R1:
			scamp5_kernel_begin();
				MOV(R1,R11);
			scamp5_kernel_end();
			break;
		case DENUM::R2:
			scamp5_kernel_begin();
				MOV(R2,R11);
			scamp5_kernel_end();
			break;
		case DENUM::R3:
			scamp5_kernel_begin();
				MOV(R3,R11);
			scamp5_kernel_end();
			break;
		case DENUM::R4:
			scamp5_kernel_begin();
				MOV(R4,R11);
			scamp5_kernel_end();
			break;
		case DENUM::R5:
			scamp5_kernel_begin();
				MOV(R5,R11);
			scamp5_kernel_end();
			break;
		case DENUM::R6:
			scamp5_kernel_begin();
				MOV(R6,R11);
			scamp5_kernel_end();
			break;
		case DENUM::R7:
			scamp5_kernel_begin();
				MOV(R7,R11);
			scamp5_kernel_end();
			break;
		case DENUM::R8:
			scamp5_kernel_begin();
				MOV(R8,R11);
			scamp5_kernel_end();
			break;
		case DENUM::R9:
			scamp5_kernel_begin();
				MOV(R9,R11);
			scamp5_kernel_end();
			break;
		case DENUM::R10:
			scamp5_kernel_begin();
				MOV(R10,R11);
			scamp5_kernel_end();
			break;
		case DENUM::R11:
			break;
		case DENUM::R12:
			scamp5_kernel_begin();
				MOV(R12,R11);
			scamp5_kernel_end();
			break;
	}
}


void shift_dreg(DENUM reg,const int x,const int y)
{
	copy_dreg(DENUM::R0,reg);
	if(x > 0)
	{
		scamp5_kernel_begin();
			WHERE(R0);
			DNEWS(R0,FLAG,west,0);
		scamp5_kernel_end();
		for(int n = 0 ; n < x-1 ; n++)
		{
			scamp5_kernel_begin();
				WHERE(R0);
				DNEWS0(R0,FLAG);
			scamp5_kernel_end();
		}
	}
	else
	{
		scamp5_kernel_begin();
			WHERE(R0);
			DNEWS(R0,FLAG,east,0);
		scamp5_kernel_end();
		for(int n = x+1 ; n < 0 ; n++)
		{
			scamp5_kernel_begin();
				WHERE(R0);
				DNEWS0(R0,FLAG);
			scamp5_kernel_end();
		}
	}

	if(y > 0)
	{
		scamp5_kernel_begin();
			WHERE(R0);
			DNEWS(R0,FLAG,north,0);
		scamp5_kernel_end();
		for(int n = 0 ; n < y-1 ; n++)
		{
			scamp5_kernel_begin();
				WHERE(R0);
				DNEWS0(R0,FLAG);
			scamp5_kernel_end();
		}
	}
	else
	{
		scamp5_kernel_begin();
			WHERE(R0);
			DNEWS(R0,FLAG,south,0);
		scamp5_kernel_end();
		for(int n = y+1 ; n < 0 ; n++)
		{
			scamp5_kernel_begin();
				WHERE(R0);
				DNEWS0(R0,FLAG);
			scamp5_kernel_end();
		}
	}
	copy_dreg(reg,DENUM::R0);
}

void shift_R11(const int x,const int y)
{
	if(x > 0)
	{
		for(int n = 0 ; n < x ; n++)
		{
			scamp5_kernel_begin();
				MOV(R0,R11);
				DNEWS(R11,R0,west,0);
			scamp5_kernel_end();
		}
	}
	else
	{
		for(int n = x ; n < 0 ; n++)
		{
			scamp5_kernel_begin();
				MOV(R0,R11);
				DNEWS(R11,R0,east,0);
			scamp5_kernel_end();
		}
	}

	if(y > 0)
	{
		for(int n = 0 ; n < y ; n++)
		{
			scamp5_kernel_begin();
				MOV(R0,R11);
				DNEWS(R11,R0,north,0);
			scamp5_kernel_end();
		}
	}
	else
	{
		for(int n = y ; n < 0 ; n++)
		{
			scamp5_kernel_begin();
				MOV(R0,R11);
				DNEWS(R11,R0,south,0);
			scamp5_kernel_end();
		}
	}
}

//grid_cell_size = 64
//grid_padding = 0
//offx = 223
//offy = 31

void load_rect_into_DREG(DENUM reg,int x,int y,int w,int h)
{
	w--; //63
	h--; //63
	switch(reg)
	{
	    case DENUM::R0:
	    	scamp5_load_rect(R0,y,x,y+h,x+w);
	    	break;

	    case DENUM::R1:
	   	    	scamp5_load_rect(R1,y,x,y+h,x+w); // R1 31, 223, 94, 286
//	    		scamp5_load_rect(R1,x,y,x+h,y+w);
	   	    	break;

	    case DENUM::R2:
	   	    	scamp5_load_rect(R2,y,x,y+h,x+w);
	   	    	break;

	    case DENUM::R3:
	   	    	scamp5_load_rect(R3,y,x,y+h,x+w);
	   	    	break;

	    case DENUM::R4:
	   	    	scamp5_load_rect(R4,y,x,y+h,x+w);
	   	    	break;

	    case DENUM::R5:
	   	    	scamp5_load_rect(R5,y,x,y+h,x+w);
	   	    	break;

	    case DENUM::R6:
	   	    	scamp5_load_rect(R6,y,x,y+h,x+w);
	   	    	break;

	    case DENUM::R7:
	   	    	scamp5_load_rect(R7,y,x,y+h,x+w);
	   	    	break;

	    case DENUM::R8:
	   	    	scamp5_load_rect(R8,y,x,y+h,x+w);
	   	    	break;

	    case DENUM::R9:
	   	    	scamp5_load_rect(R9,y,x,y+h,x+w);
	   	    	break;

	    case DENUM::R10:
	   	    	scamp5_load_rect(R10,y,x,y+h,x+w);
	   	    	break;

	    case DENUM::R11:
	   	    	scamp5_load_rect(R11,y,x,y+h,x+w);
	   	    	break;

	    case DENUM::R12:
	   	    	scamp5_load_rect(R12,y,x,y+h,x+w);
	   	    	break;
	}
}

void load_centered_rect_into_DREG(DENUM reg,int x,int y,int w,int h)
{
	w--;
	h--;
	x = x -w/2;
	y = y -h/2;

	switch(reg)
	{
	    case DENUM::R0:
	    	scamp5_load_rect(R0,y,x,y+h,x+w);
	    	break;

	    case DENUM::R1:
	   	    	scamp5_load_rect(R1,y,x,y+h,x+w);
	   	    	break;

	    case DENUM::R2:
	   	    	scamp5_load_rect(R2,y,x,y+h,x+w);
	   	    	break;

	    case DENUM::R3:
	   	    	scamp5_load_rect(R3,y,x,y+h,x+w);
	   	    	break;

	    case DENUM::R4:
	   	    	scamp5_load_rect(R4,y,x,y+h,x+w);
	   	    	break;

	    case DENUM::R5:
	   	    	scamp5_load_rect(R5,y,x,y+h,x+w);
	   	    	break;

	    case DENUM::R6:
	   	    	scamp5_load_rect(R6,y,x,y+h,x+w);
	   	    	break;

	    case DENUM::R7:
	   	    	scamp5_load_rect(R7,y,x,y+h,x+w);
	   	    	break;

	    case DENUM::R8:
	   	    	scamp5_load_rect(R8,y,x,y+h,x+w);
	   	    	break;

	    case DENUM::R9:
	   	    	scamp5_load_rect(R9,y,x,y+h,x+w);
	   	    	break;

	    case DENUM::R10:
	   	    	scamp5_load_rect(R10,y,x,y+h,x+w);
	   	    	break;

	    case DENUM::R11:
	   	    	scamp5_load_rect(R11,y,x,y+h,x+w);
	   	    	break;

	    case DENUM::R12:
	   	    	scamp5_load_rect(R12,y,x,y+h,x+w);
	   	    	break;
	}
}

void copy_F_into_areg(AENUM reg)
{
	switch(reg)
	{
		case AENUM::A:
				scamp5_kernel_begin();
					mov(A,F);
				scamp5_kernel_end();
				break;
		case AENUM::B:
				scamp5_kernel_begin();
					mov(B,F);
				scamp5_kernel_end();
				break;
		case AENUM::C:
				scamp5_kernel_begin();
					mov(C,F);
				scamp5_kernel_end();
				break;
		case AENUM::D:
				scamp5_kernel_begin();
					mov(D,F);
				scamp5_kernel_end();
				break;
		case AENUM::E:
				scamp5_kernel_begin();
					mov(E,F);
				scamp5_kernel_end();
				break;
	}
}

void copy_areg_into_F(AENUM reg)
{
	switch(reg)
	{
		case AENUM::A:
				scamp5_kernel_begin();
					mov(F,A);
				scamp5_kernel_end();
				break;
		case AENUM::B:
				scamp5_kernel_begin();
					mov(F,B);
				scamp5_kernel_end();
				break;
		case AENUM::C:
				scamp5_kernel_begin();
					mov(F,C);
				scamp5_kernel_end();
				break;
		case AENUM::D:
				scamp5_kernel_begin();
					mov(F,D);
				scamp5_kernel_end();
				break;
		case AENUM::E:
				scamp5_kernel_begin();
					mov(F,E);
				scamp5_kernel_end();
				break;
	}
}

void load_DREG_into_F(DENUM reg,int white_value,int black_value)
{
	scamp5_load_in(black_value);
	scamp5_kernel_begin();
		mov(F,IN);
	scamp5_kernel_end();
	scamp5_load_in(white_value);
	switch(reg)
	{
		case DENUM::R0:
			scamp5_kernel_begin();
				WHERE(R0);
					mov(F,IN);
				ALL();
			scamp5_kernel_end();
		break;

		case DENUM::R1:
			scamp5_kernel_begin();
				WHERE(R1);
					mov(F,IN);
				ALL();
			scamp5_kernel_end();
		break;

		case DENUM::R2:
			scamp5_kernel_begin();
				WHERE(R2);
					mov(F,IN);
				ALL();
			scamp5_kernel_end();
		break;

		case DENUM::R3:
			scamp5_kernel_begin();
				WHERE(R3);
					mov(F,IN);
				ALL();
			scamp5_kernel_end();
		break;

		case DENUM::R4:
			scamp5_kernel_begin();
				WHERE(R4);
					mov(F,IN);
				ALL();
			scamp5_kernel_end();
		break;

		case DENUM::R5:
			scamp5_kernel_begin();
				WHERE(R5);
					mov(F,IN);
				ALL();
			scamp5_kernel_end();
		break;

		case DENUM::R6:
			scamp5_kernel_begin();
				WHERE(R6);
					mov(F,IN);
				ALL();
			scamp5_kernel_end();
		break;

		case DENUM::R7:
			scamp5_kernel_begin();
				WHERE(R7);
					mov(F,IN);
				ALL();
			scamp5_kernel_end();
		break;

		case DENUM::R8:
			scamp5_kernel_begin();
				WHERE(R8);
					mov(F,IN);
				ALL();
			scamp5_kernel_end();
		break;

		case DENUM::R9:
			scamp5_kernel_begin();
				WHERE(R9);
					mov(F,IN);
				ALL();
			scamp5_kernel_end();
		break;

		case DENUM::R10:
			scamp5_kernel_begin();
				WHERE(R10);
					mov(F,IN);
				ALL();
			scamp5_kernel_end();
		break;

		case DENUM::R11:
			scamp5_kernel_begin();
				WHERE(R11);
					mov(F,IN);
				ALL();
			scamp5_kernel_end();
		break;

		case DENUM::R12:
			scamp5_kernel_begin();
				WHERE(R12);
					mov(F,IN);
				ALL();
			scamp5_kernel_end();
		break;
	}
}


void draw_16_segment_display_to_R11(int width,int height,int thickness,bool segment_toggles[16])
{
	int digit_half_height = height/2;
	int digit_half_width = width/2;

	scamp5_kernel_begin();
		CLR(R11);
	scamp5_kernel_end();

	scamp5_draw_begin(R11);
	/////////////////////////////////////////////////////////////////////////
	//upper vertical segments

	//1
	if(segment_toggles[0])
	{
		scamp5_draw_line(128-digit_half_width,128-digit_half_height,128-digit_half_width,128);
	}
	//2
	if(segment_toggles[1])
	{
		scamp5_draw_line(128,128-digit_half_height,128,128);
	}
	//3
	if(segment_toggles[2])
	{
		scamp5_draw_line(128+digit_half_width,128-digit_half_height,128+digit_half_width,128);
	}

	/////////////////////////////////////////////////////////////////////////
	//lower vertical segments

	//4
	if(segment_toggles[3])
	{
		scamp5_draw_line(128-digit_half_width,128+digit_half_height,128-digit_half_width,128);
	}
	//5
	if(segment_toggles[4])
	{
		scamp5_draw_line(128,128+digit_half_height,128,128);
	}
	//6
	if(segment_toggles[5])
	{
	scamp5_draw_line(128+digit_half_width,128+digit_half_height,128+digit_half_width,128);
	}

	/////////////////////////////////////////////////////////////////////////
	//upper horizontal segments

	//7
	if(segment_toggles[6])
	{
		scamp5_draw_line(128-digit_half_width,128-digit_half_height,128,128-digit_half_height);
	}
	//8
	if(segment_toggles[7])
	{
		scamp5_draw_line(128+digit_half_width,128-digit_half_height,128,128-digit_half_height);
	}

	/////////////////////////////////////////////////////////////////////////
	//middle horizontal segments

	//9
	if(segment_toggles[8])
	{
		scamp5_draw_line(128-digit_half_width,128,128,128);
	}
	//10
	if(segment_toggles[9])
	{
		scamp5_draw_line(128+digit_half_width,128,128,128);
	}

	/////////////////////////////////////////////////////////////////////////
	//lower horizontal segments

	//11
	if(segment_toggles[10])
	{
		scamp5_draw_line(128-digit_half_width,128+digit_half_height,128,128+digit_half_height);
	}
	//12
	if(segment_toggles[11])
	{
		scamp5_draw_line(128+digit_half_width,128+digit_half_height,128,128+digit_half_height);
	}

	/////////////////////////////////////////////////////////////////////////
	//upper diagonal segments

	//13
	if(segment_toggles[12])
	{
		scamp5_draw_line(128-digit_half_width,128-digit_half_height,128,128);
	}
	//14
	if(segment_toggles[13])
	{
		scamp5_draw_line(128+digit_half_width,128-digit_half_height,128,128);
	}

	/////////////////////////////////////////////////////////////////////////
	//lower diagonal segments

	//15
	if(segment_toggles[14])
	{
		scamp5_draw_line(128-digit_half_width,128+digit_half_height,128,128);
	}
	//16
	if(segment_toggles[15])
	{
		scamp5_draw_line(128+digit_half_width,128+digit_half_height,128,128);
	}

	scamp5_draw_end();

	for(int n = 0 ; n < thickness ; n++)
	{
		scamp5_kernel_begin();
			MOV(R10,R11);
		scamp5_kernel_end();
		shift_R11(1,1);
		scamp5_kernel_begin();
			OR(R1,R10,R11);
			MOV(R10,R1);
			MOV(R11,R10);
		scamp5_kernel_end();
		shift_R11(-1,1);
		scamp5_kernel_begin();
			OR(R1,R10,R11);
			MOV(R10,R1);
			MOV(R11,R10);
		scamp5_kernel_end();
		shift_R11(1,-1);
		scamp5_kernel_begin();
			OR(R1,R10,R11);
			MOV(R10,R1);
			MOV(R11,R10);
		scamp5_kernel_end();
		shift_R11(-1,-1);
		scamp5_kernel_begin();
			OR(R1,R10,R11);
			MOV(R10,R1);
			MOV(R11,R10);
		scamp5_kernel_end();
	}
}

void draw_16_segment_digit_to_R11(int width,int height,int thickness,int digit)
{
	if(digit == 0)
	{
		bool digit_segs[]{true,false,true,true,false,true,true,true,false,false,true,true,false,false,false,false};
		draw_16_segment_display_to_R11(width,height, thickness,digit_segs);
	}

	if(digit == 1)
	{
		bool digit_segs[]{false,true,false,false,true,false,false,false,false,false,false,false,false,false,false,false};
		draw_16_segment_display_to_R11(width,height, thickness,digit_segs);
	}

	if(digit == 2)
	{
		bool digit_segs[]{true,false,false,false,false,true,true,true,true,true,true,true,false,false,false,false};
		draw_16_segment_display_to_R11(width,height, thickness,digit_segs);
	}

	if(digit == 3)
	{
		bool digit_segs[]{true,false,false,true,false,false,true,true,true,true,true,true,false,false,false,false};
		draw_16_segment_display_to_R11(width,height, thickness,digit_segs);
	}

	if(digit == 4)
	{
		bool digit_segs[]{true,false,true,true,false,false,false,false,true,true,false,false,false,false,false,false};
		draw_16_segment_display_to_R11(width,height, thickness,digit_segs);
	}

	if(digit == 5)
	{
		bool digit_segs[]{false,false,true,true,false,false,true,true,true,true,true,true,false,false,false,false};
		draw_16_segment_display_to_R11(width,height, thickness,digit_segs);
	}

	if(digit == 6)
	{
		bool digit_segs[]{false,false,true,true,false,true,true,true,true,true,true,true,false,false,false,false};
		draw_16_segment_display_to_R11(width,height, thickness,digit_segs);
	}

	if(digit == 7)
	{
		bool digit_segs[]{true,false,false,true,false,false,true,true,false,false,false,false,false,false,false,false};
		draw_16_segment_display_to_R11(width,height, thickness,digit_segs);
	}

	if(digit == 8)
	{
		bool digit_segs[]{true,false,true,true,false,true,true,true,true,true,true,true,false,false,false,false};
		draw_16_segment_display_to_R11(width,height, thickness,digit_segs);
	}

	if(digit == 9)
	{
		bool digit_segs[]{true,false,true,true,false,false,true,true,true,true,false,false,false,false,false,false};
		draw_16_segment_display_to_R11(width,height, thickness,digit_segs);
	}
}

scamp5_kernel sobel_edge_R10([]{

    using namespace scamp5_kernel_api;

    // vertical edge
    mov(A,C,north);
    mov(B,C,south);
    add(A,A,B);
    add(A,A,C);
    add(A,A,C);

    mov(B,A,east);
    mov(A,A,west);

    sub(B,B,A);// B = B - A
    abs(D,B);// D is the vertical edge

    // horizontal edge
    mov(A,C,east);
    mov(B,C,west);
    add(A,A,B);
    add(A,A,C);
    add(A,A,C);

    mov(B,A,south);
    mov(A,A,north);

    sub(B,B,A);// B = B - A
    abs(A,B);// A is the horizontal edge

    add(A,A,D);// merge both result

    // digitize
    sub(A,A,IN);
    where(A);
    	DNEWS(R11,FLAG,east|west|north|south);
    	MOV(R1,FLAG);
    all();

    // filter stand-alone points
    NOT(R2,R11);
    CLR_IF(R1,R2);

    // merge result into R10
    MOV(R2,R10);
	OR(R10,R1,R2);

    res(A);

});


void acquire_edge_image_R10(int gain,int edge_thresold, int edge_expansion,int HDR_iterations, int HDR_exposure_time)
{

	//first exposure
	scamp5_kernel_begin();
		get_image(C);
		CLR(R10);
		respix(F);// store reset level of PIX in F
	scamp5_kernel_end();

	// apply gain and get edge map
	scamp5_load_in(edge_thresold);
	scamp5_launch_kernel(sobel_edge_R10);

	for(int n = 0 ; n < gain ; n++)
	{
		scamp5_kernel_begin();
			where(C);
				mov(A,C);
				mov(B,C);
				add(C,A,B);
			all();
		scamp5_kernel_end();
		scamp5_load_in(edge_thresold);
		scamp5_launch_kernel(sobel_edge_R10);
	}

	scamp5_kernel_begin();
		mov(D,C);
		respix(C);// store reset level of PIX in F
	scamp5_kernel_end();

	// short exposure iterations to deal with high light part
	for(int i=0;i<HDR_iterations;i++){
		vs_wait(HDR_exposure_time);

		scamp5_kernel_begin();
			getpix(C,F);
		scamp5_kernel_end();

		scamp5_load_in(edge_thresold);
		scamp5_launch_kernel(sobel_edge_R10);

		scamp5_kernel_begin();
			mov(E,C);
			add(C,C,D);
		scamp5_kernel_end();

		scamp5_load_in(edge_thresold);
		scamp5_launch_kernel(sobel_edge_R10);

		scamp5_kernel_begin();
			mov(C,E);
		scamp5_kernel_end();
	}

	for(int n = 0 ; n < edge_expansion ; n++)
	{
		scamp5_kernel_begin();
			DNEWS(R11,R10,east | west | south | north);
			MOV(R1,R10);
			OR(R10,R1,R11);
		scamp5_kernel_end();
	}
}

int DREG_SUM_R10()
{

	//OLD VERSION

	//SELECT BOTTOM ROW
	scamp5_select_pattern(255,0,0,255);

	scamp5_kernel_begin();
		CLR(R2,R4);
	scamp5_kernel_end();
	for(int m = 0 ; m < 16 ; m++)
	{
		scamp5_kernel_begin();
			MOV(R1,R10);// WHITE PIXELS COPY THOSE BELOW THEM
			NOT(R3,R10);//BLACKS PIXELS COPY THOSE ABOVE THEM
			DNEWS0(R11,R10); //FALL PIXELS

			AND(R3,SELECT,R10);//GET BOTTOM ROW OF CURRENT STACKED PIXLE IMAGE WHICH DNEWS HAS MOVEd OFF THE IMAGE PLANE IN R11
			OR(R10,R3,R11);//UPDATE STACKED PIXEL IMAGE, IE R11 + BOTTOM ROW OF PREVIOUS STACKED PIXEL IMAGE

			MOV(R1,R10);// WHITE PIXELS COPY THOSE BELOW THEM
			NOT(R3,R10);//BLACKS PIXELS COPY THOSE ABOVE THEM
			DNEWS0(R11,R10); //FALL PIXELS

			AND(R3,SELECT,R10);//GET BOTTOM ROW OF CURRENT STACKED PIXLE IMAGE WHICH DNEWS HAS MOVEd OFF THE IMAGE PLANE IN R11
			OR(R10,R3,R11);//UPDATE STACKED PIXEL IMAGE, IE R11 + BOTTOM ROW OF PREVIOUS STACKED PIXEL IMAGE

			MOV(R1,R10);// WHITE PIXELS COPY THOSE BELOW THEM
			NOT(R3,R10);//BLACKS PIXELS COPY THOSE ABOVE THEM
			DNEWS0(R11,R10); //FALL PIXELS

			AND(R3,SELECT,R10);//GET BOTTOM ROW OF CURRENT STACKED PIXLE IMAGE WHICH DNEWS HAS MOVEd OFF THE IMAGE PLANE IN R11
			OR(R10,R3,R11);//UPDATE STACKED PIXEL IMAGE, IE R11 + BOTTOM ROW OF PREVIOUS STACKED PIXEL IMAGE

			MOV(R1,R10);// WHITE PIXELS COPY THOSE BELOW THEM
			NOT(R3,R10);//BLACKS PIXELS COPY THOSE ABOVE THEM
			DNEWS0(R11,R10); //FALL PIXELS

			AND(R3,SELECT,R10);//GET BOTTOM ROW OF CURRENT STACKED PIXLE IMAGE WHICH DNEWS HAS MOVEd OFF THE IMAGE PLANE IN R11
			OR(R10,R3,R11);//UPDATE STACKED PIXEL IMAGE, IE R11 + BOTTOM ROW OF PREVIOUS STACKED PIXEL IMAGE

			MOV(R1,R10);// WHITE PIXELS COPY THOSE BELOW THEM
			NOT(R3,R10);//BLACKS PIXELS COPY THOSE ABOVE THEM
			DNEWS0(R11,R10); //FALL PIXELS

			AND(R3,SELECT,R10);//GET BOTTOM ROW OF CURRENT STACKED PIXLE IMAGE WHICH DNEWS HAS MOVEd OFF THE IMAGE PLANE IN R11
			OR(R10,R3,R11);//UPDATE STACKED PIXEL IMAGE, IE R11 + BOTTOM ROW OF PREVIOUS STACKED PIXEL IMAGE

			MOV(R1,R10);// WHITE PIXELS COPY THOSE BELOW THEM
			NOT(R3,R10);//BLACKS PIXELS COPY THOSE ABOVE THEM
			DNEWS0(R11,R10); //FALL PIXELS

			AND(R3,SELECT,R10);//GET BOTTOM ROW OF CURRENT STACKED PIXLE IMAGE WHICH DNEWS HAS MOVEd OFF THE IMAGE PLANE IN R11
			OR(R10,R3,R11);//UPDATE STACKED PIXEL IMAGE, IE R11 + BOTTOM ROW OF PREVIOUS STACKED PIXEL IMAGE

			MOV(R1,R10);// WHITE PIXELS COPY THOSE BELOW THEM
			NOT(R3,R10);//BLACKS PIXELS COPY THOSE ABOVE THEM
			DNEWS0(R11,R10); //FALL PIXELS

			AND(R3,SELECT,R10);//GET BOTTOM ROW OF CURRENT STACKED PIXLE IMAGE WHICH DNEWS HAS MOVEd OFF THE IMAGE PLANE IN R11
			OR(R10,R3,R11);//UPDATE STACKED PIXEL IMAGE, IE R11 + BOTTOM ROW OF PREVIOUS STACKED PIXEL IMAGE

			MOV(R1,R10);// WHITE PIXELS COPY THOSE BELOW THEM
			NOT(R3,R10);//BLACKS PIXELS COPY THOSE ABOVE THEM
			DNEWS0(R11,R10); //FALL PIXELS

			AND(R3,SELECT,R10);//GET BOTTOM ROW OF CURRENT STACKED PIXLE IMAGE WHICH DNEWS HAS MOVEd OFF THE IMAGE PLANE IN R11
			OR(R10,R3,R11);//UPDATE STACKED PIXEL IMAGE, IE R11 + BOTTOM ROW OF PREVIOUS STACKED PIXEL IMAGE

			MOV(R1,R10);// WHITE PIXELS COPY THOSE BELOW THEM
			NOT(R3,R10);//BLACKS PIXELS COPY THOSE ABOVE THEM
			DNEWS0(R11,R10); //FALL PIXELS

			AND(R3,SELECT,R10);//GET BOTTOM ROW OF CURRENT STACKED PIXLE IMAGE WHICH DNEWS HAS MOVEd OFF THE IMAGE PLANE IN R11
			OR(R10,R3,R11);//UPDATE STACKED PIXEL IMAGE, IE R11 + BOTTOM ROW OF PREVIOUS STACKED PIXEL IMAGE

			MOV(R1,R10);// WHITE PIXELS COPY THOSE BELOW THEM
			NOT(R3,R10);//BLACKS PIXELS COPY THOSE ABOVE THEM
			DNEWS0(R11,R10); //FALL PIXELS

			AND(R3,SELECT,R10);//GET BOTTOM ROW OF CURRENT STACKED PIXLE IMAGE WHICH DNEWS HAS MOVEd OFF THE IMAGE PLANE IN R11
			OR(R10,R3,R11);//UPDATE STACKED PIXEL IMAGE, IE R11 + BOTTOM ROW OF PREVIOUS STACKED PIXEL IMAGE

			MOV(R1,R10);// WHITE PIXELS COPY THOSE BELOW THEM
			NOT(R3,R10);//BLACKS PIXELS COPY THOSE ABOVE THEM
			DNEWS0(R11,R10); //FALL PIXELS

			AND(R3,SELECT,R10);//GET BOTTOM ROW OF CURRENT STACKED PIXLE IMAGE WHICH DNEWS HAS MOVEd OFF THE IMAGE PLANE IN R11
			OR(R10,R3,R11);//UPDATE STACKED PIXEL IMAGE, IE R11 + BOTTOM ROW OF PREVIOUS STACKED PIXEL IMAGE

			MOV(R1,R10);// WHITE PIXELS COPY THOSE BELOW THEM
			NOT(R3,R10);//BLACKS PIXELS COPY THOSE ABOVE THEM
			DNEWS0(R11,R10); //FALL PIXELS

			AND(R3,SELECT,R10);//GET BOTTOM ROW OF CURRENT STACKED PIXLE IMAGE WHICH DNEWS HAS MOVEd OFF THE IMAGE PLANE IN R11
			OR(R10,R3,R11);//UPDATE STACKED PIXEL IMAGE, IE R11 + BOTTOM ROW OF PREVIOUS STACKED PIXEL IMAGE

			MOV(R1,R10);// WHITE PIXELS COPY THOSE BELOW THEM
			NOT(R3,R10);//BLACKS PIXELS COPY THOSE ABOVE THEM
			DNEWS0(R11,R10); //FALL PIXELS

			AND(R3,SELECT,R10);//GET BOTTOM ROW OF CURRENT STACKED PIXLE IMAGE WHICH DNEWS HAS MOVEd OFF THE IMAGE PLANE IN R11
			OR(R10,R3,R11);//UPDATE STACKED PIXEL IMAGE, IE R11 + BOTTOM ROW OF PREVIOUS STACKED PIXEL IMAGE

			MOV(R1,R10);// WHITE PIXELS COPY THOSE BELOW THEM
			NOT(R3,R10);//BLACKS PIXELS COPY THOSE ABOVE THEM
			DNEWS0(R11,R10); //FALL PIXELS

			AND(R3,SELECT,R10);//GET BOTTOM ROW OF CURRENT STACKED PIXLE IMAGE WHICH DNEWS HAS MOVEd OFF THE IMAGE PLANE IN R11
			OR(R10,R3,R11);//UPDATE STACKED PIXEL IMAGE, IE R11 + BOTTOM ROW OF PREVIOUS STACKED PIXEL IMAGE

			MOV(R1,R10);// WHITE PIXELS COPY THOSE BELOW THEM
			NOT(R3,R10);//BLACKS PIXELS COPY THOSE ABOVE THEM
			DNEWS0(R11,R10); //FALL PIXELS

			AND(R3,SELECT,R10);//GET BOTTOM ROW OF CURRENT STACKED PIXLE IMAGE WHICH DNEWS HAS MOVEd OFF THE IMAGE PLANE IN R11
			OR(R10,R3,R11);//UPDATE STACKED PIXEL IMAGE, IE R11 + BOTTOM ROW OF PREVIOUS STACKED PIXEL IMAGE

			MOV(R1,R10);// WHITE PIXELS COPY THOSE BELOW THEM
			NOT(R3,R10);//BLACKS PIXELS COPY THOSE ABOVE THEM
			DNEWS0(R11,R10); //FALL PIXELS

			AND(R3,SELECT,R10);//GET BOTTOM ROW OF CURRENT STACKED PIXLE IMAGE WHICH DNEWS HAS MOVEd OFF THE IMAGE PLANE IN R11
			OR(R10,R3,R11);//UPDATE STACKED PIXEL IMAGE, IE R11 + BOTTOM ROW OF PREVIOUS STACKED PIXEL IMAGE
		scamp5_kernel_end();
	}
	scamp5_kernel_begin();
			ALL();
	scamp5_kernel_end();

	scamp5_select_pattern(255,255,0,255);
	scamp5_kernel_begin();
		DNEWS(R11,R10,south,0);
		XOR(R1,R11,R10);
		NOT(R3,SELECT);
		AND(R2,R1,R3);
		MOV(R1,R2);
	scamp5_kernel_end();

	uint8_t event_data [256*2];
	scamp5_scan_events(R1,event_data,256);

	int sum = 0;
	for(int n = 0 ; n < 256 ; n++)
	{
		if(event_data[n*2] == 0 && event_data[n*2+1] == 0)
		{
			break;
		}
		sum = sum + 255- event_data[n*2+1];
	}
	return sum;
}

int DREG_SUM_R10_STEVE_OPT()
{
	//STEVE VERSION

	//SELECT BOTTOM ROW
	scamp5_select_pattern(255,0,0,255);
	scamp5_in(F,15);
	scamp5_kernel_begin();
		icw(1,{plrb,bit,bitmode,wr,LR4});
		WHERE(R4);
	scamp5_kernel_end();
		scamp5_in(F,-120);
	scamp5_kernel_begin();
		all();
		bus(NEWS, F);

		CLR(R2,R4); //CLEAR UNUSED DNEWS DIRECTIONS
		MOV(R1,R10);//moved this out of loop below so first iteration runs properly
	scamp5_kernel_end();

	//CAN MAKE THIS QUITE A BIT FASTER BY JUST SENDING A FEW KERNELS CONTAINING MANY DUPLICATED INSTRUCTIONS
	//RATHER THAN SENDING SHORT KERNELS WITH NO DUPLICATES MANY MANY TIMES
	for(int m = 0 ; m < 16 ; m++)
	{
		scamp5_kernel_begin();
			//MOV(R1,R10);// WHITE PIXELS COPY THOSE BELOW THEM
			//NOT(R3,R10);//BLACKS PIXELS COPY THOSE ABOVE THEM
			icw(1,{SEL4,seln,sels,plrb,bitmode,plwb,nbit,LR3,RR10});//amalgamating R3=!R10 and preset of boundary bus inputs
			 icw(2,{plrb,bit,s_in,rid,nb,LR11,RR10});//DNEWS(R9,R10) with floating boundary input
			icw(1,{LR1,LR10,plrb,bit,bitmode,RR11,SEL4,seln,sels}); //R1,R10 = R9; floating boundary defined

			icw(1,{SEL4,seln,sels,plrb,bitmode,plwb,nbit,LR3,RR10});//amalgamating R3=!R10 and preset of boundary bus inputs
			 icw(2,{plrb,bit,s_in,rid,nb,LR11,RR10});//DNEWS(R9,R10) with floating boundary input
			icw(1,{LR1,LR10,plrb,bit,bitmode,RR11,SEL4,seln,sels}); //R1,R10 = R9; floating boundary defined

			icw(1,{SEL4,seln,sels,plrb,bitmode,plwb,nbit,LR3,RR10});//amalgamating R3=!R10 and preset of boundary bus inputs
			 icw(2,{plrb,bit,s_in,rid,nb,LR11,RR10});//DNEWS(R9,R10) with floating boundary input
			icw(1,{LR1,LR10,plrb,bit,bitmode,RR11,SEL4,seln,sels}); //R1,R10 = R9; floating boundary defined

			icw(1,{SEL4,seln,sels,plrb,bitmode,plwb,nbit,LR3,RR10});//amalgamating R3=!R10 and preset of boundary bus inputs
			 icw(2,{plrb,bit,s_in,rid,nb,LR11,RR10});//DNEWS(R9,R10) with floating boundary input
			icw(1,{LR1,LR10,plrb,bit,bitmode,RR11,SEL4,seln,sels}); //R1,R10 = R9; floating boundary defined
			icw(1,{SEL4,seln,sels,plrb,bitmode,plwb,nbit,LR3,RR10});//amalgamating R3=!R10 and preset of boundary bus inputs

			icw(2,{plrb,bit,s_in,rid,nb,LR11,RR10});//DNEWS(R9,R10) with floating boundary input
			icw(1,{LR1,LR10,plrb,bit,bitmode,RR11,SEL4,seln,sels}); //R1,R10 = R9; floating boundary defined

			icw(1,{SEL4,seln,sels,plrb,bitmode,plwb,nbit,LR3,RR10});//amalgamating R3=!R10 and preset of boundary bus inputs
			 icw(2,{plrb,bit,s_in,rid,nb,LR11,RR10});//DNEWS(R9,R10) with floating boundary input
			icw(1,{LR1,LR10,plrb,bit,bitmode,RR11,SEL4,seln,sels}); //R1,R10 = R9; floating boundary defined

			icw(1,{SEL4,seln,sels,plrb,bitmode,plwb,nbit,LR3,RR10});//amalgamating R3=!R10 and preset of boundary bus inputs
			 icw(2,{plrb,bit,s_in,rid,nb,LR11,RR10});//DNEWS(R9,R10) with floating boundary input
			icw(1,{LR1,LR10,plrb,bit,bitmode,RR11,SEL4,seln,sels}); //R1,R10 = R9; floating boundary defined

			icw(1,{SEL4,seln,sels,plrb,bitmode,plwb,nbit,LR3,RR10});//amalgamating R3=!R10 and preset of boundary bus inputs
			 icw(2,{plrb,bit,s_in,rid,nb,LR11,RR10});//DNEWS(R9,R10) with floating boundary input
			icw(1,{LR1,LR10,plrb,bit,bitmode,RR11,SEL4,seln,sels}); //R1,R10 = R9; floating boundary defined

			icw(1,{SEL4,seln,sels,plrb,bitmode,plwb,nbit,LR3,RR10});//amalgamating R3=!R10 and preset of boundary bus inputs
			 icw(2,{plrb,bit,s_in,rid,nb,LR11,RR10});//DNEWS(R9,R10) with floating boundary input
			icw(1,{LR1,LR10,plrb,bit,bitmode,RR11,SEL4,seln,sels}); //R1,R10 = R9; floating boundary defined

			icw(1,{SEL4,seln,sels,plrb,bitmode,plwb,nbit,LR3,RR10});//amalgamating R3=!R10 and preset of boundary bus inputs
			 icw(2,{plrb,bit,s_in,rid,nb,LR11,RR10});//DNEWS(R9,R10) with floating boundary input
			icw(1,{LR1,LR10,plrb,bit,bitmode,RR11,SEL4,seln,sels}); //R1,R10 = R9; floating boundary defined

			icw(1,{SEL4,seln,sels,plrb,bitmode,plwb,nbit,LR3,RR10});//amalgamating R3=!R10 and preset of boundary bus inputs
			 icw(2,{plrb,bit,s_in,rid,nb,LR11,RR10});//DNEWS(R9,R10) with floating boundary input
			icw(1,{LR1,LR10,plrb,bit,bitmode,RR11,SEL4,seln,sels}); //R1,R10 = R9; floating boundary defined

			icw(1,{SEL4,seln,sels,plrb,bitmode,plwb,nbit,LR3,RR10});//amalgamating R3=!R10 and preset of boundary bus inputs
			 icw(2,{plrb,bit,s_in,rid,nb,LR11,RR10});//DNEWS(R9,R10) with floating boundary input
			icw(1,{LR1,LR10,plrb,bit,bitmode,RR11,SEL4,seln,sels}); //R1,R10 = R9; floating boundary defined
			icw(1,{SEL4,seln,sels,plrb,bitmode,plwb,nbit,LR3,RR10});//amalgamating R3=!R10 and preset of boundary bus inputs

			icw(2,{plrb,bit,s_in,rid,nb,LR11,RR10});//DNEWS(R9,R10) with floating boundary input
			icw(1,{LR1,LR10,plrb,bit,bitmode,RR11,SEL4,seln,sels}); //R1,R10 = R9; floating boundary defined

			icw(1,{SEL4,seln,sels,plrb,bitmode,plwb,nbit,LR3,RR10});//amalgamating R3=!R10 and preset of boundary bus inputs
			 icw(2,{plrb,bit,s_in,rid,nb,LR11,RR10});//DNEWS(R9,R10) with floating boundary input
			icw(1,{LR1,LR10,plrb,bit,bitmode,RR11,SEL4,seln,sels}); //R1,R10 = R9; floating boundary defined

			icw(1,{SEL4,seln,sels,plrb,bitmode,plwb,nbit,LR3,RR10});//amalgamating R3=!R10 and preset of boundary bus inputs
			 icw(2,{plrb,bit,s_in,rid,nb,LR11,RR10});//DNEWS(R9,R10) with floating boundary input
			icw(1,{LR1,LR10,plrb,bit,bitmode,RR11,SEL4,seln,sels}); //R1,R10 = R9; floating boundary defined

			icw(1,{SEL4,seln,sels,plrb,bitmode,plwb,nbit,LR3,RR10});//amalgamating R3=!R10 and preset of boundary bus inputs
			 icw(2,{plrb,bit,s_in,rid,nb,LR11,RR10});//DNEWS(R9,R10) with floating boundary input
			icw(1,{LR1,LR10,plrb,bit,bitmode,RR11,SEL4,seln,sels}); //R1,R10 = R9; floating boundary defined
		scamp5_kernel_end();
	}

	scamp5_select_pattern(255,255,0,255);
	scamp5_kernel_begin();
		DNEWS(R11,R10,south,0);
		XOR(R1,R11,R10);
		NOT(R3,SELECT);
		AND(R2,R1,R3);
		MOV(R1,R2);
	scamp5_kernel_end();

	uint8_t event_data [256*2];
	scamp5_scan_events(R1,event_data,256);

	int sum = 0;
	for(int n = 0 ; n < 256 ; n++)
	{
		if(event_data[n*2] == 0 && event_data[n*2+1] == 0)
		{
			break;
		}
		sum = sum + 255- event_data[n*2+1];
	}
	return sum;
}




void DREG_STACKING_R10_INTO_R1_STEVE_OPT()
{
	//STACKS ALL WHITE PIXELS IN R10
	//SELECT BOTTOM ROW
	scamp5_select_pattern(255,0,0,255);
	scamp5_in(F,15);
	scamp5_kernel_begin();
		icw(1,{plrb,bit,bitmode,wr,LR4});
		WHERE(R4);
	scamp5_kernel_end();
		scamp5_in(F,-120);
	scamp5_kernel_begin();
		all();
		bus(NEWS, F);

		CLR(R2,R4); //CLEAR UNUSED DNEWS DIRECTIONS
		MOV(R1,R10);//moved this out of loop below so first iteration runs properly
	scamp5_kernel_end();

	//CAN MAKE THIS QUITE A BIT FASTER BY JUST SENDING A FEW KERNELS CONTAINING MANY DUPLICATED INSTRUCTIONS
	//RATHER THAN SENDING SHORT KERNELS WITH NO DUPLICATES MANY MANY TIMES
	for(int m = 0 ; m < 16 ; m++)
	{
		scamp5_kernel_begin();
			//MOV(R1,R10);// WHITE PIXELS COPY THOSE BELOW THEM
			//NOT(R3,R10);//BLACKS PIXELS COPY THOSE ABOVE THEM
			icw(1,{SEL4,seln,sels,plrb,bitmode,plwb,nbit,LR3,RR10});//amalgamating R3=!R10 and preset of boundary bus inputs
			 icw(2,{plrb,bit,s_in,rid,nb,LR11,RR10});//DNEWS(R9,R10) with floating boundary input
			icw(1,{LR1,LR10,plrb,bit,bitmode,RR11,SEL4,seln,sels}); //R1,R10 = R9; floating boundary defined

			icw(1,{SEL4,seln,sels,plrb,bitmode,plwb,nbit,LR3,RR10});//amalgamating R3=!R10 and preset of boundary bus inputs
			 icw(2,{plrb,bit,s_in,rid,nb,LR11,RR10});//DNEWS(R9,R10) with floating boundary input
			icw(1,{LR1,LR10,plrb,bit,bitmode,RR11,SEL4,seln,sels}); //R1,R10 = R9; floating boundary defined

			icw(1,{SEL4,seln,sels,plrb,bitmode,plwb,nbit,LR3,RR10});//amalgamating R3=!R10 and preset of boundary bus inputs
			 icw(2,{plrb,bit,s_in,rid,nb,LR11,RR10});//DNEWS(R9,R10) with floating boundary input
			icw(1,{LR1,LR10,plrb,bit,bitmode,RR11,SEL4,seln,sels}); //R1,R10 = R9; floating boundary defined

			icw(1,{SEL4,seln,sels,plrb,bitmode,plwb,nbit,LR3,RR10});//amalgamating R3=!R10 and preset of boundary bus inputs
			 icw(2,{plrb,bit,s_in,rid,nb,LR11,RR10});//DNEWS(R9,R10) with floating boundary input
			icw(1,{LR1,LR10,plrb,bit,bitmode,RR11,SEL4,seln,sels}); //R1,R10 = R9; floating boundary defined
			icw(1,{SEL4,seln,sels,plrb,bitmode,plwb,nbit,LR3,RR10});//amalgamating R3=!R10 and preset of boundary bus inputs

			icw(2,{plrb,bit,s_in,rid,nb,LR11,RR10});//DNEWS(R9,R10) with floating boundary input
			icw(1,{LR1,LR10,plrb,bit,bitmode,RR11,SEL4,seln,sels}); //R1,R10 = R9; floating boundary defined

			icw(1,{SEL4,seln,sels,plrb,bitmode,plwb,nbit,LR3,RR10});//amalgamating R3=!R10 and preset of boundary bus inputs
			 icw(2,{plrb,bit,s_in,rid,nb,LR11,RR10});//DNEWS(R9,R10) with floating boundary input
			icw(1,{LR1,LR10,plrb,bit,bitmode,RR11,SEL4,seln,sels}); //R1,R10 = R9; floating boundary defined

			icw(1,{SEL4,seln,sels,plrb,bitmode,plwb,nbit,LR3,RR10});//amalgamating R3=!R10 and preset of boundary bus inputs
			 icw(2,{plrb,bit,s_in,rid,nb,LR11,RR10});//DNEWS(R9,R10) with floating boundary input
			icw(1,{LR1,LR10,plrb,bit,bitmode,RR11,SEL4,seln,sels}); //R1,R10 = R9; floating boundary defined

			icw(1,{SEL4,seln,sels,plrb,bitmode,plwb,nbit,LR3,RR10});//amalgamating R3=!R10 and preset of boundary bus inputs
			 icw(2,{plrb,bit,s_in,rid,nb,LR11,RR10});//DNEWS(R9,R10) with floating boundary input
			icw(1,{LR1,LR10,plrb,bit,bitmode,RR11,SEL4,seln,sels}); //R1,R10 = R9; floating boundary defined

			icw(1,{SEL4,seln,sels,plrb,bitmode,plwb,nbit,LR3,RR10});//amalgamating R3=!R10 and preset of boundary bus inputs
			 icw(2,{plrb,bit,s_in,rid,nb,LR11,RR10});//DNEWS(R9,R10) with floating boundary input
			icw(1,{LR1,LR10,plrb,bit,bitmode,RR11,SEL4,seln,sels}); //R1,R10 = R9; floating boundary defined

			icw(1,{SEL4,seln,sels,plrb,bitmode,plwb,nbit,LR3,RR10});//amalgamating R3=!R10 and preset of boundary bus inputs
			 icw(2,{plrb,bit,s_in,rid,nb,LR11,RR10});//DNEWS(R9,R10) with floating boundary input
			icw(1,{LR1,LR10,plrb,bit,bitmode,RR11,SEL4,seln,sels}); //R1,R10 = R9; floating boundary defined

			icw(1,{SEL4,seln,sels,plrb,bitmode,plwb,nbit,LR3,RR10});//amalgamating R3=!R10 and preset of boundary bus inputs
			 icw(2,{plrb,bit,s_in,rid,nb,LR11,RR10});//DNEWS(R9,R10) with floating boundary input
			icw(1,{LR1,LR10,plrb,bit,bitmode,RR11,SEL4,seln,sels}); //R1,R10 = R9; floating boundary defined

			icw(1,{SEL4,seln,sels,plrb,bitmode,plwb,nbit,LR3,RR10});//amalgamating R3=!R10 and preset of boundary bus inputs
			 icw(2,{plrb,bit,s_in,rid,nb,LR11,RR10});//DNEWS(R9,R10) with floating boundary input
			icw(1,{LR1,LR10,plrb,bit,bitmode,RR11,SEL4,seln,sels}); //R1,R10 = R9; floating boundary defined
			icw(1,{SEL4,seln,sels,plrb,bitmode,plwb,nbit,LR3,RR10});//amalgamating R3=!R10 and preset of boundary bus inputs

			icw(2,{plrb,bit,s_in,rid,nb,LR11,RR10});//DNEWS(R9,R10) with floating boundary input
			icw(1,{LR1,LR10,plrb,bit,bitmode,RR11,SEL4,seln,sels}); //R1,R10 = R9; floating boundary defined

			icw(1,{SEL4,seln,sels,plrb,bitmode,plwb,nbit,LR3,RR10});//amalgamating R3=!R10 and preset of boundary bus inputs
			 icw(2,{plrb,bit,s_in,rid,nb,LR11,RR10});//DNEWS(R9,R10) with floating boundary input
			icw(1,{LR1,LR10,plrb,bit,bitmode,RR11,SEL4,seln,sels}); //R1,R10 = R9; floating boundary defined

			icw(1,{SEL4,seln,sels,plrb,bitmode,plwb,nbit,LR3,RR10});//amalgamating R3=!R10 and preset of boundary bus inputs
			 icw(2,{plrb,bit,s_in,rid,nb,LR11,RR10});//DNEWS(R9,R10) with floating boundary input
			icw(1,{LR1,LR10,plrb,bit,bitmode,RR11,SEL4,seln,sels}); //R1,R10 = R9; floating boundary defined

			icw(1,{SEL4,seln,sels,plrb,bitmode,plwb,nbit,LR3,RR10});//amalgamating R3=!R10 and preset of boundary bus inputs
			 icw(2,{plrb,bit,s_in,rid,nb,LR11,RR10});//DNEWS(R9,R10) with floating boundary input
			icw(1,{LR1,LR10,plrb,bit,bitmode,RR11,SEL4,seln,sels}); //R1,R10 = R9; floating boundary defined
		scamp5_kernel_end();
	}

	scamp5_select_pattern(255,255,0,255);
	scamp5_kernel_begin();
		DNEWS(R11,R10,south,0);
		XOR(R1,R11,R10);
		NOT(R3,SELECT);
		AND(R2,R1,R3);
		MOV(R1,R2);
	scamp5_kernel_end();
}

