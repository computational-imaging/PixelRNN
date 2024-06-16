//Author: Laurie Bose
//Date: 2021

#include <scamp5.hpp>
#include "MISC_FUNCTIONS.hpp"
#include "IMG_TF.hpp"

void extract_character_from_F_into_R11(bool white_on_black_char,int threshold, int size, int xpos, int ypos)
{
	////////////////////////////////////////////////////////////////////////
	//SETUP INITIAL FLOODING RECTANGLE AND DRAW ONTO AREG FOR DISPLAY
	MISC_FUNCTIONS::load_rect_into_DREG(DENUM::R10,100,100,60,60);

	////////////////////////////////////////////////////////////////////////
	//FLOOD FILL THE CHARACTER IN THE CENTER OF THE IMAGE
	//EXTRACT CHARACTER INTO DREG AND CENTER IT

		scamp5_load_in(E,threshold);
		scamp5_kernel_begin();
			sub(F,F,E);
		scamp5_kernel_end();
		if(!white_on_black_char)
		{
			scamp5_kernel_begin();
				where(F);
					NOT(R11,FLAG);
				all();
			scamp5_kernel_end();
		}
		else
		{
			scamp5_kernel_begin();
				where(F);
					MOV(R11,FLAG);
				all();
			scamp5_kernel_end();
		}

		scamp5_flood(R10,R11,0);

		scamp5_kernel_begin();
			AND(R1,R11,R10);
			MOV(R11,R1);
		scamp5_kernel_end();

	    uint8_t bound_box_data[4];
		scamp5_scan_boundingbox(R11,bound_box_data);

		int box_x = bound_box_data[1];
		int box_y = bound_box_data[0];
		int box_w = bound_box_data[3]-bound_box_data[1];
		int box_h = bound_box_data[2]-bound_box_data[0];
		int offsetx = box_x+box_w/2-128;
		int offsety = box_y+box_h/2-128;

		//CENTER CHARACTER
		MISC_FUNCTIONS::shift_R11(offsetx,-offsety);

	////////////////////////////////////////////////////////////////////////
	//SCALE CHARACTER TO FILL REGISTER

		//SCALE UP TO APPROX FILL REGISTER BASED ON BOUNDING BOX DATA
		int scaling = (size-box_w)/2;
		if((size-box_h)/2 < scaling)
		{
			scaling = (size-box_h)/2;
		}

		const int additional_scaling_steps = 80;

		if(scaling != 0)
		{
			if(scaling > 0)
			{
				IMGTF::SCALING::DIGITAL::SCALE(R11,scaling,false);

				////////////////////////////////////////////////////////////////////////
				//MAKE MASK THAT FRAMES THE EDGES OF THE IMAGE PLANE

				int tmp = 128 - size/2;
				MISC_FUNCTIONS::load_rect_into_DREG(DENUM::R1,tmp,tmp,size,size);
				scamp5_kernel_begin();
					NOT(R10,R1);
				scamp5_kernel_end();

				//SCALE UP BIT BY BIT UNTIL HIT REGISTER EDGES
				for(int n = 0 ; n < additional_scaling_steps ; n++)
				{
					//SCALE UP CHARACTER
					IMGTF::SCALING::DIGITAL::STEP_SCALE_UP_S6(n);

					scamp5_kernel_begin();
						AND(R1,R11,R10);
					scamp5_kernel_end();
					if(  scamp5_global_or(R1))
					{
						//IF ON BOUNDS OF IMAGE PLANE THEN BREAK
						break;
					}
				}

			}
			else
			{

				IMGTF::SCALING::DIGITAL::SCALE(R11,scaling,false);

				////////////////////////////////////////////////////////////////////////
				//MAKE MASK THAT FRAMES THE EDGES OF THE IMAGE PLANE

				int tmp = 128 - size/2;
				MISC_FUNCTIONS::load_rect_into_DREG(DENUM::R1,tmp,tmp,size,size);
				scamp5_kernel_begin();
					NOT(R10,R1);
				scamp5_kernel_end();

				//SCALE UP BIT BY BIT UNTIL HIT REGISTER EDGES
				for(int n = 0 ; n < additional_scaling_steps ; n++)
				{
					//SCALE UP CHARACTER
					IMGTF::SCALING::DIGITAL::STEP_SCALE_DOWN_S6(n);

					scamp5_kernel_begin();
						AND(R1,R11,R10);
					scamp5_kernel_end();
					if(  !scamp5_global_or(R1))
					{
						//IF ON BOUNDS OF IMAGE PLANE THEN BREAK
						break;
					}
				}
			}
		}

		MISC_FUNCTIONS::shift_R11(-xpos+128,+ypos-128);
}

void duplicate_R11_into_grid_R10( int grid_cell_size, int grid_size,int grid_padding)
{
	scamp5_kernel_begin();
	MOV(R10,R11);
	CLR(R1,R2,R3,R4);
	SET(R2);
		for(int n = 0 ; n < grid_size-1;n++)
		{
			for(int x = 0 ; x < (grid_cell_size+grid_padding)/2 ; x++)
			{
				DNEWS0(R9,R11);
				DNEWS0(R11,R9);
			}

			REFRESH(R11);
			OR(R2,R10,R11);
			MOV(R10,R2);
			SET(R2);
		}
	scamp5_kernel_end();

	//DO THE SECOND PART IN AREG SINCE THE DREG ON MY SCAMP ARE BAD...
	scamp5_load_in(E,0);
	scamp5_load_in(50);
	for(int i = 0 ; i < 7;i++)
	{
		scamp5_kernel_begin();
			mov(F,E);
			WHERE(R10);
				add(F,F,IN);
				add(F,F,IN);
				add(F,F,IN);
			all();

			for(int n = 0 ; n < grid_cell_size;n++)
			{
				mov(F,F,north);
			}

			sub(F,F,IN);

			where(F);
				OR(R1,FLAG,R10);
				MOV(R10,R1);
			all();
		scamp5_kernel_end();
	}
}


void duplicate_F_into_grid_C()
{

	scamp5_kernel_begin();
		bus(NEWS,F);
		bus(C,NEWS);
	scamp5_kernel_end();

    int reset_vh = 32+64;
	while(reset_vh--){
		scamp5_kernel_begin();
			mov(C,C,east);
			mov(C,C,south);
		scamp5_kernel_end();
	}
	//first duplication
	scamp5_kernel_begin();
		bus(NEWS,C);
		bus(F,NEWS);
	scamp5_kernel_end();
	reset_vh = 64;
	while(reset_vh--){
		scamp5_kernel_begin();
			mov(C,C,west);
		scamp5_kernel_end();
	}

	scamp5_kernel_begin();
		CLR(R8);
	scamp5_kernel_end();
    scamp5_draw_begin(R8);
    	scamp5_draw_rect(0,64+64,64,128+64);
    scamp5_draw_end();
	scamp5_kernel_begin();
		WHERE(R8);
			bus(NEWS,C);
			bus(F,NEWS);
		ALL();
		CLR(R8);
		bus(NEWS,F);
		bus(C, NEWS);
	scamp5_kernel_end();

	//second duplication
    reset_vh = 64;
	while(reset_vh--){
		scamp5_kernel_begin();
			mov(F,F,north);
		scamp5_kernel_end();
	}
    scamp5_draw_begin(R8);
    	scamp5_draw_rect(64,128,128,255);
    scamp5_draw_end();
	scamp5_kernel_begin();
		WHERE(R8);
			bus(NEWS,F);
			bus(C,NEWS);
		ALL();
		CLR(R8);
		bus(NEWS,C);
		bus(F, NEWS);
	scamp5_kernel_end();

	// 3rd duplication
	reset_vh = 128;
	while(reset_vh--){
		scamp5_kernel_begin();
			mov(C,C,north);
		scamp5_kernel_end();
	}
    scamp5_draw_begin(R8);
    	scamp5_draw_rect(128,128,255,255);
    scamp5_draw_end();
    scamp5_kernel_begin();
		WHERE(R8);
			bus(NEWS,C);
			bus(F,NEWS);
		ALL();
		CLR(R8);
		bus(NEWS,F);
		bus(C, NEWS);
    	scamp5_kernel_end();

	// final duplication
	reset_vh = 128;
	while(reset_vh--){
		scamp5_kernel_begin();
			mov(C,C,west);
		scamp5_kernel_end();
	}
    scamp5_draw_begin(R8);
    	scamp5_draw_rect(0,0,255,128);
    scamp5_draw_end();
    scamp5_kernel_begin();
		WHERE(R8);
			bus(NEWS,C);
			bus(F,NEWS);
		ALL();
		CLR(R8);
		bus(NEWS,F);
		bus(C, NEWS);
	scamp5_kernel_end();


}
