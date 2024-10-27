//Author: Laurie Bose
//Date: 2021

#include "AREG_POOLING.hpp"

void MAX_POOL_F(int iterations,bool maxpool_dirx, bool maxpool_diry,bool blocking) //DESTROYS CONTENT IN R0
{
	for(int n = 0 ; n < iterations ; n++)
	{
//		tick("pooling");
		//make of copy of F and shift it one pixel down
		if(!maxpool_diry)
		{
			scamp5_kernel_begin();
				bus(NEWS,F);
				bus(E,XS);
			scamp5_kernel_end();
		}
		else
		{
			scamp5_kernel_begin();
				bus(NEWS,F);
				bus(E,XN);
			scamp5_kernel_end();
		}

		//compare pixels between E and F and keep only the highest valued
		scamp5_kernel_begin();
			sub(F,F,E);
			where(F);
				//where F-E > 0 add E to revert back to the original value
				add(F,F,E);
			NOT(R0,FLAG);
			WHERE(R0);
				//where F-E > 0 copy E into F since it is a higher value
				mov(F,E);
			ALL();
		scamp5_kernel_end();

		if(maxpool_dirx)
		{
			scamp5_kernel_begin();
				bus(NEWS,F);
				bus(E,XW);
			scamp5_kernel_end();
		}
		else
		{
			scamp5_kernel_begin();
				bus(NEWS,F);
				bus(E,XE);
			scamp5_kernel_end();
		}

		//compare pixels between E and F and keep only the highest valued
		scamp5_kernel_begin();
			sub(F,F,E);
			where(F);
				//where F-E > 0 add E to revert back to the original value
				add(F,F,E);
			NOT(R0,FLAG);
			WHERE(R0);
				//where F-E > 0 copy E into F since it is a higher value
				mov(F,E);
			ALL();
		scamp5_kernel_end();
//		tock();
	}

	if(blocking)
	{
		scamp5_kernel_begin();
			mov(E,F);
		scamp5_kernel_end();
		scamp5_load_pattern(R10,2,0,252,252);
		for(int n = 0 ; n < iterations ; n++)
		{
			scamp5_kernel_begin();
				bus(NEWS,E);
				bus(E,XE);

				WHERE(R10);
					DNEWS(R10,FLAG,east);
				WHERE(R10);
					mov(F,E);
				all();
			scamp5_kernel_end();
		}
		scamp5_kernel_begin();
			mov(E,F);
		scamp5_kernel_end();
		scamp5_load_pattern(R10,2,0,252,255);
		for(int n = 0 ; n < iterations ; n++)
		{
			scamp5_kernel_begin();
				bus(NEWS,E);
				bus(E,XS);

				WHERE(R10);
					DNEWS(R10,FLAG,south);
				WHERE(R10);
					mov(F,E);
				all();
			scamp5_kernel_end();
		}
	}
}
void MAX_POOL_F2(int iterations,bool maxpool_dirx, bool maxpool_diry,bool blocking, int threshold) //DESTROYS CONTENT IN R0
{
	for(int n = 0 ; n < iterations ; n++)
	{
//		tick("pooling");
		//make of copy of F and shift it one pixel down
		if(!maxpool_diry)
		{
			scamp5_kernel_begin();
				bus(NEWS,F);
				bus(E,XS);
			scamp5_kernel_end();
		}
		else
		{
			scamp5_kernel_begin();
				bus(NEWS,F);
				bus(E,XN);
			scamp5_kernel_end();
		}

		//compare pixels between E and F and keep only the highest valued
		scamp5_kernel_begin();
			sub(F,F,E);
			where(F);
				//where F-E > 0 add E to revert back to the original value
				add(F,F,E);
			NOT(R0,FLAG);
			WHERE(R0);
				//where F-E > 0 copy E into F since it is a higher value
				mov(F,E);
			ALL();
		scamp5_kernel_end();

		if(maxpool_dirx)
		{
			scamp5_kernel_begin();
				bus(NEWS,F);
				bus(E,XW);
			scamp5_kernel_end();
		}
		else
		{
			scamp5_kernel_begin();
				bus(NEWS,F);
				bus(E,XE);
			scamp5_kernel_end();
		}

		//compare pixels between E and F and keep only the highest valued
		scamp5_kernel_begin();
			sub(F,F,E);
			where(F);
				//where F-E > 0 add E to revert back to the original value
				add(F,F,E);
			NOT(R0,FLAG);
			WHERE(R0);
				//where F-E > 0 copy E into F since it is a higher value
				mov(F,E);
			ALL();
		scamp5_kernel_end();
//		tock();
	}

	if(blocking)
	{
		scamp5_load_in(threshold);
		scamp5_kernel_begin();
			mov(E,F);
			sub(E,F,IN);
			where(E);
				MOV(R11,FLAG);
			all();
		scamp5_kernel_end();

		scamp5_load_pattern(R10,2,0,252,252);
		scamp5_load_in(-100);
		scamp5_kernel_begin();
			AND(R9,R10,R11);
			mov(E,IN);
			WHERE(R9);
				sub(E,E,IN);
				sub(E,E,IN);
			ALL();
			mov(F,E);
		scamp5_kernel_end();

		scamp5_load_pattern(R10,2,0,252,252);
		for(int n = 0 ; n < iterations ; n++)
		{
			scamp5_kernel_begin();
				bus(NEWS,E);
				bus(E,XE);

				WHERE(R10);
					DNEWS(R10,FLAG,east);
				WHERE(R10);
					mov(F,E);
				all();
			scamp5_kernel_end();
		}
		scamp5_kernel_begin();
			mov(E,F);
		scamp5_kernel_end();
		scamp5_load_pattern(R10,2,0,252,255);
		for(int n = 0 ; n < iterations ; n++)
		{
			scamp5_kernel_begin();
				bus(NEWS,E);
				bus(E,XS);

				WHERE(R10);
					DNEWS(R10,FLAG,south);
				WHERE(R10);
					mov(F,E);
				all();
			scamp5_kernel_end();
		}
	}
}
