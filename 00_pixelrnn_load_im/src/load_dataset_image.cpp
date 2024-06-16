
#include "load_dataset_image.hpp"


void refresh_dreg_storage();


#define D2A_BITS	4
const dreg_t dreg_map[6] = {R11,  R10, R9, R8, R7, R6 };
//void load_dataset_image(const char*filepath_format,int index){
void load_dataset_image(const char*filepath_format){
	char filepath[256] = "";
	vs_stopwatch timer;

	if(vs_gui_request_done() && vs_gui_is_on()){
		int bit = 0;

		snprintf(filepath,256,filepath_format,index);
//		vs_post_text("%s\n",filepath);

		timer.reset();
		vs_gui_request_image(filepath,D2A_BITS,[&](vs_dotmat const& dotmat,int s){
			if(s<0){
				vs_post_text("image not received in time!\n");
				vs_post_text("%s\n",filepath);
				return;
			}else
			if(s<D2A_BITS){
				load_dreg_image(dreg_map[s],(const uint8_t*)dotmat.get_buffer(),dotmat.get_height(),dotmat.get_width());
			}
		});
		do{
			refresh_dreg_storage();
			vs_process_message();
		}while(!vs_gui_request_done());

		scamp5_in(E,-120);
		int i = D2A_BITS;
		while(i--){
			int level = (1<<(8 - D2A_BITS))*(1<<i) - 1;
			scamp5_in(F,level);// note: range of 'scamp5_in' is [-128,127], thus +128 is out of range
			scamp5_dynamic_kernel_begin();
				WHERE(dreg_map[i]);
				add(E,E,F);
				ALL();
			scamp5_dynamic_kernel_end();
		}
	}
}


void refresh_dreg_storage(){
	scamp5_kernel_begin();
		REFRESH(dreg_map[0]);
		REFRESH(dreg_map[1]);
		REFRESH(dreg_map[2]);
		REFRESH(dreg_map[3]);
		REFRESH(dreg_map[4]);
		REFRESH(dreg_map[5]);
		REFRESH(R11);
		REFRESH(R5);
	scamp5_kernel_end();
}


void load_dreg_image(DREG target_dreg,const uint8_t*image_buffer,uint16_t n_rows,uint16_t n_cols){
	const size_t row_bytes = n_cols/8;
    scamp5_dynamic_kernel_begin();
    	CLR(target_dreg);
	scamp5_dynamic_kernel_end();
    scamp5_draw_begin(target_dreg);
    for(int r=0;r<n_rows;r++){
		const uint8_t*row_array = &image_buffer[r*row_bytes];
		int u = 0;
		while(u<(n_cols/8)){
			if(row_array[u]==0x00){
				u += 1;
			}else
			if(row_array[u]==0xFF){
				int u0 = u;
				int u1 = u;
				u += 1;
				while(u<(n_cols/8)){
					if(row_array[u]==0xFF){
						u1 = u;
						u += 1;
					}else{
						break;
					}
				};
				scamp5_draw_rect(r,u0*8,r,u1*8 + 7);
			}else{
				uint8_t w = row_array[u];
				uint8_t m = 1;
				for(int c=u*8;c<(u*8 + 8);c++){
					if(w&m){
						scamp5_draw_pixel(r,c);
					}
					m <<= 1;
				}
				u += 1;
			}
		}
		if(r%16==0){
			refresh_dreg_storage();
		}
	}
	scamp5_draw_end();
}
