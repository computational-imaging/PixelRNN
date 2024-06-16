#include <scamp5.hpp>
#include <math.h>

using namespace SCAMP5_PE;

#ifndef IMG_SCALING
#define IMG_SCALING

namespace IMGTF
{
	namespace SCALING
	{
		namespace ANALOG
		{
			void STEP_SCALE_UP_F(int step_number);
			void STEP_SCALE_DOWN_F(int step_number);

			void STEP_SCALE_UPY_F(int step_number);
			void STEP_SCALE_DOWNY_F(int step_number);

			void STEP_SCALE_UPX_F(int step_number);
			void STEP_SCALE_DOWNX_F(int step_number);

			///////////////////////////////////////////

			int STEP_SCALE(areg_t reg,int current_scaling_value, bool scale_down);
			void SCALE_Y(areg_t reg,int scaling_mag, bool scale_down);
			void SCALE_X(areg_t reg,int scaling_mag, bool scale_down);
			void SCALE(areg_t reg,int scaling_mag, bool scale_down);

			///////////////////////////////////////////

			void HALF_SCALE(areg_t reg);
		}
	}
}
#endif
