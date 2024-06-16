#include <scamp5.hpp>
#include <math.h>

using namespace SCAMP5_PE;

#ifndef IMG_TF_HELPER
#define IMG_TF_HELPER
namespace IMGTF
{
	unsigned char reverse_byte(unsigned char x);

	bool dreg_eql(dreg_t ra, dreg_t rb);

	double sin_approx3(double angle);
	double cos_approx3(double angle);
	double acos_approx3(double value);
	double tan_approx3(double angle);
}
#endif



