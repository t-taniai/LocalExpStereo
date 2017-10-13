#pragma once
#include <math.h>

struct Plane  {
	float a;
	float b;
	float c;
	float v;

	Plane(){}
	Plane(float a, float b, float c) : a(a), b(b), c(c), v(0){}
	Plane(float a, float b, float c, float y) : a(a), b(b), c(c), v(v){}

	static Plane CreatePlane(float nx, float ny, float nz, float z, float x, float y)
	{
		Plane p;
		p.a = -nx / nz;
		p.b = -ny / nz;
		p.c = z - p.a*x - p.b*y;
		p.v = 0;
		return p;
	}
	static Plane CreatePlane(float nx, float ny, float nz, float z, float x, float y, float v)
	{
		Plane p;
		p.a = -nx / nz;
		p.b = -ny / nz;
		p.c = z - p.a*x - p.b*y;
		p.v = v;
		return p;
	}
	static Plane CreatePlane(cv::Vec<float, 3> n, float z, float x, float y)
	{
		return CreatePlane(n[0], n[1], n[2], z, x, y);
	}

	static Plane CreatePlane(cv::Vec<float, 3> n, float z, float x, float y, float v)
	{
		return CreatePlane(n[0], n[1], n[2], z, x, y, v);
	}

	cv::Vec<float, 3> GetNormal() const
	{
		// Calc sqrt in double then cast to float.
		// Doing sqrt in float changes the results.
		float nz = float(1.0 / sqrt(1.0 + a*a + b*b));
		float nx = -a*nz;
		float ny = -b*nz;
		return cv::Vec<float, 3>(nx, ny, nz);
	}
	const float GetZ(float x, float y) const
	{
		return a*x + b*y + c;
	}
	const float GetZ(cv::Point p) const
	{
		return a*p.x + b*p.y + c;
	}

	bool operator ==(const Plane& in) const
	{
		return a == in.a && b == in.b && c == in.c && v == in.v;
	}

	bool operator !=(const Plane& in) const
	{
		return a != in.a || b != in.b || c != in.c || v != in.v;
	}


	cv::Vec<float, 4> toVec4() const
	{
		return cv::Vec<float, 4>(a, b, c, v);
	}
	cv::Scalar toScalar() const
	{
		return cv::Scalar(a, b, c, v);
	}
};

namespace cv
{
	template<> class DataType<Plane>
	{
	public:
		typedef float       value_type;
		typedef value_type  work_type;
		typedef value_type  channel_type;
		typedef value_type  vec_type;
		enum {
			generic_type = 0,
			depth = CV_32F,
			channels = sizeof(Plane) / sizeof(float),
			fmt = (int)'f',
			type = CV_MAKETYPE(depth, channels)
		};
	};
};