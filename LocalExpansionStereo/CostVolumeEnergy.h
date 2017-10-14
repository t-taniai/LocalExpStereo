#pragma once
#include "StereoEnergy.h"
#include "GuidedFilter.h"
#include "Plane.h"

class CostVolumeEnergy :
	public StereoEnergy
{
protected:
	std::unique_ptr<IJointFilter> filter[2];
	cv::Mat vol[2];
	int interpolate;


public:
	CostVolumeEnergy(const cv::Mat imL, const cv::Mat imR, const cv::Mat volL, const cv::Mat volR, Parameters params, float MAX_DISPARITY, float MIN_DISPARITY = 0, float MAX_VDISPARITY = 0)
		: StereoEnergy(imL, imR, params, MAX_DISPARITY, MIN_DISPARITY, MAX_VDISPARITY)
		, interpolate(1)
	{
		vol[0] = volL;
		vol[1] = volR;

		if (params.filterName == "BL")
		{
			filter[0] = std::make_unique<BilateralFilter>(imL, params.windR, params.filter_param1);
			filter[1] = std::make_unique<BilateralFilter>(imR, params.windR, params.filter_param1);
		}
		else if (params.filterName == "GF")
		{
			filter[0] = std::make_unique<FastGuidedImageFilter<double>>(imL, params.windR / 2, params.filter_param1, 1.0 / 255);
			filter[1] = std::make_unique<FastGuidedImageFilter<double>>(imR, params.windR / 2, params.filter_param1, 1.0 / 255);
		}
		else if (params.filterName == "GFfloat")
		{
			filter[0] = std::make_unique<FastGuidedImageFilter<float>>(imL, params.windR / 2, params.filter_param1, 1.0 / 255);
			filter[1] = std::make_unique<FastGuidedImageFilter<float>>(imR, params.windR / 2, params.filter_param1, 1.0 / 255);
		}
		else //if (params.filterName == "")
		{
			filter[0] = nullptr;// std::make_unique<GuidedImageFilter>(imL, params.windR / 2, params.filter_param1);
			filter[1] = nullptr;// std::make_unique<GuidedImageFilter>(imR, params.windR / 2, params.filter_param1);
		}
	}

	void setInterpolationMethod(int none_lin_quad)
	{
		interpolate = none_lin_quad;
	}

	~CostVolumeEnergy()
	{
	}


	void ComputeUnaryPotentialWithoutCheck(const cv::Rect& filterRect, const cv::Rect& targetRect, const cv::Mat& costs, const Plane& plane, Reusable& reusable = Reusable(), int mode = 0) const override
	{
		if (reusable.pIL.empty())
		{
			reusable.pIL = cv::Mat_<float>(filterRect.size());
			if (params.filterName != "")
				reusable.filter = filter[mode]->createSubregionFilter(filterRect);
		}

		int y0 = filterRect.y +filterRect.height;
		int x0 = filterRect.x +filterRect.width;
		int D = vol[mode].size.p[0];
		int D0 = int(-MIN_DISPARITY);

		if (interpolate == 1)
		for (int y = filterRect.y; y < y0; y++)
		{
			auto pC = reusable.pIL.ptr<float>(y - filterRect.y);
			float d_base = plane.b * y + plane.c;
			for (int x = filterRect.x; x < x0; x++)
			{
				float d = plane.a * x + d_base;
				float C;
				if (d < MIN_DISPARITY) C = vol[mode].at<float>(0, y, x);
				else if (d >= MAX_DISPARITY) C = vol[mode].at<float>(D - 1, y, x);
				else if (isnan<float>(d) || isinf<float>(d)) C = COST_FOR_INVALID;
				else
				{
					int d0 = int(d) + D0;
					int d1 = d0 + 1;
					float f1 = d - std::floor(d);
					float f0 = 1.0f - f1;
					if (d1 >= D || d0 < 0){
						printf("%lf %lf %lf %lf %lf %d %d %d %d %d %d\n", MAX_DISPARITY, MIN_DISPARITY, d, f1, f0, d0, d1, D, D0, y, x);
						C = COST_FOR_INVALID;
					}
					else{
						C = f0 * vol[mode].at<float>(d0, y, x) + f1 * vol[mode].at<float>(d1, y, x);
					}
				}

				pC[x - filterRect.x] = std::min(C, params.th_col);
			}
		}
		else if (interpolate == 0)
		for (int y = filterRect.y; y < y0; y++)
		{
			auto pC = reusable.pIL.ptr<float>(y - filterRect.y);
			float d_base = plane.b * y + plane.c;
			for (int x = filterRect.x; x < x0; x++)
			{
				int d = (int)(plane.a * x + d_base + 0.5) + D0;
				float C;
				if (d < 0)
					C = vol[mode].at<float>(0, y, x);
				else if (d >= D)
					C = vol[mode].at<float>(D - 1, y, x);
				else if (isnan<float>(d_base) || isinf<float>(d_base))
					C = COST_FOR_INVALID;
				else 
					C = vol[mode].at<float>(d, y, x);

				pC[x - filterRect.x] = std::min(C, params.th_col);
			}
		}
		else if (interpolate == 2)
		// a = y1. / (d1 - d2). / (d1 - d3);
		// b = y2. / (d2 - d1). / (d2 - d3);
		// c = y3. / (d3 - d1). / (d3 - d2);
		// 
		// polynom: r*d ^ 2 + p*d + q
		// r = a + b + c;
		// p = -(a.*(d2 + d3) + b.*(d1 + d3) + c.*(d1 + d2));
		// q = a.*d2.*d3 + b.*d1.*d3 + c.*d1.*d2;
		for (int y = filterRect.y; y < y0; y++)
		{
			auto pC = reusable.pIL.ptr<float>(y - filterRect.y);
			float d_base = plane.b * y + plane.c;
			for (int x = filterRect.x; x < x0; x++)
			{
				float d = plane.a * x + d_base;
				int d2 = int(d + 0.5) + D0;
				float C;
				if (d2 < 0)
					C = vol[mode].at<float>(0, y, x);
				else if (d2 >= D)
					C = vol[mode].at<float>(D - 1, y, x);
				else if (isnan<float>(d) || isinf<float>(d))
					C = COST_FOR_INVALID;
				else
				{
					int d3 = std::min(d2 + 1, D-1);
					int d1 = std::max(d2 - 1, 0);
					float y1 = vol[mode].at<float>(d1, y, x);
					float y2 = vol[mode].at<float>(d2, y, x);
					float y3 = vol[mode].at<float>(d3, y, x);

					float rd1 = (float)d1, rd2 = (float)d2, rd3 = (float)d3;
					float a = y1 / (rd1 - rd2) / (rd1 - rd3);
					float b = y2 / (rd2 - rd1) / (rd2 - rd3);
					float c = y3 / (rd3 - rd1) / (rd3 - rd2);

					float r = a + b + c;
					float p = -(a*(rd2 + rd3) + b*(rd1 + rd3) + c*(rd1 + rd2));
					float q = a*rd2*rd3 + b*rd1*rd3 + c*rd1*rd2;

					d = d + (float)D0;
					C = r*d*d + p*d + q;
				}

				pC[x - filterRect.x] = std::min(C, params.th_col);
			}
		}

		cv::Rect subrect = targetRect - filterRect.tl();
		if (params.filterName != "")
			reusable.filter->filter(reusable.pIL)(subrect).copyTo(costs(subrect));
		else
			reusable.pIL(subrect).copyTo(costs(subrect));
	}

	void ComputeUnaryPotential(const cv::Rect& filterRect, const cv::Rect& targetRect, const cv::Mat& costs, const Plane& plane, Reusable& reusable = Reusable(), int mode = 0) const override
	{
		ComputeUnaryPotentialWithoutCheck(filterRect, targetRect, costs, plane, reusable, mode);

		cv::Rect subrect = targetRect - filterRect.tl();
		cv::Mat validMask = IsValiLabel(plane, targetRect);
		costs(subrect).setTo(cv::Scalar(COST_FOR_INVALID), ~validMask);
	}
};

