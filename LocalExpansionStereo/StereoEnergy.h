#pragma once

#include "Plane.h"
#include "GuidedFilter.h"
#include <vector>
#include <memory>
#include <math.h>
#include <fstream>
#include <opencv2/opencv.hpp>
//#include <opencv2/ximgproc.hpp>


struct Parameters
{
	float alpha;
	float omega;
	float th_grad;
	float th_col;
	float lambda;
	float th_smooth;
	float epsilon;
	float filter_param1;
	int windR;
	int neighborNum;
	std::string filterName; // "BF" or "GF" or ""

	Parameters(float lambda = 20, int windR = 20, std::string filterName = "BF", float filter_param1 = 10)
		: alpha(0.9)
		, omega(10.0)
		, th_grad(2.0)
		, th_col(10.0)
		, lambda(lambda)
		, th_smooth(1.0)
		, epsilon(0.01)
		, windR(windR)
		, neighborNum(8)
		, filterName(filterName)
		, filter_param1(filter_param1)
	{}
};

class StereoEnergy
{
public:
	static const int COST_FOR_INVALID = 1000000;

	enum {
		NB_LE = 0,
		NB_GE = 1,
		NB_EL = 2,
		NB_EG = 3,
		NB_LL = 4,
		NB_GL = 5,
		NB_LG = 6,
		NB_GG = 7
	};

	std::vector<cv::Point> neighbors;

protected:
	const int width;
	const int height;
	const float MAX_DISPARITY;
	const float MIN_DISPARITY;
	const float MAX_VDISPARITY;
	cv::Mat I[2];

	std::vector<cv::Mat> smoothnessCoeff[2];

	const cv::Rect imageDomain;
	cv::Mat coordinates_m;
	cv::Mat coordinates;

	const int M; // neighbor range (should be 1 for 8-neighbors)

public:
	Parameters params;
	
	StereoEnergy(const cv::Mat imL, const cv::Mat imR, Parameters params, float MAX_DISPARITY, float MIN_DISPARITY = 0, float MAX_VDISPARITY = 0)
		: width(imL.cols)
		, height(imL.rows)
		, MAX_DISPARITY(MAX_DISPARITY)
		, MIN_DISPARITY(MIN_DISPARITY)
		, MAX_VDISPARITY(MAX_VDISPARITY)
		, params(params)
		, imageDomain(0, 0, width, height)
		, M(1)
	{
		const cv::Size size = cv::Size(width, height);
		I[0] = cv::Mat(size, CV_MAKE_TYPE(CV_32F, 3));
		I[1] = cv::Mat(size, CV_MAKE_TYPE(CV_32F, 3));

		coordinates_m = cv::Mat::zeros(imL.rows + 2 * M, imL.cols + 2 * M, CV_MAKE_TYPE(cv::DataType<float>::depth, 4));
		coordinates = coordinates_m(cv::Rect(M, M, imL.cols, imL.rows));

		imL.convertTo(I[0], I[0].type());
		imR.convertTo(I[1], I[1].type());

		neighbors.resize(params.neighborNum);
		neighbors[0] = cv::Point(-1, +0);
		neighbors[1] = cv::Point(+1, +0);
		neighbors[2] = cv::Point(+0, -1);
		neighbors[3] = cv::Point(+0, +1);
		if (params.neighborNum >= 8)
		{
			neighbors[4] = cv::Point(-1, -1);
			neighbors[5] = cv::Point(+1, -1);
			neighbors[6] = cv::Point(-1, +1);
			neighbors[7] = cv::Point(+1, +1);
		}

		for (int y = 0; y < imL.rows; y++){
			for (int x = 0; x < imL.cols; x++){
				coordinates.at<cv::Vec<float, 4>>(y, x) = cv::Vec<float, 4>(x, y, 1.0, 0);
			}
		}
		initSmoothnessCoeff();
	}

	Plane createRandomLabel(cv::Point s) const
	{
		float zs = cv::theRNG().uniform(MIN_DISPARITY, MAX_DISPARITY);
		float vs = MAX_VDISPARITY != 0 ? cv::theRNG().uniform(-MAX_VDISPARITY, MAX_VDISPARITY) : 0;

		// CV_PI/4‚Å a = 1, b = 0 ‚Ü‚Å
		cv::Vec3d n = cvutils::getRandomUnitVector(CV_PI / 3);
		//cv::Vec3d n = cvutils::getRandomUnitVector(0.5);
		return Plane::CreatePlane(n, zs, s.x, s.y, vs);
	}

	void initSmoothnessCoeff()
	{
		for (int m = 0; m < 2; m++)
		{
			cv::Mat I_m;
			cv::copyMakeBorder(I[m], I_m, M, M, M, M, cv::BORDER_CONSTANT, cv::Scalar::all(0));

			cv::Rect rect_ee = cv::Rect(M, M, I[m].cols, I[m].rows);
			cv::Mat IL_ee = I_m(rect_ee);
			smoothnessCoeff[m].resize(neighbors.size());
			for (int i = 0; i < neighbors.size(); i++)
			{
				cv::Mat IL_nb = I_m(rect_ee + neighbors[i]).clone();
				absdiff(IL_nb, IL_ee, IL_nb);
				cv::exp(-cvutils::channelSum(IL_nb) / params.omega, smoothnessCoeff[m][i]);
				smoothnessCoeff[m][i] = cv::max(params.epsilon, smoothnessCoeff[m][i]);

				// set invalid pairwise terms to zero
				if (neighbors[i].x < 0)
					smoothnessCoeff[m][i].colRange(0, -neighbors[i].x) = 0;
				if (neighbors[i].x > 0)
					smoothnessCoeff[m][i].colRange(width - neighbors[i].x, width) = 0;
				if (neighbors[i].y < 0)
					smoothnessCoeff[m][i].rowRange(0, -neighbors[i].y) = 0;
				if (neighbors[i].y > 0)
					smoothnessCoeff[m][i].rowRange(height - neighbors[i].y, height) = 0;

				cv::Mat tmp;
				cv::copyMakeBorder(smoothnessCoeff[m][i], tmp, M, M, M, M, cv::BORDER_CONSTANT, cv::Scalar::all(0));
				smoothnessCoeff[m][i] = tmp;
			}
		}
	}

	double computeSmoothnessCost(cv::Mat labeling_m, int mode = 0) const
	{
		cv::Rect rect_ee = imageDomain + cv::Point(M, M);
		cv::Mat label_ee = labeling_m(rect_ee);
		cv::Mat coord_ee = coordinates_m(rect_ee);
		cv::Mat disp0_of_ee_at_ee = cvutils::channelDot(label_ee, coord_ee);
		if (disp0_of_ee_at_ee.depth() != CV_32F)
			disp0_of_ee_at_ee.convertTo(disp0_of_ee_at_ee, CV_32F);
		
		cv::Mat sumCost = cv::Mat::zeros(imageDomain.size(), CV_32F);
		for (int i = 0; i < neighbors.size(); i++)
		{
			if (neighbors[i].y*width + neighbors[i].x < 0)
				continue;

			cv::Rect rect_nb = rect_ee + neighbors[i];
			cv::Mat label_nb = labeling_m(rect_nb);
			cv::Mat coord_nb = coordinates_m(rect_nb);
			cv::Mat disp0_of_nb_at_ee = cvutils::channelDot(label_nb, coord_ee);
			cv::Mat disp0_of_ee_at_nb = cvutils::channelDot(label_ee, coord_nb);
			cv::Mat disp0_of_nb_at_nb = cvutils::channelDot(label_nb, coord_nb);
			if (labeling_m.depth() != CV_32F)
			{
				disp0_of_nb_at_ee.convertTo(disp0_of_nb_at_ee, CV_32F);
				disp0_of_ee_at_nb.convertTo(disp0_of_ee_at_nb, CV_32F);
				disp0_of_nb_at_nb.convertTo(disp0_of_nb_at_nb, CV_32F);
			}
			cv::Mat cost00_nb = cv::abs(disp0_of_ee_at_ee - disp0_of_nb_at_ee) + cv::abs(disp0_of_ee_at_nb - disp0_of_nb_at_nb);
			cv::threshold(cost00_nb, cost00_nb, params.th_smooth, 0, cv::THRESH_TRUNC);
			cost00_nb = cost00_nb.mul(smoothnessCoeff[mode][i](rect_ee), params.lambda);

			cv::add(sumCost, cost00_nb, sumCost);
			//cv::add(sumCost, smoothnessCoeffL[i](rect_ee), sumCost);
		}

		return cv::sum(sumCost)[0];
	}

	double computeSmoothnessCost_(cv::Mat labeling, int mode = 0)
	{
		double sum = 0;
		for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
		{
			cv::Point ps(x, y);
			int s = ps.y*width + ps.x;
			for (int k = 0; k < neighbors.size(); k++){
				cv::Point pt = ps + neighbors[k];
				int t = pt.y*width + pt.x;

				if (s > t || imageDomain.contains(pt) == false) continue;
				Plane ls = labeling.at<Plane>(ps);
				Plane lt = labeling.at<Plane>(pt);
				sum += computeSmoothnessTerm(ls, lt, ps, pt, mode);
				//sum += computePatchWeight(ps, pt, mode);
			}
		}
		return sum;
	}

	float computeSmoothnessTerm(Plane ls, Plane lt, cv::Point ps, int neighborId, int mode = 0) const
	{
		cv::Point pt = ps + neighbors[neighborId];
		return smoothnessCoeff[mode][neighborId].at<float>(ps + cv::Point(M, M))
			* std::min(fabs(ls.GetZ(ps) - lt.GetZ(ps)) + fabs(ls.GetZ(pt) - lt.GetZ(pt)), params.th_smooth) * params.lambda;
	}

	float computeSmoothnessTerm(Plane ls, Plane lt, cv::Point ps, cv::Point pt, int mode = 0) const
	{
		return std::max(computePatchWeight(ps, pt, mode), params.epsilon)
			* std::min(fabs(ls.GetZ(ps) - lt.GetZ(ps)) + fabs(ls.GetZ(pt) - lt.GetZ(pt)), params.th_smooth) * params.lambda;
	}

	float computeSmoothnessTermWithoutConst(Plane ls, Plane lt, cv::Point ps, cv::Point pt) const
	{
		return std::min(fabs(ls.GetZ(ps) - lt.GetZ(ps)) + fabs(ls.GetZ(pt) - lt.GetZ(pt)), params.th_smooth);
	}

	float computeSmoothnessTermConst(cv::Point ps, cv::Point pt, int mode = 0) const
	{
		return std::max(computePatchWeight(ps, pt, mode), params.epsilon) * params.lambda;
	}
	float computeSmoothnessTermConst(cv::Point s, int neighborId, int mode = 0) const
	{
		return smoothnessCoeff[mode][neighborId].at<float>(s + cv::Point(M, M));
	}
	float computePatchWeight(cv::Point s, cv::Point t, int mode = 0) const
	{
		const cv::Mat& I = this->I[mode];
		cv::Vec3f dI = I.at<cv::Vec3f>(s) -I.at<cv::Vec3f>(t);
		float absdiff = fabs(dI[0]) + fabs(dI[1]) + fabs(dI[2]);
		return std::exp(-absdiff / params.omega);
	}

	cv::Rect getRectWithMargin() const
	{
		return cv::Rect(0, 0, imageDomain.width + 2 * M, imageDomain.height + 2 * M);
	}

	cv::Rect getRectWithoutMargin() const
	{
		return imageDomain + cv::Point(M, M);
	}

	cv::Mat computeDisparities(const cv::Mat& labeling) const 
	{
		return cvutils::channelDot(coordinates, labeling);
	}

	cv::Mat computeNormalMap(const cv::Mat& labeling) const
	{
		cv::Mat normalMap;
		std::vector<cv::Mat> abc;
		cv::split(labeling, abc);
		cv::Mat nzMap = abc[0].mul(abc[0]) + abc[1].mul(abc[1]) + 1.0;
		cv::sqrt(nzMap, nzMap);
		cv::divide(1.0, nzMap, nzMap);
		abc[0] = (abc[0].mul(nzMap, -1.0) + 1.0) / 2.0;
		abc[1] = (abc[1].mul(nzMap, -1.0) + 1.0) / 2.0;
		abc[2] = abc[0];
		abc[0] = nzMap;
		abc.resize(3);
		cv::merge(abc, normalMap);
		return normalMap;
	}

	void computeLocalSmoothnessTerms10(const cv::Mat& labeling0_m, const cv::Mat& labeling1_m, cv::Rect region, cv::Mat& cost10, int mode = 0) const
	{
		cv::Rect rect_ee = cv::Rect(M + region.x, M + region.y, region.width, region.height);
		cv::Mat label1_ee = labeling1_m(rect_ee);
		cv::Mat coord_ee = coordinates_m(rect_ee);
		cv::Mat disp1_of_ee_at_ee = cvutils::channelDot(label1_ee, coord_ee);
		if (disp1_of_ee_at_ee.depth() != CV_32F)
		{
			disp1_of_ee_at_ee.convertTo(disp1_of_ee_at_ee, CV_32F);
		}

		cv::Mat sumCost = cv::Mat::zeros(imageDomain.size(), CV_32F);
		for (int i = 0; i < neighbors.size(); i++)
		{
			cv::Rect rect_le = rect_ee + neighbors[i];
			cv::Mat label0_le = labeling0_m(rect_le);
			cv::Mat coord_le = coordinates_m(rect_le);

			cv::Mat disp0_of_le_at_ee = cvutils::channelDot(label0_le, coord_ee);
			cv::Mat disp0_of_le_at_le = cvutils::channelDot(label0_le, coord_le);
			cv::Mat disp1_of_ee_at_le = cvutils::channelDot(label1_ee, coord_le);

			if (disp0_of_le_at_ee.depth() != CV_32F)
			{
				disp0_of_le_at_ee.convertTo(disp0_of_le_at_ee, CV_32F);
				disp0_of_le_at_le.convertTo(disp0_of_le_at_le, CV_32F);
				disp1_of_ee_at_le.convertTo(disp1_of_ee_at_le, CV_32F);
			}

			cv::Mat smoothnessCoeffL_nb = smoothnessCoeff[mode][i](rect_ee);

			cv::Mat cost10_bb = cv::abs(disp1_of_ee_at_ee - disp0_of_le_at_ee) + cv::abs(disp1_of_ee_at_le - disp0_of_le_at_le);
			cv::threshold(cost10_bb, cost10_bb, params.th_smooth, 0, cv::THRESH_TRUNC);
			cost10_bb = cost10_bb.mul(smoothnessCoeffL_nb, params.lambda);
			cv::add(sumCost, cost10_bb, sumCost);
		}
		cost10 = sumCost;
	}


	void computeSmoothnessTermsFusion(const cv::Mat& labeling0_m, const cv::Mat& labeling1_m, cv::Rect region, std::vector<cv::Mat>& cost00, std::vector<cv::Mat>& cost01, std::vector<cv::Mat>& cost10, std::vector<cv::Mat>& cost11, bool onlyForward = false, int mode = 0) const
	{
		cv::Rect rect_ee = cv::Rect(M + region.x, M + region.y, region.width, region.height);
		cv::Mat label0_ee = labeling0_m(rect_ee);
		cv::Mat label1_ee = labeling1_m(rect_ee);
		cv::Mat coord_ee = coordinates_m(rect_ee);
		cv::Mat disp0_of_ee_at_ee = cvutils::channelDot(label0_ee, coord_ee);
		cv::Mat disp1_of_ee_at_ee = cvutils::channelDot(label1_ee, coord_ee);
		if (disp0_of_ee_at_ee.depth() != CV_32F)
		{
			disp0_of_ee_at_ee.convertTo(disp0_of_ee_at_ee, CV_32F);
			disp1_of_ee_at_ee.convertTo(disp1_of_ee_at_ee, CV_32F);
		}

		cost00 = std::vector<cv::Mat>(neighbors.size());
		cost01 = std::vector<cv::Mat>(neighbors.size());
		cost10 = std::vector<cv::Mat>(neighbors.size());
		cost11 = std::vector<cv::Mat>(neighbors.size());
		for (int i = 0; i < neighbors.size(); i++)
		{
			if (onlyForward && (neighbors[i].y * width + neighbors[i].x <= 0))
				continue;

			cv::Rect rect_le = rect_ee + neighbors[i];
			cv::Mat label0_le = labeling0_m(rect_le);
			cv::Mat label1_le = labeling1_m(rect_le);
			cv::Mat coord_le = coordinates_m(rect_le);

			cv::Mat disp0_of_le_at_ee = cvutils::channelDot(label0_le, coord_ee);
			cv::Mat disp0_of_ee_at_le = cvutils::channelDot(label0_ee, coord_le);
			cv::Mat disp0_of_le_at_le = cvutils::channelDot(label0_le, coord_le);
			cv::Mat disp1_of_le_at_ee = cvutils::channelDot(label1_le, coord_ee);
			cv::Mat disp1_of_ee_at_le = cvutils::channelDot(label1_ee, coord_le);
			cv::Mat disp1_of_le_at_le = cvutils::channelDot(label1_le, coord_le);

			if (disp0_of_le_at_ee.depth() != CV_32F)
			{
				disp0_of_le_at_ee.convertTo(disp0_of_le_at_ee, CV_32F);
				disp0_of_ee_at_le.convertTo(disp0_of_ee_at_le, CV_32F);
				disp0_of_le_at_le.convertTo(disp0_of_le_at_le, CV_32F);
				disp1_of_le_at_ee.convertTo(disp1_of_le_at_ee, CV_32F);
				disp1_of_ee_at_le.convertTo(disp1_of_ee_at_le, CV_32F);
				disp1_of_le_at_le.convertTo(disp1_of_le_at_le, CV_32F);
			}

			cv::Mat smoothnessCoeffL_nb = smoothnessCoeff[mode][i](rect_ee);

			cost00[i] = cv::abs(disp0_of_ee_at_ee - disp0_of_le_at_ee) + cv::abs(disp0_of_ee_at_le - disp0_of_le_at_le);
			cv::threshold(cost00[i], cost00[i], params.th_smooth, 0, cv::THRESH_TRUNC);
			cost00[i] = cost00[i].mul(smoothnessCoeffL_nb, params.lambda);

			cost01[i] = cv::abs(disp0_of_ee_at_ee - disp1_of_le_at_ee) + cv::abs(disp0_of_ee_at_le - disp1_of_le_at_le);
			cv::threshold(cost01[i], cost01[i], params.th_smooth, 0, cv::THRESH_TRUNC);
			cost01[i] = cost01[i].mul(smoothnessCoeffL_nb, params.lambda);

			cost10[i] = cv::abs(disp1_of_ee_at_ee - disp0_of_le_at_ee) + cv::abs(disp1_of_ee_at_le - disp0_of_le_at_le);
			cv::threshold(cost10[i], cost10[i], params.th_smooth, 0, cv::THRESH_TRUNC);
			cost10[i] = cost10[i].mul(smoothnessCoeffL_nb, params.lambda);

			cost11[i] = cv::abs(disp1_of_ee_at_ee - disp1_of_le_at_ee) + cv::abs(disp1_of_ee_at_le - disp1_of_le_at_le);
			cv::threshold(cost11[i], cost11[i], params.th_smooth, 0, cv::THRESH_TRUNC);
			cost11[i] = cost11[i].mul(smoothnessCoeffL_nb, params.lambda);
		}
	}

	void computeSmoothnessTermsExpansion(const cv::Mat& labeling0_m, Plane label1, cv::Rect region, std::vector<cv::Mat>& cost00, std::vector<cv::Mat>& cost01, std::vector<cv::Mat>& cost10, bool onlyForward = false, int mode = 0) const
	{
		cv::Rect rect_ee = cv::Rect(M + region.x, M + region.y, region.width, region.height);
		cv::Mat label0_ee = labeling0_m(rect_ee);
		cv::Mat coord_ee = coordinates_m(rect_ee);
		cv::Scalar sc = label1.toScalar();
		cv::Mat disp0_of_ee_at_ee = cvutils::channelDot(label0_ee, coord_ee);
		cv::Mat disp1_at_ee = cvutils::channelSum(coord_ee.mul(sc));
		if (disp0_of_ee_at_ee.depth() != CV_32F)
		{
			disp0_of_ee_at_ee.convertTo(disp0_of_ee_at_ee, CV_32F);
			disp1_at_ee.convertTo(disp1_at_ee, CV_32F);
		}

		cost00 = std::vector<cv::Mat>(neighbors.size());
		cost01 = std::vector<cv::Mat>(neighbors.size());
		cost10 = std::vector<cv::Mat>(neighbors.size());
		for (int i = 0; i < neighbors.size(); i++)
		{
			if (onlyForward && (neighbors[i].y * width + neighbors[i].x <= 0))
				continue;

			cv::Rect rect_le = rect_ee + neighbors[i];
			cv::Mat label0_le = labeling0_m(rect_le);
			cv::Mat coord_le = coordinates_m(rect_le);

			cv::Mat disp0_of_le_at_ee = cvutils::channelDot(label0_le, coord_ee);
			cv::Mat disp0_of_ee_at_le = cvutils::channelDot(label0_ee, coord_le);
			cv::Mat disp0_of_le_at_le = cvutils::channelDot(label0_le, coord_le);
			cv::Mat disp1_at_le = cvutils::channelSum(coord_le.mul(sc));

			if (disp0_of_le_at_ee.depth() != CV_32F)
			{
				disp0_of_le_at_ee.convertTo(disp0_of_le_at_ee, CV_32F);
				disp0_of_ee_at_le.convertTo(disp0_of_ee_at_le, CV_32F);
				disp0_of_le_at_le.convertTo(disp0_of_le_at_le, CV_32F);
				disp1_at_le.convertTo(disp1_at_le, CV_32F);
			}

			cv::Mat smoothnessCoeffL_nb = smoothnessCoeff[mode][i](rect_ee);

			cost00[i] = cv::abs(disp0_of_ee_at_ee - disp0_of_le_at_ee) + cv::abs(disp0_of_ee_at_le - disp0_of_le_at_le);
			cv::threshold(cost00[i], cost00[i], params.th_smooth, 0, cv::THRESH_TRUNC);
			cost00[i] = cost00[i].mul(smoothnessCoeffL_nb, params.lambda);

			cost01[i] = cv::abs(disp0_of_ee_at_ee - disp1_at_ee) + cv::abs(disp0_of_ee_at_le - disp1_at_le);
			cv::threshold(cost01[i], cost01[i], params.th_smooth, 0, cv::THRESH_TRUNC);
			cost01[i] = cost01[i].mul(smoothnessCoeffL_nb, params.lambda);

			cost10[i] = cv::abs(disp1_at_ee - disp0_of_le_at_ee) + cv::abs(disp1_at_le - disp0_of_le_at_le);
			cv::threshold(cost10[i], cost10[i], params.th_smooth, 0, cv::THRESH_TRUNC);
			cost10[i] = cost10[i].mul(smoothnessCoeffL_nb, params.lambda);
		}
	}

	// Avoid extreme labels
	bool IsValiLabel(Plane label, cv::Point pos) const
	{
		float ds = label.GetZ(pos);
		float a5 = label.a * 5;
		float b5 = label.b * 5;
		float d;

		return (
			ds >= MIN_DISPARITY && ds <= MAX_DISPARITY
			&& ((d = ds + a5 + b5) >= MIN_DISPARITY) && d <= MAX_DISPARITY
			&& ((d = ds + a5 - b5) >= MIN_DISPARITY) && d <= MAX_DISPARITY
			&& ((d = ds - a5 + b5) >= MIN_DISPARITY) && d <= MAX_DISPARITY
			&& ((d = ds - a5 - b5) >= MIN_DISPARITY) && d <= MAX_DISPARITY
		);
	}

	// Avoid extreme labels
	cv::Mat IsValiLabel(Plane label, cv::Rect pos) const
	{
		if (pos.width == 1 && pos.height == 1)
		{
			if(IsValiLabel(label, pos.tl())) cv::Mat_<uchar>(1, 1, 255);
			else cv::Mat_<uchar>::zeros(1, 1);
		}
		else
		{
			float a5 = label.a * 5;
			float b5 = label.b * 5;
			const cv::Scalar lower(MIN_DISPARITY);
			const cv::Scalar upper(MAX_DISPARITY);
			cv::Mat mask = cv::Mat_<uchar>(pos.size(), (uchar)255);

			cv::Mat disp = cvutils::channelSum(coordinates(pos).mul(label.toScalar()));
			for (int y = 0; y < mask.rows; y++)
			{
				for (int x = 0; x < mask.cols; x++)
				{
					float ds = disp.at<float>(y, x);
					float d;

					mask.at<uchar>(y, x) = (ds >= MIN_DISPARITY && ds <= MAX_DISPARITY
						&& ((d = ds + a5 + b5) >= MIN_DISPARITY) && d <= MAX_DISPARITY
						&& ((d = ds + a5 - b5) >= MIN_DISPARITY) && d <= MAX_DISPARITY
						&& ((d = ds - a5 + b5) >= MIN_DISPARITY) && d <= MAX_DISPARITY
						&& ((d = ds - a5 - b5) >= MIN_DISPARITY) && d <= MAX_DISPARITY
						) ? (uchar)255 : 0;
				}
			}
			return mask;
		}
	}

	virtual ~StereoEnergy(void)
	{
	}

	struct Reusable {
		cv::Mat pIL;
		cv::Mat pIR;
		cv::Rect filterRect;
		cv::Mat kernel;
		std::shared_ptr<IJointFilter> filter;
		//cv::Ptr<cv::ximgproc::GuidedFilter> cvfilter;
	};
	
	virtual void ComputeUnaryPotentialWithoutCheck(const cv::Rect& filterRect, const cv::Rect& targetRect, const cv::Mat& costs, const Plane& plane, Reusable& reusable = Reusable(), int mode = 0) const {};
	virtual void ComputeUnaryPotential(const cv::Rect& filterRect, const cv::Rect& targetRect, const cv::Mat& costs, const Plane& plane, Reusable& reusabl = Reusable(), int mode = 0) const {};
};

class NaiveStereoEnergy : public StereoEnergy
{
protected:
	std::unique_ptr<IJointFilter> filter[2];
	cv::Mat ExI[2];
	float thresh_color;
	float thresh_gradient;

public:
	NaiveStereoEnergy(const cv::Mat imL, const cv::Mat imR, Parameters params, float MAX_DISPARITY, float MIN_DISPARITY = 0, float MAX_VDISPARITY = 0)
		: StereoEnergy(imL, imR, params, MAX_DISPARITY, MIN_DISPARITY, MAX_VDISPARITY)
	{
		auto size = imL.size();
		const cv::Mat gray = cv::Mat(size, CV_MAKE_TYPE(CV_32F, 1));

		const int ksize = 1;
		const double factor = 0.5;

		for (int m = 0; m < 2; m++)
		{
			cv::Mat GX;
			//cv::Mat GY;
			cv::cvtColor(I[m], gray, cv::COLOR_BGR2GRAY);
			GX = cv::Mat(size, CV_MAKE_TYPE(CV_32F, 1));
			//GY = cv::Mat(size, CV_MAKE_TYPE(CV_32F, 1));
			cv::Sobel(gray, GX, CV_32F, 1, 0, ksize, factor, 0, cv::BORDER_REPLICATE);
			//cv::Sobel(gray, GY, CV_32F, 0, 1, ksize, factor, 0, cv::BORDER_REPLICATE);

			std::vector<cv::Mat> bands;
			cv::split(I[m] * (1.0 - params.alpha), bands);
			bands.push_back(GX * params.alpha);
			//bands.push_back(GY);
			cv::merge(bands, ExI[m]);
		}
		thresh_color = params.th_col * (1.0 - params.alpha);
		thresh_gradient = params.th_grad * params.alpha;

		if (params.filterName == "BF")
		{
			filter[0] = std::make_unique<BilateralFilter>(imL, params.windR, params.filter_param1);
			filter[1] = std::make_unique<BilateralFilter>(imR, params.windR, params.filter_param1);
		}
		else if (params.filterName == "GF")
		{
			// Running time slightly improves by changing double to float (but results also slightly change).
			filter[0] = std::make_unique<FastGuidedImageFilter<double>>(imL, params.windR / 2, params.filter_param1, 1.0 / 255);
			filter[1] = std::make_unique<FastGuidedImageFilter<double>>(imR, params.windR / 2, params.filter_param1, 1.0 / 255);
		}
		else //if (params.filterName == "")
		{
			filter[0] = nullptr;
			filter[1] = nullptr;
		}
	}

	virtual ~NaiveStereoEnergy()
	{
	}


	void ComputeUnaryPotentialWithoutCheck(const cv::Rect& filterRect, const cv::Rect& targetRect, const cv::Mat& costs, const Plane& plane, Reusable& reusable = Reusable(), int mode = 0) const override
	{
		if (reusable.pIL.empty())
		{
			reusable.pIL = ExI[mode](filterRect);
			reusable.pIR = cv::Mat(filterRect.size(), ExI[mode].type());
			if (params.filterName != "")
				reusable.filter = filter[mode]->createSubregionFilter(filterRect);
		}

		cv::Point2f src_pnt[3], dst_pnt[3];
		float sign = mode ? -1 : 1;

		int x00 = filterRect.x;
		int y00 = filterRect.y;
		int x11 = filterRect.x + filterRect.width;
		int y11 = filterRect.y + filterRect.height;
		dst_pnt[0] = cv::Point2f(0, 0);
		dst_pnt[1] = cv::Point2f(0, y11 - y00);
		dst_pnt[2] = cv::Point2f(x11 - x00, 0);
		src_pnt[0].x = (float)x00 - sign*plane.GetZ(x00, y00);
		src_pnt[0].y = (float)y00;
		src_pnt[1].x = (float)x00 - sign*plane.GetZ(x00, y11);
		src_pnt[1].y = (float)y11;
		src_pnt[2].x = (float)x11 - sign*plane.GetZ(x11, y00);
		src_pnt[2].y = (float)y00;
		if (plane.v != 0)
		{
			src_pnt[0].y += plane.v;
			src_pnt[1].y += plane.v;
			src_pnt[2].y += plane.v;
		}

		cv::Mat mat = cv::getAffineTransform(src_pnt, dst_pnt);

		cv::warpAffine(ExI[1 - mode], reusable.pIR, mat, filterRect.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
		cv::Mat rawCosts = cv::Mat_<float>(reusable.pIL.size());
		float _alpha = 1.0f - params.alpha;
		for(int y = 0; y < reusable.pIL.rows; y++)
		for (int x = 0; x < reusable.pIL.cols; x++)
		{
			using IVec = cv::Vec<float, 4>;
			auto& I0 = reusable.pIL.at<IVec>(y, x);
			auto& I1 = reusable.pIR.at<IVec>(y, x);
			rawCosts.at<float>(y, x) =
				std::min(thresh_color, fabs(I0.val[0] - I1.val[0]) + fabs(I0.val[1] - I1.val[1]) + fabs(I0.val[2] - I1.val[2])) +
				std::min(thresh_gradient, fabs(I0.val[3] - I1.val[3]));
		}

		cv::Rect subrect = targetRect - filterRect.tl();

		if (params.filterName != "")
		{
			reusable.filter->filter(rawCosts)(subrect).copyTo(costs(subrect));
			
			//reusable.cvfilter->filter(rawCosts, rawCosts);
			//rawCosts(subrect).copyTo(costs(subrect));
		}
		else
			rawCosts(subrect).copyTo(costs(subrect));
	}

	void ComputeUnaryPotential(const cv::Rect& filterRect, const cv::Rect& targetRect, const cv::Mat& costs, const Plane& plane, Reusable& reusable = Reusable(), int mode = 0) const override
	{
		ComputeUnaryPotentialWithoutCheck(filterRect, targetRect, costs, plane, reusable, mode);

		cv::Rect subrect = targetRect - filterRect.tl();
		cv::Mat validMask = IsValiLabel(plane, targetRect);
		costs(subrect).setTo(cv::Scalar(COST_FOR_INVALID), ~validMask);
	}
};
