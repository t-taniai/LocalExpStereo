#pragma once
#include <opencv2/opencv.hpp>
#include "Plane.h"
#include "Utilities.hpp"

class IProposer
{
protected:
	const int K;

	cv::Mat labeling;
	int iter;
	int outerIter;
	cv::Rect rect;

public:
	IProposer(int K)
		: K(K)
	{
	}

	virtual IProposer* createInstance() = 0;

	virtual ~IProposer()
	{
	}

	virtual void startIterations(cv::Mat labeling, cv::Rect rect, int outerIter) = 0;
	virtual Plane getNextProposal() = 0;
	virtual bool isContinued() = 0;
};


class ExpansionProposer : public IProposer
{
protected:
	cv::Point selectRandomPixelInRect(cv::Rect rect)
	{
		int n = cv::theRNG().uniform(0, rect.height * rect.width);

		int xx = n % rect.width;
		int yy = n / rect.width;
		return cv::Point(rect.x + xx, rect.y + yy);
	}

public:

	ExpansionProposer(int K)
		: IProposer(K)
	{
	}

	virtual IProposer* createInstance() override
	{
		return (IProposer*)(new ExpansionProposer(K));
	}

	virtual ~ExpansionProposer()
	{
	}

	virtual void startIterations(cv::Mat labeling, cv::Rect unitRegion, int outerIter) override
	{
		this->labeling = labeling(unitRegion);
		this->outerIter = outerIter;
		rect = unitRegion;
		iter = 0;
	}
	virtual Plane getNextProposal() override
	{
		auto p = selectRandomPixelInRect(cv::Rect(0, 0, labeling.cols, labeling.rows));
		auto label = labeling.at<Plane>(p);
		iter++;
		return label;
	}
	virtual bool isContinued() override
	{
		return iter < K;
	}
};



class RandomProposer : public ExpansionProposer
{
protected:
	const float MIN_DISPARITY;
	const float MAX_DISPARITY;
	const float MAX_VDISPARITY;
	const float randomNmax;
	const bool doEarlyStop;

	float getDisparityPerturbationWidth(int m)
	{
		return (MAX_DISPARITY - MIN_DISPARITY) * pow(0.5f, m + 1);
	}

public:

	RandomProposer(int K, float maxDisp, float minDisp = 0, float maxVDisp = 0, bool doEarlyStop = true)
		: ExpansionProposer(K)
		, MIN_DISPARITY(minDisp)
		, MAX_DISPARITY(maxDisp)
		, MAX_VDISPARITY(maxVDisp)
		, doEarlyStop(doEarlyStop)
		, randomNmax(1.0)
	{
	}

	virtual ~RandomProposer()
	{
	}

	virtual IProposer* createInstance() override
	{
		return (IProposer*)(new RandomProposer(K, MAX_DISPARITY, MIN_DISPARITY, MAX_VDISPARITY, doEarlyStop));
	}


	virtual Plane getNextProposal() override
	{
		auto p = selectRandomPixelInRect(cv::Rect(0, 0, labeling.cols, labeling.rows));
		auto in = labeling.at<Plane>(p);
		int m = outerIter + iter;
		iter++;

		cv::Point s = rect.tl() + p;
		float zs = in.GetZ(float(s.x), float(s.y));
		float dz = getDisparityPerturbationWidth(m);
		float minz = std::max(MIN_DISPARITY, zs - dz);
		float maxz = std::min(MAX_DISPARITY, zs + dz);
		zs = cv::theRNG().uniform(minz, maxz);

		float vs = in.v;
		if (MAX_VDISPARITY != 0)
		{
			float dv = MAX_VDISPARITY * pow(0.5f, m + 1);
			float minv = std::max(-MAX_VDISPARITY, vs - dv);
			float maxv = std::min(+MAX_VDISPARITY, vs + dv);
			vs = cv::theRNG().uniform(minv, maxv);
		}
		float nr = randomNmax * pow(0.5f, m);
		cv::Vec<float, 3> nv = in.GetNormal() + (cv::Vec<float, 3>) cvutils::getRandomUnitVector() * nr;

		nv = nv / sqrt(nv.ddot(nv));

		return Plane::CreatePlane(nv[0], nv[1], nv[2], zs, float(s.x), float(s.y), vs);
	}
	virtual bool isContinued() override
	{
		return (iter < K) && !(doEarlyStop && getDisparityPerturbationWidth(outerIter + iter) < 0.1);
	}
};

class RansacProposer : public IProposer
{
protected:
	cv::Mat coord;
	cv::Mat disps;
	const int MAX_SAM;
	const float conf;

	std::vector<int> randperm(int len)
	{
		std::vector<int> v;

		v.reserve(len);
		for (int i = 0; i < len; ++i)
			v.push_back(i);

		std::random_shuffle(v.begin(), v.end());

		return v;
	}

	// [u v 1] * [a b c]^T =[d]
	Plane RANSACPlane(cv::Mat pts, cv::Mat disp, float threshold)
	{
		int len = pts.rows;
		int max_i = 3;
		int max_sam = MAX_SAM;
		int no_sam = 0;
		cv::Mat div = cv::Mat_<float>::zeros(3, 1);
		cv::Mat inls = cv::Mat_<uchar>(len, 1, (uchar)0);
		int no_i_c = 0;
		cv::Mat N = cv::Mat_<float>(3, 1);
		cv::Mat result;

		cv::Mat ranpts = cv::Mat_<float>::zeros(3, 3);

		while (no_sam < max_sam)
		{
			no_sam = no_sam + 1;
			auto ransam = randperm(len);
			for (int i = 0; i < 3; i++)
			{
				ranpts.at<cv::Vec3f>(i) = pts.at<cv::Vec3f>(ransam[i]);
				div.at<float>(i) = disp.at<float>(ransam[i]);
			}
			/// compute a distance of all points to a plane given by pts(:, sam) to dist
			cv::solve(ranpts, div, N, cv::DECOMP_SVD);
			cv::Mat dist = cv::abs(pts * N - disp);
			cv::Mat v = dist < threshold;
			int no_i = cv::countNonZero(v);

			if (max_i < no_i)
			{
				// Re - estimate plane and inliers
				cv::Mat b = cv::Mat_<float>::zeros(no_i, 1);
				cv::Mat A = cv::Mat_<float>::zeros(no_i, 3);

				// MATLAB: A = pts(v, :);
				for (int i = 0, j = 0; i < no_i; i++)
				if (v.at<uchar>(i))
				{
					A.at<cv::Vec3f>(j) = pts.at<cv::Vec3f>(i);
					b.at<float>(j) = disp.at<float>(i);
					j++;
				}

				cv::solve(A, b, N, cv::DECOMP_SVD);
				dist = cv::abs(pts * N - disp);
				v = dist < threshold;
				int no = cv::countNonZero(v);

				if (no > no_i_c)
				{
					result = N.clone();
					no_i_c = no;
					inls = v;
					max_i = no_i;
					max_sam = std::min(max_sam, computeSampleCount(no, len, 3, conf));
				}
			}
		}
		return Plane(result.at<float>(0), result.at<float>(1), result.at<float>(2));
	}


	int computeSampleCount(int ni, int ptNum, int pf, double conf)
	{
		int SampleCnt;

		// MATLAB: q = prod([(ni - pf + 1) : ni] . / [(ptNum - pf + 1) : ptNum]);
		double q = 1.0;
		for (double a = (ni - pf + 1), b = (ptNum - pf + 1); a <= ni; a += 1.0, b += 1.0)
			q *= (a / b);

		const double eps = 1e-4;

		if ((1.0 - q) < eps)
			SampleCnt = 1;
		else
			SampleCnt = int(log(1.0 - conf) / log(1.0 - q));

		if (SampleCnt < 1)
			SampleCnt = 1;
		return SampleCnt;
	}
public:

	RansacProposer(int K, int MAX_SAM = 500, float conf = 0.95)
		: IProposer(K)
		, MAX_SAM(MAX_SAM)
		, conf(conf)

	{
	}

	virtual IProposer* createInstance() override
	{
		return (IProposer*)(new RansacProposer(K, MAX_SAM, conf));
	}

	virtual ~RansacProposer()
	{
	}


	virtual void startIterations(cv::Mat labeling, cv::Rect unitRegion, int outerIter) override
	{
		this->labeling = labeling(unitRegion);
		this->outerIter = outerIter;
		rect = unitRegion;
		iter = 0;

		coord = cv::Mat(unitRegion.size(), CV_32FC3);
		disps = cv::Mat(unitRegion.size(), CV_32FC1);
		for (int y = 0; y < coord.rows; y++)
		for (int x = 0; x < coord.cols; x++){
			auto c = cv::Vec3f((float)x + unitRegion.x, (float)y + unitRegion.y, 1.0f);
			coord.at<cv::Vec3f>(y, x) = c;
			auto v = labeling.at<cv::Vec4f>(y + unitRegion.y, x + unitRegion.x);
			disps.at<float>(y, x) = v[0] * c[0] + v[1] * c[1] + v[2];
		}
		coord = coord.reshape(1, coord.rows * coord.cols);
		disps = disps.reshape(1, disps.rows * disps.cols);
	}
	virtual Plane getNextProposal() override
	{
		iter++;
		return RANSACPlane(coord, disps, 1.0);
	}
	virtual bool isContinued() override
	{
		return iter < K;
	}

};

