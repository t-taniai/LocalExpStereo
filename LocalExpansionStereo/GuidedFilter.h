#pragma once

// OpenCV
#include <opencv2/opencv.hpp>
#include <memory>


class IJointFilter
{
protected:
	cv::Mat I;
	int R;

public:
	IJointFilter(const cv::Mat& I, const int R)
		: I(I), R(R)
	{
	}
	virtual ~IJointFilter()
	{
	}

	virtual std::shared_ptr<IJointFilter> createSubregionFilter (const cv::Rect& rect) const = 0;
	virtual cv::Mat filter(const cv::Mat& p) const = 0;
};


template <typename Type>
class GuidedImageFilter : public IJointFilter
{
protected:
	std::vector<cv::Mat> Ichannels;
	double eps;
	cv::Mat realI;
	cv::Mat mean_I_r, mean_I_g, mean_I_b;
	cv::Mat invrr, invrg, invrb, invgg, invgb, invbb;
	cv::Mat N;
	static const int DEPTH = cv::DataDepth<Type>::value;

	cv::Mat boxfilter(const cv::Mat& I) const
	{
		cv::Mat q;
		cv::boxFilter(I, q, -1, cv::Size(2 * R + 1, 2 * R + 1), cv::Point(-1, -1), false, cv::BORDER_CONSTANT);
		return q;
	}

public:
	virtual ~GuidedImageFilter()
	{
	}

	// Do not use this constructor.
	GuidedImageFilter()
		: IJointFilter(cv::Mat(), 0)
	{
	}

	GuidedImageFilter(const cv::Mat& _I, const int R, double eps, double scaling = 1.0)
		: IJointFilter(_I, R)
		, eps(eps)
	{
		if (I.depth() == DEPTH)
			realI = (scaling != 1.0) ? (I * scaling) : I;
		else
			I.convertTo(realI, DEPTH, scaling);

		cv::split(realI, Ichannels);

		N = boxfilter(cv::Mat_<Type>::ones(realI.size()));
		mean_I_r = boxfilter(Ichannels[0]) / N;
		mean_I_g = boxfilter(Ichannels[1]) / N;
		mean_I_b = boxfilter(Ichannels[2]) / N;

		// variance of I in each local patch: the matrix Sigma in Eqn (14).
		// Note the variance in each local patch is a 3x3 symmetric matrix:
		//           rr, rg, rb
		//   Sigma = rg, gg, gb
		//           rb, gb, bb
		cv::Mat var_I_rr = boxfilter(Ichannels[0].mul(Ichannels[0])) / N - mean_I_r.mul(mean_I_r) + eps;
		cv::Mat var_I_rg = boxfilter(Ichannels[0].mul(Ichannels[1])) / N - mean_I_r.mul(mean_I_g);
		cv::Mat var_I_rb = boxfilter(Ichannels[0].mul(Ichannels[2])) / N - mean_I_r.mul(mean_I_b);
		cv::Mat var_I_gg = boxfilter(Ichannels[1].mul(Ichannels[1])) / N - mean_I_g.mul(mean_I_g) + eps;
		cv::Mat var_I_gb = boxfilter(Ichannels[1].mul(Ichannels[2])) / N - mean_I_g.mul(mean_I_b);
		cv::Mat var_I_bb = boxfilter(Ichannels[2].mul(Ichannels[2])) / N - mean_I_b.mul(mean_I_b) + eps;

		// Inverse of Sigma + eps * I
		invrr = var_I_gg.mul(var_I_bb) - var_I_gb.mul(var_I_gb);
		invrg = var_I_gb.mul(var_I_rb) - var_I_rg.mul(var_I_bb);
		invrb = var_I_rg.mul(var_I_gb) - var_I_gg.mul(var_I_rb);
		invgg = var_I_rr.mul(var_I_bb) - var_I_rb.mul(var_I_rb);
		invgb = var_I_rb.mul(var_I_rg) - var_I_rr.mul(var_I_gb);
		invbb = var_I_rr.mul(var_I_gg) - var_I_rg.mul(var_I_rg);

		cv::Mat covDet = invrr.mul(var_I_rr) + invrg.mul(var_I_rg) + invrb.mul(var_I_rb);

		invrr /= covDet;
		invrg /= covDet;
		invrb /= covDet;
		invgg /= covDet;
		invgb /= covDet;
		invbb /= covDet;
	}


	virtual std::shared_ptr<IJointFilter> createSubregionFilter (const cv::Rect& rect) const override
	{
		return std::make_shared<GuidedImageFilter>(realI(rect), R, eps);
	}

	cv::Mat filter_mat(const cv::Mat& p) const
	{
		cv::Mat mean_p = boxfilter(p) / N;

		cv::Mat mean_Ip_r = boxfilter(Ichannels[0].mul(p)) / N;
		cv::Mat mean_Ip_g = boxfilter(Ichannels[1].mul(p)) / N;
		cv::Mat mean_Ip_b = boxfilter(Ichannels[2].mul(p)) / N;

		// covariance of (I, p) in each local patch.
		cv::Mat cov_Ip_r = mean_Ip_r - mean_I_r.mul(mean_p);
		cv::Mat cov_Ip_g = mean_Ip_g - mean_I_g.mul(mean_p);
		cv::Mat cov_Ip_b = mean_Ip_b - mean_I_b.mul(mean_p);

		cv::Mat a_r = invrr.mul(cov_Ip_r) + invrg.mul(cov_Ip_g) + invrb.mul(cov_Ip_b);
		cv::Mat a_g = invrg.mul(cov_Ip_r) + invgg.mul(cov_Ip_g) + invgb.mul(cov_Ip_b);
		cv::Mat a_b = invrb.mul(cov_Ip_r) + invgb.mul(cov_Ip_g) + invbb.mul(cov_Ip_b);

		cv::Mat b = mean_p - a_r.mul(mean_I_r) - a_g.mul(mean_I_g) - a_b.mul(mean_I_b); // Eqn. (15) in the paper;

		cv::Mat q =
			(boxfilter(a_r).mul(Ichannels[0])
				+ boxfilter(a_g).mul(Ichannels[1])
				+ boxfilter(a_b).mul(Ichannels[2])
				+ boxfilter(b)) / N;  // Eqn. (16) in the paper;
		return q;
	}

	// This code reduces redudant data access.
	// Not explicitly vectorized but hopefully done by auto vectorization of the compiler.
	// Benchmark for Adirondack:
	//   Desktop) 302 sec -> 237 sec.(22% reduction)
	//   Laptop)  498 sec -> 408 sec.(18% reduction)
	cv::Mat filter_raw(const cv::Mat& p) const
	{
		int rows = p.rows, cols = p.cols;
		cv::Mat mean_p = boxfilter(p);

		cv::Mat mean_Ip_r(p.size(), p.depth());
		cv::Mat mean_Ip_g(p.size(), p.depth());
		cv::Mat mean_Ip_b(p.size(), p.depth());

		for (int i = 0; i < rows; i++)
		{
			auto pp = p.ptr<Type>(i);
			auto pmean_Ip_r = mean_Ip_r.ptr<Type>(i);
			auto pmean_Ip_g = mean_Ip_g.ptr<Type>(i);
			auto pmean_Ip_b = mean_Ip_b.ptr<Type>(i);

			auto pI_r = Ichannels[0].ptr<Type>(i);
			auto pI_g = Ichannels[1].ptr<Type>(i);
			auto pI_b = Ichannels[2].ptr<Type>(i);

			for (int j = 0; j < cols; j++)
			{
				auto vp = pp[j];
				pmean_Ip_r[j] = pI_r[j] * vp;
				pmean_Ip_g[j] = pI_g[j] * vp;
				pmean_Ip_b[j] = pI_b[j] * vp;
			}
		}
		mean_Ip_r = boxfilter(mean_Ip_r);
		mean_Ip_g = boxfilter(mean_Ip_g);
		mean_Ip_b = boxfilter(mean_Ip_b);


		cv::Mat a_r(p.size(), p.depth());
		cv::Mat a_g(p.size(), p.depth());
		cv::Mat a_b(p.size(), p.depth());
		cv::Mat b(p.size(), p.depth());

		for (int i = 0; i < rows; i++)
		{
			auto pa_r = a_r.ptr<Type>(i);
			auto pa_g = a_g.ptr<Type>(i);
			auto pa_b = a_b.ptr<Type>(i);

			auto pN = N.ptr<Type>(i);
			auto pmean_p = mean_p.ptr<Type>(i);
			auto pmean_Ip_r = mean_Ip_r.ptr<Type>(i);
			auto pmean_Ip_g = mean_Ip_g.ptr<Type>(i);
			auto pmean_Ip_b = mean_Ip_b.ptr<Type>(i);

			auto pmean_I_r = mean_I_r.ptr<Type>(i);
			auto pmean_I_g = mean_I_g.ptr<Type>(i);
			auto pmean_I_b = mean_I_b.ptr<Type>(i);

			auto pinvrr = invrr.ptr<Type>(i);
			auto pinvrg = invrg.ptr<Type>(i);
			auto pinvrb = invrb.ptr<Type>(i);
			auto pinvgg = invgg.ptr<Type>(i);
			auto pinvgb = invgb.ptr<Type>(i);
			auto pinvbb = invbb.ptr<Type>(i);

			auto pb = b.ptr<Type>(i);
			for (int j = 0; j < cols; j++)
			{
				auto n = pN[j];
				auto mp = pmean_p[j] / n;
				auto mIr = pmean_I_r[j];
				auto mIg = pmean_I_g[j];
				auto mIb = pmean_I_b[j];

				auto cov_Ip_r = pmean_Ip_r[j] / n - mIr*mp;
				auto cov_Ip_g = pmean_Ip_g[j] / n - mIg*mp;
				auto cov_Ip_b = pmean_Ip_b[j] / n - mIb*mp;

				pa_r[j] = pinvrr[j] * cov_Ip_r + pinvrg[j] * cov_Ip_g + pinvrb[j] * cov_Ip_b;
				pa_g[j] = pinvrg[j] * cov_Ip_r + pinvgg[j] * cov_Ip_g + pinvgb[j] * cov_Ip_b;
				pa_b[j] = pinvrb[j] * cov_Ip_r + pinvgb[j] * cov_Ip_g + pinvbb[j] * cov_Ip_b;

				pb[j] = mp - pa_r[j] * mIr - pa_g[j] * mIg - pa_b[j] * mIb;
			}
		}

		a_r = boxfilter(a_r);
		a_g = boxfilter(a_g);
		a_b = boxfilter(a_b);
		b = boxfilter(b);

		for (int i = 0; i < rows; i++)
		{
			auto pa_r = a_r.ptr<Type>(i);
			auto pa_g = a_g.ptr<Type>(i);
			auto pa_b = a_b.ptr<Type>(i);
			auto pb = b.ptr<Type>(i);
			auto pN = N.ptr<Type>(i);

			auto pI_r = Ichannels[0].ptr<Type>(i);
			auto pI_g = Ichannels[1].ptr<Type>(i);
			auto pI_b = Ichannels[2].ptr<Type>(i);

			for (int j = 0; j < cols; j++)
			{
				pb[j] = (pb[j] + pa_r[j] * pI_r[j] + pa_g[j] * pI_g[j] + pa_b[j] * pI_b[j]) / pN[j];
			}
		}
		return b;
	}
	cv::Mat filter(const cv::Mat& _p) const override
	{
		cv::Mat p;
		if (_p.depth() != DEPTH) _p.convertTo(p, DEPTH);
		else p = _p;

		// This code is the largest bottleneck of the while algorithm.
		cv::Mat q = filter_raw(p);
		//cv::Mat q = filter_mat(p);

		cv::Mat _q;

		if (q.depth() != _p.depth())
			q.convertTo(_q, _p.depth());
		else
			_q = q;

		return _q;
	}


};


/* Our implementation of GuidedImageFilter is slightly different from cv::ximgproc::GuidedFilter.

1) The OpenCV version uses a mean filter that uses BORDER_DEFAULT interpolation.
   We instead use BORDER_CONSTANT (zero if out-of-boundary) and normalization is done using true kernel sizes (see cv::Mat N).
   Therefore, results are different near the image boundaries (within 2*R pixels from the boundaries).

2) When the guide image I is scaled from [0, 255] to [0, 1] by doing GuidedImageFilter(I, R, eps, 1.0 / 255),
   results by the OpenCV version does not correspond if simply do cv::xiimgproc::createGuidedFilter(I/255, R, eps).
   Instead, cv::xiimgproc::createGuidedFilter(I, R, eps * 255 * 255) seems to produce the same reslts.

*/
template <typename Type>
class FastGuidedImageFilter : public GuidedImageFilter<Type>
{

public:
	FastGuidedImageFilter()
		: GuidedImageFilter<Type>()
	{
	}

	FastGuidedImageFilter(const cv::Mat& _I, const int R, double eps, double scaling = 1.0)
		: GuidedImageFilter<Type>(_I, R, eps, scaling)
	{
	}

	// Reuse the global statistics to create subregion filter.
	// The filtered results correspond to those by subregion filter of GuidedImageFilter
	// within a valid rectangle margined by 2*R from the boundary of the subregion.
	std::shared_ptr<IJointFilter> createSubregionFilter(const cv::Rect& rect) const override
	{
		auto filter = std::make_shared<FastGuidedImageFilter>();
		filter->R = R;
		filter->eps = eps;

		filter->I = I(rect);
		filter->realI = realI(rect);
		filter->mean_I_r = mean_I_r(rect);
		filter->mean_I_g = mean_I_g(rect);
		filter->mean_I_b = mean_I_b(rect);
		filter->Ichannels.resize(3);
		filter->Ichannels[0] = Ichannels[0](rect);
		filter->Ichannels[1] = Ichannels[1](rect);
		filter->Ichannels[2] = Ichannels[2](rect);

		filter->invrr = invrr(rect);
		filter->invrg = invrg(rect);
		filter->invrb = invrb(rect);
		filter->invgg = invgg(rect);
		filter->invgb = invgb(rect);
		filter->invbb = invbb(rect);

		filter->N = boxfilter(cv::Mat_<Type>::ones(rect.size()));
		return filter;
	}
};

class BilateralFilter : public IJointFilter
{
	const double sig2;

	cv::Mat channelSum(const cv::Mat& m1) const
	{
		cv::Mat m = m1.reshape(1, m1.rows*m1.cols);
		cv::reduce(m, m, 1, cv::REDUCE_SUM);
		return m.reshape(1, m1.rows);
	}

public:
	BilateralFilter(const cv::Mat& I, const int R, double sig2)
		: IJointFilter(I, R), sig2(sig2)
	{
		if (this->I.depth() != CV_32F)
			this->I.convertTo(this->I, CV_32F);
	}

	std::shared_ptr<IJointFilter> createSubregionFilter (const cv::Rect& rect) const override
	{
		return std::make_shared<BilateralFilter>(I(rect), R, sig2);
	}


	cv::Mat filter(const cv::Mat& p) const override
	{
		cv::Mat q = cv::Mat(p.size(), p.type());
		cv::Rect filterDomain(0, 0, I.cols, I.rows);

		for (int y = 0; y < q.rows; y++)
		{
			for (int x = 0; x < q.cols; x++)
			{
				cv::Rect patch = cv::Rect(x - R, y - R, 2 * R + 1, 2 * R + 1) & filterDomain;

				cv::Mat w = cv::abs(I(patch) - I.at<cv::Vec3f>(y, x));
				cv::exp(-channelSum(w) / sig2, w);

				q.at<float>(y, x) = (float)p(patch).dot(w);
			}
		}
		return q;
	}

};
