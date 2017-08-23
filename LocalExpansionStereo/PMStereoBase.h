#pragma once
#include "StereoEnergy.h"
#include "Utilities.hpp"
#include "Evaluator.h"
#include <vector>

class PMStereoBase
{
public:
	bool debug;
	std::string saveDir;

protected:
	cv::Mat currentLabeling_[2];
	cv::Mat currentLabeling_m_[2];
	cv::Mat currentCost_[2];

	const int nNodes;
	const int width;
	const int height;
	const cv::Rect imageDomain;

	std::unique_ptr<StereoEnergy> stereoEnergy;

	const int M;
	Evaluator* evaluator;
	const Parameters params;

	double dispVisScaling;
	double dispVisOffset;

public:
	PMStereoBase(cv::Mat imL, cv::Mat imR, Parameters params, double maxDisparity, double minDisparity = 0, double maxVDisparity = 0) :
		width(imL.cols),
		height(imL.rows),
		nNodes(imL.cols * imL.rows),
		stereoEnergy(std::make_unique<NaiveStereoEnergy>(imL, imR, params, maxDisparity, minDisparity, maxVDisparity)),
		imageDomain(0, 0, imL.cols, imL.rows),
		M(1),
		evaluator(nullptr),
		params(params),
		debug(true)
	{
		currentLabeling_m_[0] = cv::Mat::zeros(height + 2 * M, width + 2 * M, CV_MAKE_TYPE(cv::DataType<float>::depth, 4));
		currentLabeling_m_[1] = cv::Mat::zeros(height + 2 * M, width + 2 * M, CV_MAKE_TYPE(cv::DataType<float>::depth, 4));
		currentLabeling_[0] = currentLabeling_m_[0](stereoEnergy->getRectWithoutMargin());
		currentLabeling_[1] = currentLabeling_m_[1](stereoEnergy->getRectWithoutMargin());
		currentCost_[0] = cv::Mat::zeros(height, width, CV_32F);
		currentCost_[1] = cv::Mat::zeros(height, width, CV_32F);

		dispVisScaling = 255.0 / (maxDisparity - minDisparity);
		dispVisOffset = -minDisparity * dispVisScaling;
	}
	virtual ~PMStereoBase()
	{
	}

	void setStereoEnergyCPU(std::unique_ptr<StereoEnergy> energy)
	{
		stereoEnergy = std::move(energy);
	}


	void setVisualizationParams(double scaling, double offset)
	{
		dispVisOffset = offset;
		dispVisScaling = scaling;
	}

	const StereoEnergy& getEnergyInstance()
	{
		return *stereoEnergy;
	}

	void setEvaluator(Evaluator* evaluator)
	{
		this->evaluator = evaluator;
	}

	bool isValid(int x, int y, int mode = 0)
	{
		return true;
	}

protected:

	void viewConsistencyCheck(cv::Mat& check0 = cv::Mat(), cv::Mat& check1 = cv::Mat())
	{
		cv::Mat disp[2] = { stereoEnergy->computeDisparities(currentLabeling_[0]), stereoEnergy->computeDisparities(currentLabeling_[1]) };
		cv::Mat fail[2];
		doConsistencyCheck(disp[0], disp[1], fail[0], fail[1], 1.5);

		cv::Mat visCheck[2];
		for (int m = 0; m < 2; m++){
			cv::Mat disp8u;
			disp[m].convertTo(disp8u, CV_8U, dispVisScaling, dispVisOffset);
			std::vector<cv::Mat> ch(3);
			ch[0] = disp8u.clone();
			ch[1] = disp8u.clone();
			ch[2] = disp8u.clone();

			ch[0].setTo(cv::Scalar(255), fail[m] == 128);
			ch[2].setTo(cv::Scalar(255), fail[m] == 255);
			cv::merge(ch, visCheck[m]);
		}
		check0 = visCheck[0];
		check1 = visCheck[1];
	}


	void doConsistencyCheck(const cv::Mat& dispL, const cv::Mat& dispR, cv::Mat& failL, cv::Mat& failR, double dispThreshold = 1.5)
	{
		cv::Mat fail[2];
		cv::Mat disp[2] = { dispL, dispR };
		for (int i = 0; i < 2; i++) {
			fail[i] = cv::Mat::zeros(cv::Size(width, height), CV_8U);
		}

		for (int i = 0; i < 2; i++){
			for (int y = 0; y < height; y++)
			for (int x = 0; x < width; x++){
				cv::Point p(x, y);

				float ds = disp[i].at<float>(p);
				int rx = (float)x - ds * (i ? -1 : 1) + 0.5;

				cv::Point q(rx, y);
				if (imageDomain.contains(q)){
					float dsr = disp[1 - i].at<float>(q);
					if (fabs(dsr - ds) > dispThreshold && isValid(x, y, i)){
						fail[i].at<uchar>(p) = 255;
					}
				}
				else if (isValid(x, y, i)){
					// not-occluded but invisible regions
					fail[i].at<uchar>(p) = 128;
				}
			}
		}

		failL = fail[0];
		failR = fail[1];
	}

	void postProcess(cv::Mat labeingL, cv::Mat labeingR, float threshod = 1.0)
	{
		cv::Mat LR[2] = { labeingL, labeingR };

		// LR-consistency check
		cv::Mat fail[2];
		cv::Mat fail2[2];

		for (int i = 0; i < 2; i++) {
			fail[i] = cv::Mat::zeros(cv::Size(width, height), CV_8U);
			fail2[i] = cv::Mat::zeros(cv::Size(width, height), CV_8U);
		}

		cv::Mat disp[2] = { stereoEnergy->computeDisparities(LR[0]), stereoEnergy->computeDisparities(LR[1]) };

		doConsistencyCheck(disp[0], disp[1], fail[0], fail[1], threshod);
		fail[0] = fail[0] > 0;
		fail[1] = fail[1] > 0;

		cv::dilate(fail[0], fail2[0], cv::Mat());
		cv::dilate(fail[1], fail2[1], cv::Mat());

		//// horizontal NN-interpolation
		for (int i = 0; i < 2; i++){
			#pragma omp parallel for
			for (int y = 0; y < height; y++)
			for (int x = 0; x < width; x++){
				cv::Point p(x, y);

				if (fail[i].at<uchar>(p) == 0 || isValid(x, y, i) == false) continue;

				Plane *pl = NULL, *pr = NULL;
				int xx;
				for (xx = x; xx >= 0 && fail2[i].at<uchar>(y, xx) == 255; xx--);
				if (xx >= 0 && isValid(xx, y, i))
					pl = &LR[i].at<Plane>(y, xx);

				for (xx = x; xx < width && fail2[i].at<uchar>(y, xx) == 255; xx++);
				if (xx < width && isValid(xx, y, i))
					pr = &LR[i].at<Plane>(y, xx);

				if (pl == NULL && pr == NULL)
					//LR[i][s] = *pr;
					;
				else if (pl == NULL)
					LR[i].at<Plane>(p) = *pr;
				else if (pr == NULL)
					LR[i].at<Plane>(p) = *pl;
				else if (pl->GetZ(p) < pr->GetZ(p))
					LR[i].at<Plane>(p) = *pl;
				else
					LR[i].at<Plane>(p) = *pr;
			}
			
			//cv::imshow(cv::format("filled%d", i), stereoEnergy->computeDisparities(LR[i]) / MAX_DISPARITY); cv::waitKey(10);

		}

		//OutputFiles(L, 0, "postL");
		//OutputFiles(R, 0, "postR");

		using Triplet = std::tuple<Plane, float, float>;

		//// median filter
		for (int i = 0; i < 2; i++){
			cv::Mat LRcopy = LR[i].clone();
			
			#pragma omp parallel for
			for (int y = 0; y < height; y++)
			for (int x = 0; x < width; x++){
				cv::Point p(x, y);

				if (fail[i].at<uchar>(p) == 0 || isValid(x, y, i) == false) continue;
				std::vector<Triplet> median;
				double sumw = 0;

				cv::Rect patch = cvutils::getLargerRect(cv::Rect(p, cv::Size(1, 1)), params.windR) & imageDomain;

				for (int yy = patch.y; yy < patch.br().y; yy++)
				for (int xx = patch.x; xx < patch.br().x; xx++){
					cv::Point q(xx, yy);

					if (isValid(xx, yy, i) == false) continue;

					float w = stereoEnergy->computePatchWeight(p, q, i);
					sumw += w;
					//median.push_back(Triplet(LR[i].at<Plane>(q), w, LR[i].at<Plane>(q).GetZ(p)));
					median.push_back(Triplet(LRcopy.at<Plane>(q), w, LRcopy.at<Plane>(q).GetZ(p)));

				}

				std::sort(begin(median), end(median), [](const Triplet& a, const Triplet& b){
					return std::get<2>(a) < std::get<2>(b);
				});

				double center = sumw / 2.0;
				sumw = 0;
				for (int j = 0; j < median.size(); j++){
					sumw += std::get<1>(median[j]);
					if (sumw > center) {
						LR[i].at<Plane>(p) = std::get<0>(median[j]);
						break;
					}
				}
			}
			//cv::imshow(cv::format("filtered%d", i), stereoEnergy->computeDisparities(LR[i]) / MAX_DISPARITY); cv::waitKey(10);
		}

		//OutputFiles(L, 1, "postL");
		//OutputFiles(R, 1, "postR");
	}

	cv::Point getPoint(int s)
	{
		return cv::Point(s % width, s / width);
	}

	double computeCurrentEnergy(double *data = NULL, double *smooth = NULL)
	{
		double dataCost = cv::sum(currentCost_[0])[0];
		double smoothnessCost = stereoEnergy->computeSmoothnessCost(currentLabeling_m_[0]);
		if (data) *data = dataCost;
		if (smooth) *smooth = smoothnessCost;
		return dataCost + smoothnessCost;
	}


};
