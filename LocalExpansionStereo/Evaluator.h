#pragma once

#include "TimeStamper.h"
#include <opencv2/opencv.hpp>
#include "StereoEnergy.h"

class Evaluator
{
	
protected:
	TimeStamper timer;
	const float DISPARITY_FACTOR;
	const cv::Mat dispGT;
	const cv::Mat nonoccMask;
	cv::Mat occMask;
	cv::Mat validMask;
	std::string saveDir;
	std::string header;
	int validPixels;
	int nonoccPixels;
	FILE *fp_energy;
	FILE *fp_output;
	float qprecision;
	float errorThreshold;
	bool showedOnce;

public:
	double lastAccuracy;
	bool showProgress;
	bool saveProgress;
	bool printProgress;

	std::string getSaveDirectory()
	{
		return saveDir;
	}

	Evaluator(cv::Mat dispGT, cv::Mat nonoccMask, float disparityFactor, std::string header = "result", std::string saveDir = "./", bool show = true, bool print = true, bool save = true)
		: dispGT(dispGT)
		, nonoccMask(nonoccMask)
		, header(header)
		, saveDir(saveDir)
		, DISPARITY_FACTOR(disparityFactor)
		, showProgress(show)
		, saveProgress(save)
		, printProgress(print)
		, fp_energy(nullptr)
		, fp_output(nullptr)
	{

		if (save)
		{
			//fp_energy = fopen((saveDir + "log_energy.txt").c_str(), "w");
			//if (fp_energy != nullptr)
			//{
			//	fprintf(fp_energy, "%s\t%s\t%s\t%s\t%s\t%s\n", "Time", "Eng", "Data", "Smooth", "all", "nonocc");
			//	fflush(fp_energy);
			//}

			fp_output = fopen((saveDir + "log_output.txt").c_str(), "w");
			if (fp_output != nullptr)
			{
				fprintf(fp_output, "%s\t%s\t%s\t%s\t%s\t%s\n", "Time", "Eng", "Data", "Smooth", "all", "nonocc");
				fflush(fp_output);
			}
		}

		showedOnce = false;
		errorThreshold = 0.5f;
		qprecision = 1.0f / DISPARITY_FACTOR;

		validMask = (dispGT > 0.0f) & (dispGT != INFINITY);
		validPixels = cv::countNonZero(validMask);
		occMask = ~nonoccMask & validMask;
		nonoccPixels = cv::countNonZero(nonoccMask);
	}
	~Evaluator()
	{
		if (fp_energy != nullptr) fclose(fp_energy);
		if (fp_output != nullptr) fclose(fp_output);
	}

	void setPrecision(float precision)
	{
		qprecision = precision;
	}
	void setErrorThreshold(float t)
	{
		errorThreshold = t;
	}

	//void outputFiles(cv::Mat labeling, int index, const char *header = "result", bool showImage = true, bool justShow = false)
	//{
	//	cv::Mat disparityMap = cvutils::channelDot(labeling, coordinates) * (DisparityFactor / 255.0);
	//	cv::Mat normalMap = getNormalMap(labeling);

	//	if (justShow == false){
	//		char str[512];
	//		sprintf(str, "%sD%02d.png", header, index);
	//		cv::imwrite(saveDir + str, disparityMap * 255);
	//		sprintf(str, "%sN%02d.png", header, index);
	//		cv::imwrite(saveDir + str, normalMap * 255);
	//	}
	//}

	void quantize(cv::Mat m, float precision)
	{
		cv::Mat qm = cv::Mat(m.size(), CV_32S);
		m.convertTo(qm, CV_32S, 1.0 / precision);
		qm.convertTo(m, m.type(), precision);
	}

	void evaluate(cv::Mat labeling_m, cv::Mat unaryCost2, const StereoEnergy& energy2, bool show, bool save, bool print, int index, int mode)
	{
		bool isTicking = timer.isTicking();
		stop();

		cv::Mat labeling = labeling_m(energy2.getRectWithoutMargin());
		double sc2 = energy2.computeSmoothnessCost(labeling_m);
		double dc2 = cv::sum(unaryCost2)[0];
		double eng2 = sc2 + dc2;

		cv::Mat disp = energy2.computeDisparities(labeling);
		if (qprecision > 0)
			quantize(disp, qprecision);

		cv::Mat disparityMapVis = disp * DISPARITY_FACTOR / 255;
		cv::Mat normalMapVis = energy2.computeNormalMap(labeling);
		//cv::Mat vdispMapVis;
		//cv::extractChannel(labeling, vdispMapVis, 3);
		//vdispMapVis = (vdispMapVis + 3.0) / 6.0;

		cv::Mat errorMap = cv::abs(disp - dispGT) <= errorThreshold;
		cv::Mat errorMapVis = errorMap | (~validMask);
		errorMapVis.setTo(cv::Scalar(200), occMask & (~errorMapVis));

		double all = 1.0 - (double)cv::countNonZero(errorMap & validMask) / validPixels;
		double nonocc = 1.0 - (double)cv::countNonZero(errorMap & nonoccMask) / nonoccPixels;
		all *= 100.0;
		nonocc *= 100.0;

		if (mode == 0)
			lastAccuracy = all;

		if (showProgress && show)
		{
			if (showedOnce == false)
			{
				showedOnce = true;
				cv::namedWindow(header + std::to_string(mode) + "V", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
				cv::namedWindow(header + std::to_string(mode) + "D", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
				cv::namedWindow(header + std::to_string(mode) + "N", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
				cv::namedWindow(header + std::to_string(mode) + "E", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
			}
			//cv::imshow(header + std::to_string(mode) + "V", vdispMapVis);
			cv::imshow(header + std::to_string(mode) + "D", disparityMapVis);
			cv::imshow(header + std::to_string(mode) + "N", normalMapVis);
			cv::imshow(header + std::to_string(mode) + "E", errorMapVis);
			cv::waitKey(10);
		}

		if (saveProgress && save){
			cv::imwrite(saveDir + cv::format("%s%dD%02d.png", header, mode, index), disparityMapVis * 255);
			//cv::imwrite(saveDir + cv::format("%s%dV%02d.png", header, mode, index), vdispMapVis * 255);
			cv::imwrite(saveDir + cv::format("%s%dN%02d.png", header, mode, index), normalMapVis * 255);
			cv::imwrite(saveDir + cv::format("%s%dE%02d.png", header, mode, index), errorMapVis);

			if (fp_output != nullptr && mode == 0)
			{
				fprintf(fp_output, "%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n", getCurrentTime(), eng2, dc2, sc2, all, nonocc);
				fflush(fp_output);
			}
		}

		// Output energy values in inner loops
		if (mode == 0 && fp_energy != nullptr && saveProgress)
		{
			fprintf(fp_energy, "%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n", getCurrentTime(), eng2, dc2, sc2, all, nonocc);
			fflush(fp_energy);
		}

		if (printProgress && print) if ( mode == 0)
			std::cout << cv::format("%2d %5.1lf\t%.0lf\t%.0lf\t%.0lf\t%4.2lf\t%4.2lf", index, getCurrentTime(), eng2, dc2, sc2, all, nonocc) << std::endl;

		if (isTicking)
			start();
	}

	void start()
	{
		timer.start();
	}

	void stop()
	{
		timer.stop();
	}

	double getCurrentTime()
	{
		return timer.getCurrentTime();
	}

};