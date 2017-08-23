#pragma once
#include "LayerManager.h"
#include "StereoEnergy.h"
#include "Utilities.hpp"
#include "TimeStamper.h"
#include "PMStereoBase.h"
#include "../maxflow/graph.h"

//#define USE_GPU

#define STOP_TIMER(eval) {if(eval != nullptr){eval->stop();}}
#define START_TIMER(eval) {if(eval != nullptr){eval->start();}}

class FastGCStereo : public PMStereoBase
{
public:

protected:
	LayerManager layermng;
	bool doInnerLoopLog;

	void localExpansionMovesForLayer_CPU(const struct LayerManager::Layer& layer, cv::Mat& currentCost, cv::Mat& currentLabeling, cv::Mat& currentLabeling_m, int mode, int i, int iteration, bool doGC = true)
	{
		// typically a 16-time loop
		cv::Mat proposalCost = cv::Mat(height, width, CV_32F);
		for (int j = 0; j < layer.disjointRegionSets.size(); j++)
		{
			//cv::Mat proposalLabeling = cv::Mat(currentLabeling.size(), currentLabeling.type(), INVALID_LABEL.toScalar());

			#pragma omp parallel for
			for (int n = 0; n < layer.disjointRegionSets[j].size(); n++)
			{
				int r = layer.disjointRegionSets[j][n];
				auto& sharedRegion = layer.sharedRegions[r];
				auto& unitRegion = layer.unitRegions[r];
				cv::Mat subCurrentCost = currentCost(sharedRegion);
				cv::Mat subProposalCost = proposalCost(sharedRegion);
				cv::Mat subCurrentLabeling = currentLabeling(sharedRegion);

				StereoEnergy::Reusable reusable;
				for (auto& proposer : layer.proposers)
				{
					auto prop = proposer->createInstance(); // Need to create a new instance for multi-threading
					prop->startIterations(currentLabeling, unitRegion, iteration);
					while (prop->isContinued())
					{
						auto label = prop->getNextProposal();

						stereoEnergy->ComputeUnaryPotential(layer.filterRegions[r], sharedRegion, proposalCost(layer.filterRegions[r]), label, reusable, mode);

						cv::Mat updateMask;
						if (doGC){
							updateMask = cv::Mat_<uchar>(sharedRegion.size());
							expansionMoveBK(updateMask, label, sharedRegion, subProposalCost, mode);
						}
						else {
							updateMask = subCurrentCost > subProposalCost;
						}
						subProposalCost.copyTo(subCurrentCost, updateMask);
						subCurrentLabeling.setTo(label.toScalar(), updateMask);
					}
					delete prop;
				}
			}
			if (debug && evaluator != nullptr && doInnerLoopLog) evaluator->evaluate(currentLabeling_m, currentCost, *stereoEnergy, false, false, false, 0, mode);
			//if (evaluator == nullptr)
			//{
			//	cv::imshow("disp", (stereoEnergy->computeDisparities(currentLabeling) - MIN_DISPARITY) / (MAX_DISPARITY - MIN_DISPARITY));
			//	cv::waitKey(1);
			//}
		}// for each disjoint group
	}


public:
	FastGCStereo(cv::Mat imL, cv::Mat imR, Parameters params, double maxDisparity, double minDisparity = 0, double maxVDisparity = 0)
		: PMStereoBase(imL, imR, params, maxDisparity, minDisparity, maxVDisparity)
		, layermng(imL.cols, imL.rows, params.windR, 0)
	{
		doInnerLoopLog = false;
	}


	virtual ~FastGCStereo(void)
	{
	}

	void addLayer(int unitRegionSize, std::vector<IProposer*> proposers)
	{
		layermng.addLayer(unitRegionSize);
		layermng.layers[layermng.layers.size() - 1].proposers = proposers;
	}

	void initCurrentFast(int mode, cv::Mat& labeling)
	{
		cv::Mat currentCost = currentCost_[mode];
		cv::Mat currentLabeling = currentLabeling_[mode];
		cv::Mat currentLabeling_m = currentLabeling_m_[mode];

		if (labeling.empty())
		{
			currentCost = 0;
			auto& layer = layermng.layers[0];
			#pragma omp parallel for
			for (int j = 0; j < layer.unitRegions.size(); j++){
				cv::Rect unit = layer.unitRegions[j];
				cv::Point pnt = selectRandomPixelInRect(unit);
				auto label = stereoEnergy->createRandomLabel(pnt);
				currentLabeling(unit) = label.toScalar();

				const int R = params.windR;
				cv::Rect filterRegion = cv::Rect(unit.x - R, unit.y - R, unit.width + R * 2, unit.height + R * 2) & imageDomain;
				stereoEnergy->ComputeUnaryPotential(filterRegion, unit, currentCost(filterRegion), label, NaiveStereoEnergy::Reusable(), mode);
			}
		}
		else // May start from a given labeling but this is very slow.
		{
			labeling.copyTo(currentLabeling);

			#pragma omp parallel for
			for (int y = 0; y < height; y++)
			for (int x = 0; x < width; x++)
			{
				const int R = params.windR;
				cv::Rect unit(x, y, 1, 1);
				cv::Rect filterRegion = cv::Rect(unit.x - R, unit.y - R, unit.width + R * 2, unit.height + R * 2) & imageDomain;

				stereoEnergy->ComputeUnaryPotential(filterRegion, unit, currentCost(filterRegion), currentLabeling.at<Plane>(y, x), NaiveStereoEnergy::Reusable(), mode);
			}
		}
	}

	virtual void run(int maxIteration, const std::vector<int>& viewModes = {0, 1}, int pmInit = 0, cv::Mat& labeling = cv::Mat(), cv::Mat& rawlabeling = cv::Mat())
	{
		for (int mode : viewModes)
		{
			currentCost_[mode] = INFINITY;
			initCurrentFast(mode, labeling);
			if (debug && evaluator != nullptr) evaluator->evaluate(currentLabeling_m_[mode], currentCost_[mode], *stereoEnergy, true, true, true, 0, mode);
		}
		START_TIMER(evaluator);

		for (int iteration = 0; iteration < pmInit; iteration++)
		{
			for (int mode : viewModes)
			{
				cv::Mat currentCost = currentCost_[mode];
				cv::Mat currentLabeling = currentLabeling_[mode];
				cv::Mat currentLabeling_m = currentLabeling_m_[mode];

				for (int i = 0; i < layermng.layers.size(); i++){
					const auto& layer = layermng.layers[i];

					localExpansionMovesForLayer_CPU(layer, currentCost, currentLabeling, currentLabeling_m, mode, i, iteration, false);
				}
				if (debug && evaluator != nullptr) evaluator->evaluate(currentLabeling_m, currentCost, *stereoEnergy, true, true, true, iteration + 1, mode);
			}//endfor viewModes

			// Do consistency check at the end of each iteration
			if (debug && viewModes.size() == 2)
			{
				STOP_TIMER(evaluator);
				cv::Mat check0, check1;
				viewConsistencyCheck(check0, check1);
				cv::imwrite(cv::format("%s/result0C%02d.png", saveDir.c_str(), iteration + 1), check0);
				cv::imwrite(cv::format("%s/result1C%02d.png", saveDir.c_str(), iteration + 1), check1);
				START_TIMER(evaluator);
			}
		}

		for (int iteration = 0; iteration < maxIteration; iteration++){
			for (int mode : viewModes)
			{
				cv::Mat currentCost = currentCost_[mode];
				cv::Mat currentLabeling = currentLabeling_[mode];
				cv::Mat currentLabeling_m = currentLabeling_m_[mode];

				for (int i = 0; i < layermng.layers.size(); i++){
					const auto& layer = layermng.layers[i];

					localExpansionMovesForLayer_CPU(layer, currentCost, currentLabeling, currentLabeling_m, mode, i, iteration);
				}
				if (debug && evaluator != nullptr) evaluator->evaluate(currentLabeling_m, currentCost, *stereoEnergy, true, true, true, iteration + 1 + pmInit, mode);

			}//endfor viewModes

			// Do consistency check at the end of each iteration
			if (debug && viewModes.size() == 2)
			{
				STOP_TIMER(evaluator);
				cv::Mat check0, check1;
				viewConsistencyCheck(check0, check1);
				cv::imwrite(cv::format("%s/result0C%02d.png", saveDir.c_str(), iteration + pmInit + 1), check0);
				cv::imwrite(cv::format("%s/result1C%02d.png", saveDir.c_str(), iteration + pmInit + 1), check1);
				START_TIMER(evaluator);
			}
		}

		if (viewModes.size() == 2)
		{
			rawlabeling = currentLabeling_[0].clone();
			postProcess(currentLabeling_[0], currentLabeling_[1], 1.5);
			labeling = currentLabeling_[0].clone();

			if (debug && evaluator != nullptr) evaluator->evaluate(currentLabeling_m_[0], currentCost_[0], *stereoEnergy, true, true, true, maxIteration + 1 + pmInit, 0);
			if (debug && evaluator != nullptr) evaluator->evaluate(currentLabeling_m_[1], currentCost_[1], *stereoEnergy, true, true, true, maxIteration + 1 + pmInit, 1);
			
			if(debug)
			{
				STOP_TIMER(evaluator);
				cv::Mat check0, check1;
				viewConsistencyCheck(check0, check1);
				cv::imwrite(cv::format("%s/result0C%02d.png", saveDir.c_str(), maxIteration + 1 + pmInit), check0);
				cv::imwrite(cv::format("%s/result1C%02d.png", saveDir.c_str(), maxIteration + 1 + pmInit), check1);
				START_TIMER(evaluator);
			}
		}
		else {
			for (int mode : viewModes)
			if(mode == 0)
			{
				rawlabeling = currentLabeling_[0].clone();
				labeling = currentLabeling_[0].clone();
			}
		}
	}


protected:

	cv::Point selectRandomPixelInRect(cv::Rect rect)
	{
		int n = cv::theRNG().uniform(0, rect.height * rect.width);

		int xx = n % rect.width;
		int yy = n / rect.width;
		return cv::Point(rect.x + xx, rect.y + yy);
	}


	double fusionMoveBK(cv::Mat updateMask, const cv::Mat& labeling1_m, cv::Rect region, const cv::Mat proposalCosts, int mode = 0)
	{
		std::vector<cv::Mat> cost00;
		std::vector<cv::Mat> cost01;
		std::vector<cv::Mat> cost10;
		std::vector<cv::Mat> cost11;

		cv::Mat currentCost = currentCost_[mode];
		cv::Mat currentLabeling = currentLabeling_[mode];
		cv::Mat currentLabeling_m = currentLabeling_m_[mode];


		cv::Mat labeling1 = labeling1_m(imageDomain + cv::Point(M, M));
		stereoEnergy->computeSmoothnessTermsFusion(currentLabeling_m, labeling1_m, region, cost00, cost01, cost10, cost11, true);
		// We do not use cost11 here, since they can be ignored with our smoothness term formutation

		int N = region.width * region.height;
		typedef Graph<float, float, double> G;
		G graph(N, 4 * N);

		graph.add_node(N);
		cv::Mat subCurrent = currentCost(region);
		for (int y = 0; y < region.height; y++){
			for (int x = 0; x < region.width; x++){
				int s = y*region.width + x;
				graph.add_tweights(s, subCurrent.at<float>(y, x), proposalCosts.at<float>(y, x));

				bool x0 = x == 0;
				bool x1 = x == region.width - 1;
				bool y0 = y == 0;
				bool y1 = y == region.height - 1;

				if (x0 || x1 || y0 || y1)
				{
					cv::Point ps = cv::Point(x, y) + region.tl();
					for (int k = 0; k < stereoEnergy->neighbors.size(); k++)
					{
						cv::Point pt = ps + stereoEnergy->neighbors[k];
						if (region.contains(pt))
							continue;
						if (imageDomain.contains(pt) == false)
							continue;

						// pt is always label0
						float _cost00 = stereoEnergy->computeSmoothnessTerm(currentLabeling.at<Plane>(ps), currentLabeling.at<Plane>(pt), ps, k, mode);
						float _cost10 = stereoEnergy->computeSmoothnessTerm(labeling1.at<Plane>(ps), currentLabeling.at<Plane>(pt), ps, k, mode);

						graph.add_tweights(s, _cost00, _cost10);
					}
				}
			}
		}

		// ee <-> ge
		// ***
		// **@
		// ***
		for (int y = 0; y < region.height; y++){
			for (int x = 0; x < region.width - 1; x++){
				int i = y*region.width + x;
				int j = y*region.width + x + 1;
				float B = cost10[StereoEnergy::NB_GE].at<float>(y, x);
				float C = cost01[StereoEnergy::NB_GE].at<float>(y, x);
				float D = cost00[StereoEnergy::NB_GE].at<float>(y, x);
				graph.add_edge(i, j, B + C - D, 0);
				graph.add_tweights(i, C, 0);
				graph.add_tweights(j, D - C, 0);
			}
		}

		// ee <-> eg
		// ***
		// ***
		// *@*
		for (int y = 0; y < region.height - 1; y++){
			for (int x = 0; x < region.width; x++){
				int i = y*region.width + x;
				int j = (y + 1)*region.width + x;
				float B = cost10[StereoEnergy::NB_EG].at<float>(y, x);
				float C = cost01[StereoEnergy::NB_EG].at<float>(y, x);
				float D = cost00[StereoEnergy::NB_EG].at<float>(y, x);
				graph.add_edge(i, j, B + C - D, 0);
				graph.add_tweights(i, C, 0);
				graph.add_tweights(j, D - C, 0);
			}
		}

		if (stereoEnergy->neighbors.size() >= 8)
		{
			// ee <-> gg
			// ***
			// ***
			// **@
			for (int y = 0; y < region.height - 1; y++){
				for (int x = 0; x < region.width - 1; x++){
					int i = y*region.width + x;
					int j = (y + 1)*region.width + x + 1;
					float B = cost10[StereoEnergy::NB_GG].at<float>(y, x);
					float C = cost01[StereoEnergy::NB_GG].at<float>(y, x);
					float D = cost00[StereoEnergy::NB_GG].at<float>(y, x);
					graph.add_edge(i, j, B + C - D, 0);
					graph.add_tweights(i, C, 0);
					graph.add_tweights(j, D - C, 0);
				}
			}

			// ee <-> lg
			// ***
			// ***
			// @**
			for (int y = 0; y < region.height - 1; y++){
				for (int x = 1; x < region.width; x++){
					int i = y*region.width + x;
					int j = (y + 1)*region.width + x - 1;
					float B = cost10[StereoEnergy::NB_LG].at<float>(y, x);
					float C = cost01[StereoEnergy::NB_LG].at<float>(y, x);
					float D = cost00[StereoEnergy::NB_LG].at<float>(y, x);
					graph.add_edge(i, j, B + C - D, 0);
					graph.add_tweights(i, C, 0);
					graph.add_tweights(j, D - C, 0);
				}
			}
		}


		double flow = graph.maxflow();

		for (int y = 0; y < region.height; y++){
			for (int x = 0; x < region.width; x++){
				updateMask.at<uchar>(y, x) = graph.what_segment(y*region.width + x) == G::SOURCE ? 255 : 0;
			}
		}

#if 0
		// check if the flow and graph-construciton are correct
		const cv::Point po(M, M);
		cv::Mat newCost = subCurrent.clone();
		proposalCosts.copyTo(newCost, updateMask);
		double _flow = cv::sum(newCost)[0];
		cv::Mat newLabeling_m = currentLabeling_m.clone();
		newLabeling_m(region + po).setTo(label1.toScalar(), updateMask);

		cv::Rect region_m = cv::Rect(region.x - M, region.y - M, region.width + 2 * M, region.height + 2 * M) & imageDomain;

		for (int y = region_m.y; y < region_m.y + region_m.height; y++)
		for (int x = region_m.x; x < region_m.x + region_m.width; x++){
			cv::Point ps(x, y);

			for (int i = 0; i < stereoEnergy->neighbors.size(); i++)
			{
				cv::Point n = stereoEnergy->neighbors[i];
				cv::Point pt = ps + n;
				if (imageDomain.contains(pt) == false)
					continue;
				if (region.contains(ps) == false && region.contains(pt) == false)
					continue;

				// if forward
				if (n.y > 0 || (n.y == 0 && n.x > 0))
				{
					_flow += stereoEnergy->computeSmoothnessTerm(newLabeling_m.at<Plane>(ps + po), newLabeling_m.at<Plane>(pt + po), ps, pt);
				}
			}
		}
		printf("f1 = %lf, f2 = %lf, df = %lf\n", flow, _flow, (flow - _flow) / _flow);
		CV_Assert(fabs(flow - _flow) <= _flow * 1e-5);
#endif

		return flow;
	}
	double expansionMoveBK(cv::Mat updateMask, const Plane label1, const cv::Rect region, const cv::Mat proposalCosts, int mode = 0)
	{
		std::vector<cv::Mat> cost00;
		std::vector<cv::Mat> cost01;
		std::vector<cv::Mat> cost10;

		cv::Mat currentCost = currentCost_[mode];
		cv::Mat currentLabeling = currentLabeling_[mode];
		cv::Mat currentLabeling_m = currentLabeling_m_[mode];


		stereoEnergy->computeSmoothnessTermsExpansion(currentLabeling_m, label1, region, cost00, cost01, cost10, true, mode);

		int N = region.width * region.height;
		typedef Graph<float, float, double> G;
		G graph(N, 4 * N);

		graph.add_node(N);
		cv::Mat subCurrent = currentCost(region);
		for (int y = 0; y < region.height; y++){
			for (int x = 0; x < region.width; x++){
				int s = y*region.width + x;
				graph.add_tweights(s, subCurrent.at<float>(y, x), proposalCosts.at<float>(y, x));

				bool x0 = x == 0;
				bool x1 = x == region.width - 1;
				bool y0 = y == 0;
				bool y1 = y == region.height - 1;

				if (x0 || x1 || y0 || y1)
				{
#if 0
					if (x0)
						graph.add_tweights(s, cost00[StereoEnergy::NB_LE].at<float>(y, x), cost10[StereoEnergy::NB_LE].at<float>(y, x));
					if (x1)
						graph.add_tweights(s, cost00[StereoEnergy::NB_GE].at<float>(y, x), cost10[StereoEnergy::NB_GE].at<float>(y, x));
					if (y0)
						graph.add_tweights(s, cost00[StereoEnergy::NB_EL].at<float>(y, x), cost10[StereoEnergy::NB_EL].at<float>(y, x));
					if (y1)
						graph.add_tweights(s, cost00[StereoEnergy::NB_EG].at<float>(y, x), cost10[StereoEnergy::NB_EG].at<float>(y, x));

					if (x0 || y0)
						graph.add_tweights(s, cost00[StereoEnergy::NB_LL].at<float>(y, x), cost10[StereoEnergy::NB_LL].at<float>(y, x));
					if (x1 || y0)
						graph.add_tweights(s, cost00[StereoEnergy::NB_GL].at<float>(y, x), cost10[StereoEnergy::NB_GL].at<float>(y, x));
					if (x0 || y1)
						graph.add_tweights(s, cost00[StereoEnergy::NB_LG].at<float>(y, x), cost10[StereoEnergy::NB_LG].at<float>(y, x));
					if (x1 || y1)
						graph.add_tweights(s, cost00[StereoEnergy::NB_GG].at<float>(y, x), cost10[StereoEnergy::NB_GG].at<float>(y, x));
#else
					cv::Point ps = cv::Point(x, y) + region.tl();
					for (int k = 0; k < stereoEnergy->neighbors.size(); k++)
					{
						cv::Point pt = ps + stereoEnergy->neighbors[k];
						if (region.contains(pt))
							continue;
						if (imageDomain.contains(pt) == false)
							continue;

						// pt is always label0
						float _cost00 = stereoEnergy->computeSmoothnessTerm(currentLabeling.at<Plane>(ps), currentLabeling.at<Plane>(pt), ps, k, mode);
						float _cost10 = stereoEnergy->computeSmoothnessTerm(label1, currentLabeling.at<Plane>(pt), ps, k, mode);

						graph.add_tweights(s, _cost00, _cost10);
					}
#endif
				}
			}
		}

		// ee <-> ge
		// ***
		// **@
		// ***
		for (int y = 0; y < region.height; y++){
			for (int x = 0; x < region.width - 1; x++){
				int i = y*region.width + x;
				int j = y*region.width + x + 1;
				float B = cost10[StereoEnergy::NB_GE].at<float>(y, x);
				float C = cost01[StereoEnergy::NB_GE].at<float>(y, x);
				float D = cost00[StereoEnergy::NB_GE].at<float>(y, x);
				graph.add_edge(i, j, std::max(0.f, B + C - D), 0); // Values B+C-D can be slightly negative due to numerical errors
				graph.add_tweights(i, C, 0);
				graph.add_tweights(j, D - C, 0);
			}
		}

		// ee <-> eg
		// ***
		// ***
		// *@*
		for (int y = 0; y < region.height - 1; y++){
			for (int x = 0; x < region.width; x++){
				int i = y*region.width + x;
				int j = (y + 1)*region.width + x;
				float B = cost10[StereoEnergy::NB_EG].at<float>(y, x);
				float C = cost01[StereoEnergy::NB_EG].at<float>(y, x);
				float D = cost00[StereoEnergy::NB_EG].at<float>(y, x);
				graph.add_edge(i, j, std::max(0.f, B + C - D), 0);
				graph.add_tweights(i, C, 0);
				graph.add_tweights(j, D - C, 0);
			}
		}

		if (stereoEnergy->neighbors.size() >= 8)
		{
			// ee <-> lg
			// ***
			// ***
			// @**
			for (int y = 0; y < region.height - 1; y++){
				for (int x = 1; x < region.width; x++){
					int i = y*region.width + x;
					int j = (y + 1)*region.width + x - 1;
					float B = cost10[StereoEnergy::NB_LG].at<float>(y, x);
					float C = cost01[StereoEnergy::NB_LG].at<float>(y, x);
					float D = cost00[StereoEnergy::NB_LG].at<float>(y, x);
					graph.add_edge(i, j, std::max(0.f, B + C - D), 0);
					graph.add_tweights(i, C, 0);
					graph.add_tweights(j, D - C, 0);
				}
			}

			// ee <-> gg
			// ***
			// ***
			// **@
			for (int y = 0; y < region.height - 1; y++){
				for (int x = 0; x < region.width - 1; x++){
					int i = y*region.width + x;
					int j = (y + 1)*region.width + x + 1;
					float B = cost10[StereoEnergy::NB_GG].at<float>(y, x);
					float C = cost01[StereoEnergy::NB_GG].at<float>(y, x);
					float D = cost00[StereoEnergy::NB_GG].at<float>(y, x);
					graph.add_edge(i, j, std::max(0.f, B + C - D), 0);
					graph.add_tweights(i, C, 0);
					graph.add_tweights(j, D - C, 0);
				}
			}

		}

		double flow = graph.maxflow();

		for (int y = 0; y < region.height; y++){
			for (int x = 0; x < region.width; x++){
				updateMask.at<uchar>(y, x) = graph.what_segment(y*region.width + x) == G::SOURCE ? 255 : 0;
			}
		}

#if 0
		// check if the flow and graph-construciton are correct
		const cv::Point po(M, M);
		cv::Mat newCost = subCurrent.clone();
		proposalCosts.copyTo(newCost, updateMask);
		double _flow = cv::sum(newCost)[0];
		cv::Mat newLabeling_m = currentLabeling_m.clone();
		newLabeling_m(region + po).setTo(label1.toScalar(), updateMask);

		cv::Rect region_m = cv::Rect(region.x - M, region.y - M, region.width + 2 * M, region.height + 2 * M) & imageDomain;

		for (int y = region_m.y; y < region_m.y + region_m.height; y++)
		for (int x = region_m.x; x < region_m.x + region_m.width; x++){
			cv::Point ps(x, y);

			for (int i = 0; i < pmEnergy->neighbors.size(); i++)
			{
				cv::Point n = pmEnergy->neighbors[i];
				cv::Point pt = ps + n;
				if (imageDomain.contains(pt) == false)
					continue;
				if (region.contains(ps) == false && region.contains(pt) == false)
					continue;

				// if forward
				if (n.y > 0 || (n.y == 0 && n.x > 0))
				{
					_flow += pmEnergy->computeSmoothnessTerm(newLabeling_m.at<Plane>(ps + po), newLabeling_m.at<Plane>(pt + po), ps, pt);
				}
			}
		}
		printf("f1 = %lf, f2 = %lf, df = %lf\n", flow, _flow, (flow - _flow) / _flow);
		CV_Assert(fabs(flow - _flow) <= _flow * 1e-5);
#endif

		return flow;
	}
};

