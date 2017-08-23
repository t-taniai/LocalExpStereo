#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include "Proposer.h"

class LayerManager
{
	const int height;
	const int width;
	const int windowR;

public:
	struct Layer {
		int heightBlocks;
		int widthBlocks;
		int regionUnitSize;
		std::vector<cv::Rect> unitRegions;
		std::vector<cv::Rect> sharedRegions;
		std::vector<cv::Rect> filterRegions;
		std::vector<std::vector<int>> disjointRegionSets;
		//std::vector<std::unique_ptr<IProposer<float>>> proposers;
		std::vector<IProposer*> proposers;
	};
	std::vector<Layer> layers;

	LayerManager& LayerManager::operator=(const LayerManager& obj) {
		this->layers = obj.layers;
		return *this;
	}

	LayerManager(int width, int height, int windowR, int localLabelSetNum)
		: height(height)
		, width(width)
		, windowR(windowR)
	{
		layers = std::vector<Layer>();
	}

	~LayerManager(void)
	{
	}

	void addLayer(int unitRegionSize)
	{
		Layer layer;
		layer.regionUnitSize = unitRegionSize;
		layer.heightBlocks = (height / unitRegionSize) + ((height % unitRegionSize) ? 1 : 0);
		layer.widthBlocks  = (width  / unitRegionSize) + ((width  % unitRegionSize) ? 1 : 0);

		layer.sharedRegions.resize(layer.heightBlocks * layer.widthBlocks);
		layer.filterRegions.resize(layer.heightBlocks * layer.widthBlocks);
		layer.unitRegions.resize(layer.heightBlocks * layer.widthBlocks);
		
		layer.disjointRegionSets.resize(16);
		cv::Rect imageDomain(0, 0, width, height);

		#pragma omp parallel for
		for ( int i = 0; i < layer.heightBlocks; i++ ){
			for ( int j = 0; j < layer.widthBlocks; j++ ){
				int r = i*layer.widthBlocks + j;
				cv::Rect &sharedRegion = layer.sharedRegions[r];
				cv::Rect &filterRegion = layer.filterRegions[r];
				cv::Rect &unitRegion   = layer.unitRegions[r];
				
				unitRegion.x = j * unitRegionSize;
				unitRegion.y = i * unitRegionSize;
				unitRegion.width  = unitRegionSize;
				unitRegion.height = unitRegionSize;
				unitRegion = unitRegion & imageDomain;
				
				sharedRegion.x = (j-1) * unitRegionSize;
				sharedRegion.y = (i-1) * unitRegionSize;
				sharedRegion.width  = unitRegionSize * 3;
				sharedRegion.height = unitRegionSize * 3;
				sharedRegion = sharedRegion & imageDomain;

				filterRegion.x = (j-1) * unitRegionSize - windowR;
				filterRegion.y = (i-1) * unitRegionSize - windowR;
				filterRegion.width  = unitRegionSize * 3 + windowR*2;
				filterRegion.height = unitRegionSize * 3 + windowR*2;
				filterRegion = filterRegion & imageDomain;
			}
		}
		
		for ( int i = 0; i < layer.heightBlocks; i++ ){
			for ( int j = 0; j < layer.widthBlocks; j++ ){
				int r = i*layer.widthBlocks + j;
				layer.disjointRegionSets[(i%4)*4 + (j%4)].push_back(r);
			}
		}
		auto it = layer.disjointRegionSets.begin();
		while (  it != layer.disjointRegionSets.end() ){
			it->shrink_to_fit();
			if ( it->size() == 0 ){
				it = layer.disjointRegionSets.erase(it);
			} else {
				it++;
			}
		}

		layers.push_back(layer);
	}
};

