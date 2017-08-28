
#define _CRT_SECURE_NO_WARNINGS

#include <omp.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include "FastGCStereo.h"
#include "Evaluator.h"
#include "ArgsParser.h"
#include "CostVolumeEnergy.h"
#include "Utilities.hpp"
#include <direct.h>

struct Options
{
	std::string outputDir = "./results";
	std::string targetDir = "E:/dataset";

	int iterations = 7;
	int pmIterations = 2;
	bool doDual = false;

	float smooth_weight = 1.0;
	int filterRadious = 20;

	int threadNum = -1;

	void loadOptionValues(ArgsParser& argParser)
	{
		argParser.TryGetArgment("outputDir", outputDir);
		argParser.TryGetArgment("targetDir", targetDir);

		argParser.TryGetArgment("filterRadious", filterRadious);
		argParser.TryGetArgment("smooth_weight", smooth_weight);
		argParser.TryGetArgment("pmInit", pmIterations);
		argParser.TryGetArgment("iterations", iterations);
		argParser.TryGetArgment("threadNum", threadNum);
		argParser.TryGetArgment("doDual", doDual);
	}

	void printOptionValues(FILE * out = stdout)
	{
		fprintf(out, "----------- parameter settings -----------\n", outputDir.c_str());
		fprintf(out, "outputDir      : %s\n", outputDir.c_str());
		fprintf(out, "targetDir      : %s\n", targetDir.c_str());

		fprintf(out, "threadNum      : %d\n", threadNum);
		fprintf(out, "doDual         : %d\n", (int)doDual);
		fprintf(out, "pmIterations   : %d\n", pmIterations);
		fprintf(out, "iterations     : %d\n", iterations);
		
		fprintf(out, "filterRadious  : %d\n", filterRadious);
		
		fprintf(out, "smooth_weight  : %f\n", smooth_weight);
	}
};

const Parameters paramsBF = Parameters(20, 20, "BF", 10);
const Parameters paramsGF = Parameters(1.0, 20, "GF", 0.0001);

struct Calib
{
	float cam0[3][3];
	float cam1[3][3];
	float doffs;
	float baseline;
	int width;
	int height;
	int ndisp;
	int isint;
	int vmin;
	int vmax;
	float dyavg;
	float dymax;
	//cam0 = [2852.758 0 1424.085; 0 2852.758 953.053; 0 0 1]
	//cam1 = [2852.758 0 1549.445; 0 2852.758 953.053; 0 0 1]
	//doffs = 125.36
	//baseline = 178.089
	//width = 2828
	//height = 1924
	//ndisp = 260
	//isint = 0
	//vmin = 36
	//vmax = 218
	//dyavg = 0.408
	//dymax = 1.923

	Calib(std::string filename)
		: doffs(0)
		, baseline(0)
		, width(0)
		, height(0)
		, ndisp(0)
		, isint(0)
		, vmin(0)
		, vmax(0)
		, dyavg(0)
		, dymax(0)
	{
		FILE* fp = fopen(filename.c_str(), "r");
		char buff[512];

		if (fp != nullptr)
		{
			if (fgets(buff, sizeof(buff), fp) != nullptr) sscanf(buff, "cam0 = [%f %f %f; %f %f %f; %f %f %f]\n", &cam0[0][0], &cam0[0][1], &cam0[0][2], &cam0[1][0], &cam0[1][1], &cam0[1][2], &cam0[2][0], &cam0[2][1], &cam0[2][2]);
			if (fgets(buff, sizeof(buff), fp) != nullptr) sscanf(buff, "cam1 = [%f %f %f; %f %f %f; %f %f %f]\n", &cam1[0][0], &cam1[0][1], &cam1[0][2], &cam1[1][0], &cam1[1][1], &cam1[1][2], &cam1[2][0], &cam1[2][1], &cam1[2][2]);
			if (fgets(buff, sizeof(buff), fp) != nullptr) sscanf(buff, "doffs = %f\n", &doffs);
			if (fgets(buff, sizeof(buff), fp) != nullptr) sscanf(buff, "baseline = %f\n", &baseline);
			if (fgets(buff, sizeof(buff), fp) != nullptr) sscanf(buff, "width = %d\n", &width);
			if (fgets(buff, sizeof(buff), fp) != nullptr) sscanf(buff, "height = %d\n", &height);
			if (fgets(buff, sizeof(buff), fp) != nullptr) sscanf(buff, "ndisp = %d\n", &ndisp);
			if (fgets(buff, sizeof(buff), fp) != nullptr) sscanf(buff, "isint = %d\n", &isint);
			if (fgets(buff, sizeof(buff), fp) != nullptr) sscanf(buff, "vmin = %d\n", &vmin);
			if (fgets(buff, sizeof(buff), fp) != nullptr) sscanf(buff, "vmax = %d\n", &vmax);
			if (fgets(buff, sizeof(buff), fp) != nullptr) sscanf(buff, "dyavg = %f\n", &dyavg);
			if (fgets(buff, sizeof(buff), fp) != nullptr) sscanf(buff, "dymax = %f\n", &dymax);
			fclose(fp);
		}
	}
};

void fillOutOfView(cv::Mat& volume, int mode)
{
	int D = volume.size.p[0];
	int H = volume.size.p[1];
	int W = volume.size.p[2];

	if (mode == 0)
	for (int d = 1; d < D; d++)
	for (int y = 0; y < H; y++)
	{
		auto p = volume.ptr<float>(d, y);
		auto q = p + d;
		float v = *q;
		while (p != q){
			*p = v;
			p++;
		}
	}
	else
	for (int d = 1; d < D; d++)
	for (int y = 0; y < H; y++)
	{
		auto q = volume.ptr<float>(d, y) + W;
		auto p = q - d;
		float v = p[-1];
		while (p != q){
			*p = v;
			p++;
		}
	}
}

void MidV2(const std::string inputDir, const std::string outputDir, const Options& options)
{
	int ndisp = 59, gt_scale = 4;
	{
		FILE *fp = fopen((inputDir + "info.txt").c_str(), "r");
		if (fp != nullptr) {
			fscanf(fp, "%d", &gt_scale); // Scaling factor of intensity to disparity for ground truth
			fscanf(fp, "%d", &ndisp);
			fclose(fp);
		}
	}
	double maxdisp = ndisp - 1;
	double vdisp = 0;
	printf("ndisp = %d\n", ndisp);

	cv::Mat imL = cv::imread(inputDir + "imL.png");
	cv::Mat imR = cv::imread(inputDir + "imR.png");

	if (imL.empty() || imR.empty()) {
		printf("Files imL.png and imR.png not found in\n");
		printf("%s\n", inputDir.c_str());
		return;
	}

	double errorThresh = 0.5;

	cv::Mat dispGT = cv::imread(inputDir + "groundtruth.png", cv::IMREAD_GRAYSCALE);
	cv::Mat nonocc = cv::imread(inputDir + "nonocc.png", cv::IMREAD_GRAYSCALE);
	if (!nonocc.empty()) nonocc = nonocc == 255;

	if (dispGT.empty()) dispGT = cv::Mat_<float>::zeros(imL.size());
	if (nonocc.empty()) nonocc = cv::Mat_<uchar>::zeros(imL.size());

	dispGT.convertTo(dispGT, CV_32F, 1.0 / gt_scale);

	Parameters param = paramsGF;
	param.windR = options.filterRadious;
	param.lambda = options.smooth_weight;

	{
		_mkdir((outputDir + "debug").c_str());

		Evaluator *eval = new Evaluator(dispGT, nonocc, 255.0 / (maxdisp), "result", outputDir + "debug\\");
		eval->setPrecision(1.0 / gt_scale);
		eval->showProgress = false;
		eval->setErrorThreshold(errorThresh);

		FastGCStereo stereo(imL, imR, param, maxdisp, 0, vdisp);
		stereo.saveDir = outputDir + "debug\\";
		stereo.setEvaluator(eval);

		IProposer* prop1 = new ExpansionProposer(1);
		IProposer* prop2 = new RandomProposer(7, maxdisp);
		IProposer* prop3 = new ExpansionProposer(2);
		IProposer* prop4 = new RansacProposer(1);
		stereo.addLayer(5, { prop1, prop4, prop2 });
		stereo.addLayer(15, { prop3, prop4 });
		stereo.addLayer(25, { prop3, prop4 });

		cv::Mat labeling, rawdisp;
		if (options.doDual)
			stereo.run(options.iterations, { 0, 1 }, options.pmIterations, labeling, rawdisp);
		else
			stereo.run(options.iterations, { 0 }, options.pmIterations, labeling, rawdisp);

		delete prop1;
		delete prop2;
		delete prop3;
		delete prop4;

		cvutils::io::save_pfm_file(outputDir + "disp0.pfm", stereo.getEnergyInstance().computeDisparities(labeling));
		if (options.doDual)
			cvutils::io::save_pfm_file(outputDir + "disp0raw.pfm", stereo.getEnergyInstance().computeDisparities(rawdisp));

		{
			FILE *fp = fopen((outputDir + "time.txt").c_str(), "w");
			if (fp != nullptr) { fprintf(fp, "%lf\n", eval->getCurrentTime()); fclose(fp); }
		}
		delete eval;
	}
}

void MidV3(const std::string inputDir, const std::string outputDir, const Options& options)
{
	Calib calib(inputDir + "calib.txt");
	double maxdisp = calib.ndisp - 1;
	printf("ndisp = %d\n", calib.ndisp);
	printf("dyavr = %lf\n", calib.dyavg);
	printf("dymax = %lf\n", calib.dymax);

	cv::Mat imL = cv::imread(inputDir + "im0.png");
	cv::Mat imR = cv::imread(inputDir + "im1.png");

	if (imL.empty() || imR.empty()) {
		printf("Files im0.png and im1.png not found in\n");
		printf("%s\n", inputDir.c_str());
		return;
	}

	double errorThresh = 2.0;
	if (cvutils::contains(inputDir, "trainingQ") || cvutils::contains(inputDir, "testQ"))
		errorThresh = errorThresh / 4.0;
	else if (cvutils::contains(inputDir, "trainingH") || cvutils::contains(inputDir, "testH"))
		errorThresh = errorThresh / 2.0;

	cv::Mat dispGT = cvutils::io::read_pfm_file(inputDir + "disp0GT.pfm");
	cv::Mat nonocc = cv::imread(inputDir + "mask0nocc.png", cv::IMREAD_GRAYSCALE);
	if (!nonocc.empty()) nonocc = nonocc == 255;

	if (dispGT.empty()) dispGT = cv::Mat_<float>::zeros(imL.size());
	if (nonocc.empty()) nonocc = cv::Mat_<uchar>::zeros(imL.size());

	Parameters param = paramsGF;
	param.windR = options.filterRadious;
	param.lambda = options.smooth_weight;
	param.th_col = 0.5; // tau_CNN in the paper

	int sizes[] = { calib.ndisp, imL.rows, imL.cols };
	cv::Mat volL = cv::Mat_<float>(3, sizes);
	cv::Mat volR = cv::Mat_<float>(3, sizes);
	cvutils::io::loadMatBinary(inputDir + "im0.acrt", volL, false);
	cvutils::io::loadMatBinary(inputDir + "im1.acrt", volR, false);
	fillOutOfView(volL, 0);
	fillOutOfView(volR, 1);

	{
		_mkdir((outputDir + "debug").c_str());

		Evaluator *eval = new Evaluator(dispGT, nonocc, 255.0 / (maxdisp), "result", outputDir + "debug\\");
		eval->setPrecision(-1);
		eval->showProgress = false;
		eval->setErrorThreshold(errorThresh);

		FastGCStereo stereo(imL, imR, param, maxdisp);
		stereo.setStereoEnergyCPU(std::make_unique<CostVolumeEnergy>(imL, imR, volL, volR, param, maxdisp));
		stereo.saveDir = outputDir + "debug\\";
		stereo.setEvaluator(eval);

		int w = imL.cols;
		IProposer* prop1 = new ExpansionProposer(1);
		IProposer* prop2 = new RandomProposer(7, maxdisp);
		IProposer* prop3 = new ExpansionProposer(2);
		IProposer* prop4 = new RansacProposer(1);
		stereo.addLayer(w * 0.01, { prop1, prop4, prop2 });
		stereo.addLayer(w * 0.03, { prop3, prop4 });
		stereo.addLayer(w * 0.09, { prop3, prop4 });

		cv::Mat labeling, rawdisp;
		if (options.doDual)
			stereo.run(options.iterations, { 0, 1 }, options.pmIterations, labeling, rawdisp);
		else
			stereo.run(options.iterations, { 0 }, options.pmIterations, labeling, rawdisp);

		delete prop1;
		delete prop2;
		delete prop3;
		delete prop4;

		cvutils::io::save_pfm_file(outputDir + "disp0.pfm", stereo.getEnergyInstance().computeDisparities(labeling));
		if(options.doDual) 
			cvutils::io::save_pfm_file(outputDir + "disp0raw.pfm", stereo.getEnergyInstance().computeDisparities(rawdisp));

		{
			FILE *fp = fopen((outputDir + "time.txt").c_str(), "w");
			if (fp != nullptr){ fprintf(fp, "%lf\n", eval->getCurrentTime()); fclose(fp); }
		}

		delete eval;
	}
}



int main(int argc, const char** args)
{
	ArgsParser parser(argc, args);
	Options options;
	options.loadOptionValues(parser);
	options.printOptionValues();

	int nThread = omp_get_max_threads();
	unsigned int seed = (unsigned int)time(NULL);
	//seed = 0;
	#pragma omp parallel for
	for (int j = 0; j < nThread; j++)
	{
		srand(seed + j);
		cv::theRNG() = seed + j;
	}

	if (options.threadNum > 0)
		omp_set_num_threads(options.threadNum);
	options.targetDir = "..\\data\\MiddV2\\teddy";

	if (options.outputDir.length())
		_mkdir((options.outputDir).c_str());

	MidV2(options.targetDir + "/", options.outputDir + "/", options);
	printf("\n\n");

	std::string mode;
	if (parser.TryGetArgment("mode", mode) && mode == "MiddV2")
	{
		printf("Running Middlebury V2 mode.\n");
		MidV2(options.targetDir + "/", options.outputDir + "/", options);
	}

	if (parser.TryGetArgment("mode", mode) && mode == "MiddV3")
	{
		printf("Running Middlebury V3 mode.\n");
		printf("This mode assumes MC-CNN matching cost files in targetDir.\n");
		MidV3(options.targetDir + "/", options.outputDir + "/", options);
	}

	return 0;
}

/* Make a bat file....

set bin=%~dp0x64\Release\LocalExpansionStereo.exe
set datasetroot=%~dp0data
set resultsroot=%~dp0results

mkdir "%resultsroot%"
"%bin%" -targetDir "%datasetroot%\MiddV2\cones" -outputDir "%resultsroot%\cones_nogt" -mode MiddV2 -smooth_weight 1
"%bin%" -targetDir "%datasetroot%\MiddV2\cones" -outputDir "%resultsroot%\cones" -mode MiddV2 -smooth_weight 1
"%bin%" -targetDir "%datasetroot%\MiddV3\Adirondack" -outputDir "%resultsroot%\Adirondack" -mode MiddV3 -smooth_weight 0.5
pause;


*/