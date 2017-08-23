# Continuous 3D Label Stereo Matching using Local Expansion Moves

This is an implementatioin of a stereo matching method described in

```
@article{Taniai16,
  author    = {Tatsunori Taniai and
               Yasuyuki Matsushita and
               Yoichi Sato and
               Takeshi Naemura},
  title     = {{Continuous Stereo Matching using Local Expansion Moves}},
  journal   = {CoRR (arXiv)},
  volume    = {abs/1603.08328},
  year      = {2016},
  url       = {http://arxiv.org/abs/1603.08328},
}
```

If you use our code, please cite the above paper. (Note that we have slightly changed the title in the latest preprint v2 in 2017 from the above v1 in 2016). We also encourage to cite the following conference paper too.

```
@inproceedings{Taniai14,
  author    = {Tatsunori Taniai and
               Yasuyuki Matsushita and
               Takeshi Naemura},
  title     = {{Continuous Stereo Matching using Locally Shared Labels}},
  booktitle = {{IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}},
  year      = {2014},
  pages     = {1613--1620},
}
```
See also our project site http://taniai.space/projects/stereo/ and paper https://arxiv.org/abs/1603.08328.

## Running environment
- Visual Studio 2017 Community
- OpenCV 3.02 (will be automatically installed via NuGet)
- Maxflow code v3.01 by Boykov and Kolmogorov (http://vision.csd.uwo.ca/code/)

## How to Run?
1. Download and extract maxflow source code to "maxflow" directory. 
2. Download and extract an example dataset (see Adirondack below) in "data/MiddV3/Adirondack".
3. Build the solution with release mode.
4. Run demo.bat file. Results will be saved in "results/cones", "results/teddy", and "results/Adirondack".

## Options
- -mode MiddV2: Use settings for Middlebury V2. Assume imL.png and imR.png, etc. 
- -mode MiddV3: Use settings for Middlebury V3. Assume im0.png and im1.png, etc. with MC-CNN matching cost files.
- -targetDir {string}: Directory that contains target image pairs.
- -outputDir {string}: Directory for saving results. disp0.pfm is the primary result. Intermediate results are also saved in "debug" sub-directory.
- -doDual {0,1}: Estimate left and right disparities and do post-processing using consistency check.
- -iterations {int}: Number of main iterations.
- -pmIterations {int}: Number of initial iterations performed before main iterations without smoothness terms (this accelerates inference).
- -smooth_weight {float}: Smoothness weight (lambda in the paper).
- -filterRedious {int}: The redius of matching windows (ie, filterRedious/2 is the kernel radius of guided image filter).

## Updates
- The function of initial iterations (option: pmIterations) is added to accelerate the inference.
- The implementation of guided image filter has been improved from the paper, which reduces the running times by half.

## Pre-computed MC-CNN matching costs
We use matching cost volumes computed by MC-CNN-acrt (https://github.com/jzbontar/mc-cnn).
We provide pre-computed matching cost data for 30 test and training image pairs of Middlebury benchmark V3.
For demonstration, please use Adirondack below that contains image pairs, calibration data, and ground truth.
- trainingH (23.0 GB): http://www.hci.iis.u-tokyo.ac.jp/datasets/data/LocalExpStereo/trainingH.rar 
- testH (16.7 GB): http://www.hci.iis.u-tokyo.ac.jp/datasets/data/LocalExpStereo/testH.rar
- Adirondack (1.2 GB): http://www.hci.iis.u-tokyo.ac.jp/datasets/data/LocalExpStereo/Adirondack.zip

Note that these matching costs are raw outputs from CNNs without cross-based filter and SGM aggregation.
