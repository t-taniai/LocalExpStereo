The info.txt files specify the scaling factor and max disparity of ground truth.
```
4
59
```
The former value is used to devide intensities of groundtruth.png to get disparity values.
Therefore, if an intensity of groundtruth.png is 18, then 4.5 = 18/4 is the actual disparity in the above case.

The latter value defines the maximum disparity allowed during the inference.
In the above case, disparities range in [0, 58].
