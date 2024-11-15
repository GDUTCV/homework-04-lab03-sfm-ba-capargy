# CS4277/CS5477: Structure from Motion and Bundle Adjustment

## Setting Up

If you are using Anaconda, you can run the following lines to setup:
```bash
conda create -n sfm python==3.7.6
conda activate sfm
pip  install -r requirements.txt
```

## Running Scripts
To run the scripts:
```bash
python preprocess.py --dataset temple  # performs preprocessing for temple dataset
python preprocess.py --dataset mini-temple  # performs preprocessing for mini-temple dataset
python sfm.py --dataset temple # performs structure from motion without bundle adjustment 
python sfm.py --dataset mini-temple --ba # performs structure from motion with bundle adjustment on mini-temple dataset
python sfm.py --dataset mini-temple # performs structure from motion without bundle adjustment on mini-temple dataset
```

To visualize, run:
```bash
python visualize.py --dataset mini-temple  # visualize 3d point cloud from reconstruction.
```

update:

The first time I cloned the homework from GitHub, I found that the solutions were already provided in the assignment. At that moment, I was at my wits' end. So my next step was to try to understand each function in the code.Maybe you can track the history from the branch and see that I submitted it on the wrong branch, but I made every effort to understand the entire process and tried to modify the code without affecting its functionality.Whatever I hope these words can earn your understanding and mercy.

重建文件结果分成了几部分：
暴力匹配的npy文件
ransac过滤的npy文件
三维重建结果的npy文件
