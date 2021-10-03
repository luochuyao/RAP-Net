# RAP-Net

# RAP-Net

This is a Pytorch implementation of RAP-Net, a novel attention model for radar echo prediction (precipitation nowcasting) as described in the following paper:

RAP-Net: Region Attention Predictive Network for Precipitation Nowcasting, by Chuyao Luo, Zheng Zhang, Rui Ye, Xutao Li, Yunming Ye.

# Setup
Required python libraries: torch (>=1.7.0) + opencv + numpy.
Tested in ubuntu + nvidia 3090 Ti with cuda (>=11.0).

# Datasets
We conduct experiments on CIKM AnalytiCup 2017 datasets: [CIKM_AnalytiCup_Address](https://tianchi.aliyun.com/competition/entrance/231596/information) or [CIKM_Rardar](https://drive.google.com/drive/folders/1IqQyI8hTtsBbrZRRht3Es9eES_S4Qv2Y?usp=sharing) 

# Training
Use any '.py' script in folds of experiment and ablation_study to train these models. To train the proposed model on the radar, we can simply run the CIKM_rap_net.py in the fold of experiment

You might want to change the parameter and setting, you can change the details of variable ‘args’ in each files for each model

The preprocess method and data root path can be modified in the data/data_iterator.py file.







