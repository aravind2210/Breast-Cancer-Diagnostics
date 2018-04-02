# Breast-Cancer-Diagnostics

## Dataset Overview

The Breast Cancer (Wisconsin) dataset used in this paper is publicly available in the UCI machine learning repository and was created by Dr. William H. Wolberg, Dr. W. Nick Street, and Olvi L. Mangasarian. The doner is W. Nick Street. The dataset is created by Dr. Wolberg by taking suspected tumor samples via a thin needle from patient’s solid breast masses and the samples are placed on dent-shaped glass slides. The slide is dent shaped in order to distinguish the nearby cells with the tumor cells. The collected tissue samples are then examined under microscope and features are computed from the digitized image of a fine needle aspirate (FNA) of breast mass. This is done via a graphical computer program, which is capable of performing the analysis of cytological features based on a digital scan.
The program uses a curve fitting algorithm to compute ten real-valued features for each cell nucleus and then, it calculates the mean value, extreme value(worst) and standard error of each feature for the image.

The dataset is a classification problem consisting of 569 instances of which 357 of the observations are benign and 212 are malignant, where each one represents FNA test measurements for one diagnosis case. In this dataset, there are 32 attributes for each instance, where the first two attributes correspond to a unique identification number and the diagnosis status (benign / malignant). The rest 30 features are computations for ten real-valued features, along with their mean, standard error, and the mean of the three largest values ("worst" value) for each cell nucleus respectively.


