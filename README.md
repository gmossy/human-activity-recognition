Udacity Machine Learning Capstone Project
# human-activity-recognition 
Udacity Machine Learning Capstone - by Glenn Mossy - A study of Machine Learning Based Human Activity Recognizers
12/22/2019

In this README file, you will find a link were to download the data, what the data files are named, and instructions on how to use the  data, how to run the project, and Acknowledgements.

You will find the complete report in this link: ![Capstone Report](https://github.com/gmossy/human-activity-recognition/blob/master/Capstone_Project_Report.MD "Capstone Report") and the ![Juypter notebook](https://github.com/gmossy/human-activity-recognition/blob/master/human_activity_recognizer.ipynb "Juypter notebook")
"
## Motivation
The purpose of this project was to analyze data generated from smartphone sensors and to build and evaluate different models that can accurately classify 6 different activity types: walking, walking upstairs, walking downstairs, sitting, standing, and laying, plus 6 postural Transitions, Stand to Sit, Sit to Stand, Sit to Lie, Lie to Sit, Stand to Lie, Lie to Stand. 


## Obtaining the dataset
This project was written in the Anaconda environment and Python 3.7
The data is provided as a single zip file that is about 75.9 megabytes in size. The direct link for this download is as follows:
HAPT Data Set, the data set file to download will be "HAPT Data Set.zip".
Smartphone-Based Recognition of Human Activities and Postural Transitions Data Set Version 2.1
http://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions
Download the dataset and unzip all files into a new directory in your current working directory named “./data/HAPT Data Set”.

I"m using the updated 2nd version which includes postural transitions that were not part of the previous version of the dataset. Moreover, the activity labels were updated in order to include postural transitions that were not part of the previous version of the dataset.  Also the original raw inertial signals from the smartphone sensors are included in a RawData folder.  The raw data is available in order to allow you to make additional online tests with the data.  

## Dataset Description
The dataset is then divided in two parts and they can be used separately.  

1. Inertial sensor data 
- Raw triaxial signals from the accelerometer and gyroscope of all the trials with with participants. 
- The labels of all the performed activities.
  
2. Records of activity windows. Each one composed of:
- A 561-feature vector with time and frequency domain variables. 
- Its associated activity label. 
- An identifier of the subject who carried out the experiment.

## Dataset Files and File details
This dataset repository and root folder "HAPT Data Set" contains the following:
- 'README.txt'

- 'features_info.txt': Shows information about the variables used on the feature vector.
- 'features.txt': List of all features.
- 'activity_labels.txt': Links the activity ID with their activity name for each of the 12 activities.

- 'Train/X_train.txt': Training set.  Includes the activity ID, as column "Y"
- 'Train/y_train.txt': Training labels.
- 'Train/subject_id_train.txt': Each row identifies the subject who performed the activity for each window sample. Its range is from 1 to 30. 

- 'Test/X_test.txt': Test set.
- 'Test/y_test.txt': Test labels.
- 'Test/subject_id_test.txt': Each row identifies the subject who performed the activity for each window sample. Its range is from 1 to 30. 
Informational Summary of the components of the Dataset 
X_train	The shape of the training set dataframe is (7767, 561)   
y_train	The shape of the training set labels dataframe is (7767, 1)  
X_test	The shape of the testing set dataframe is (3162, 561) 
y_test	The shape of the test set labels dataframe is (3162, 1)
Activity_labels	Links the class labels with their activity name.  .
subject_train	Each row identifies the subject who performed the activity for each window sample.
subject_test	Each row identifies the subject who performed the activity for each window sample.
features	List of all featuress.

- 'RawData/acc_expXX_userYY.txt': The raw triaxial acceleration signal for the experiment number XX and associated to the user number YY. Every row is one acceleration sample (three axis) captured at a frequency of 50Hz. 

- 'RawData/gyro_expXX_userYY.txt': The raw triaxial angular speed signal for the experiment number XX and associated to the user number YY. Every row is one angular velocity sample (three axis) captured at a frequency of 50Hz. 

- 'RawData/labels.txt': include all the activity labels available for the dataset (1 per row). 
   Column 1: experiment number ID, 
   Column 2: user number ID, 
   Column 3: activity number ID 
   Column 4: Label start point (in number of signal log samples (recorded at 50Hz))
   Column 5: Label end point (in number of signal log samples)

An example of how to load the data is like this:
## Using the dataset with Pandas
X_train = pd.read_csv('./data/HAPT Data Set/Train/X_train.txt', sep='\s+', header=None)
y_train = pd.read_csv('./data/HAPT Data Set/Train/y_train.txt', sep='\s+', header=None, names='Y')

## Acknowledgements  
Jorge L. Reyes-Ortiz(1,2), Davide Anguita(1), Luca Oneto(1) and Xavier Parra(2) 
1 - Smartlab, DIBRIS - Università  degli Studi di Genova, Genoa (16145), Italy. 
2 - CETpD - Universitat Politècnica de Catalunya. Vilanova i la Geltrú (08800), Spain   
har '@' smartlab.ws 
www.smartlab.ws

Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using Smartphones. 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013.
UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/index.php
Youtube video, Activity Recognition Experiment Using Smartphone Sensor, https://www.youtube.com/watch?v=XOEN9W05_4A, Oct 19, 2012
Udacity - for setting up the project and course contents.
