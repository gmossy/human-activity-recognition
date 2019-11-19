Udacity Machine Learning Capstone Project
# human-activity-recognition
Udacity Machine Learning Capstone - by Glenn Mossy - A Machine Learning Based Human Activity Recognizer


In this README file, I will include a link and instructions on how to the run the project once the data is downloaded. 
The data is provided as a single zip file that is about 75.9 megabytes in size. The direct link for this download is below:
HAPT Data Set, the data set file to download will be "HAPT Data Set.zip".
http://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions
Download the dataset and unzip all files into a new directory in your current working directory named “./data/HAPT Data Set”.

Installations
This project was written in the Anaconda environment and Python 3.7
Motivation
The purpose of this project was to analyze data generated from smartphone sensors and to build a model to could accurately classify 6 different activity types: walking, walking upstairs, walking downstairs, sitting, standing, and laying, plus 6 postural Transitions, Stand to Sit, Sit to Stand, Sit to Lie, Lie to Sit, Stand to Lie, Lie to Stand. 

This version provides the original raw inertial signals from the smartphone sensors, instead of the ones pre-processed into windows which were provided in version 1. This change was done in order to be able to make online tests with the data. Moreover, the activity labels were updated in order to include postural transitions that were not part of the previous version of the dataset. 

The dataset is then divided in two parts and they can be used separately.  

1. Inertial sensor data 
- Raw triaxial signals from the accelerometer and gyroscope of all the trials with with participants. 
- The labels of all the performed activities.
  
2. Records of activity windows. Each one composed of:
- A 561-feature vector with time and frequency domain variables. 
- Its associated activity label. 
- An identifier of the subject who carried out the experiment.


Files
This dataset repository contains the following:
- 'README.txt'

- 'RawData/acc_expXX_userYY.txt': The raw triaxial acceleration signal for the experiment number XX and associated to the user number YY. Every row is one acceleration sample (three axis) captured at a frequency of 50Hz. 

- 'RawData/gyro_expXX_userYY.txt': The raw triaxial angular speed signal for the experiment number XX and associated to the user number YY. Every row is one angular velocity sample (three axis) captured at a frequency of 50Hz. 

- 'RawData/labels.txt': include all the activity labels available for the dataset (1 per row). 
   Column 1: experiment number ID, 
   Column 2: user number ID, 
   Column 3: activity number ID 
   Column 4: Label start point (in number of signal log samples (recorded at 50Hz))
   Column 5: Label end point (in number of signal log samples)

- 'features_info.txt': Shows information about the variables used on the feature vector.

- 'features.txt': List of all features.

- 'activity_labels.txt': Links the activity ID with their activity name.

- 'Train/X_train.txt': Training set.

- 'Train/y_train.txt': Training labels.

- 'Test/X_test.txt': Test set.

- 'Test/y_test.txt': Test labels.

- 'Train/subject_id_train.txt': Each row identifies the subject who performed the activity for each window sample. Its range is from 1 to 30. 

- 'Test/subject_id_test.txt': Each row identifies the subject who performed the activity for each window sample. Its range is from 1 to 30. 

      
Acknowledgements      
Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using Smartphones. 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013.
UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/index.php
Youtube video, Activity Recognition Experiment Using Smartphone Sensor, https://www.youtube.com/watch?v=XOEN9W05_4A, Oct 19, 2012
Udacity - for setting up the project and course contents.
