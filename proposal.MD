# Machine Learning Engineer Nanodegree
## Capstone Proposal
Glenn Mossy 
November 6, 2019

## Proposal - A Machine Learning Based Human Activity Recognizer.
### Domain Background
The goal of activity recognition is to recognize common human activities in real life settings. Human activity recognition (HAR) plays an important role in people’s daily life for its competence in learning profound high-level knowledge about human activity from raw sensor inputs. [1]  

Human Activity Recognitions appealing applications ranging from security-related applications, logistics support, location-based services and exploitation of Ambient Intelligence (AmI) by helping handicapped or elderly people to live more independently, rate health levels, activate household controls, appliances and provide safety. With knowledge from the model, results can be used to perform Ambient Assisted Living.  Better predicting of human behavior means we can proactively deliver specilized programs and targeted interventions to the populations that are most in need. HAR is very multifaceted, useful many fields and may be referred to as goal recognition, behavior recognition, location estimation, etc. 

Human Activity Recognition aims to identify the actions carried out by a person provided by a set of observations from the person's movements and surrounding environment, and then model a wide range of human activities. Recognition can be accomplished by exploiting the information retrieved from various sources such as body-worn sensors.  It is believed that by empowering computer monitoring to monitor the behavior of agents then these computers would become and suited to act on our behalf. 
 
This project is based on sensor that are based on a body mounted accelerometer and gyroscope but many additional sensors, such as microphones, motion video capture, human physiological signals or vital signs could be later added to this project in order to capture data to provide features to the model that would improve the accuracy, provide better models, and increase the number of activities that can be understood, with the ultimate goal of providing a real-time prediction of each and every activity that the person is doing.

My interest in this field stems from my work with AT&T Public sector where I am a Senior Solutions Architect providing technical leadership on Technology Innovation initiatives for our Nations Department of Defense, DISA, Army and Cyber programs.  In this role I support multi-vendor, high risk/reward integration projects, including rapid software development, enterprise networking and automation that enables business processes, and provides operations automation. I have participated in AT&T Software Symposium Hackathons in which the project was to provide assisted living support to wheelchair bound, severely disabled patients.  I am also an adjunct instruction at Frederick Community College teaching students how to program Arduino microcontrollers and sensors to interact with the physical environment. 

### Problem Statement
Human Activity Recognition or HAR for short, is the problem of predicting what a person is doing based on a time series recording of their movement using sensors. The idea is that once the subject’s activity is recognized and known, it can be saved in a model that can then be used to recognize future those activities with high accuracy when applied, and then provide useful assistance with the result. 

Movements that we will study in this project will be normal indoor activities such as standing, sitting, and stair climbing. Sensors are body mounted and record accelerometer and gyroscope data in three dimensions (x, y, z).  The objective is to classify activities into one of the six activities performed.

It is a challenging problem because there is no clear analytical way to relate the sensor data to specific actions in a general way. It is technically challenging because of the large volume of sensor data collected (e.g. tens or hundreds of observations per second) and the classical use of hand crafted features and heuristics from this data in developing predictive models.
Classical approaches to the problem involve hand crafting features from the time series data based on fixed-size windows and training machine learning models, such as ensembles of decision trees or random forest methods.

### Datasets and Inputs
This project uses the dataset “A Public Domain Dataset for Human Activity Recognition Using Smartphones" that was collected and made available by David Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz [1]. It is freely available from the UCI Machine Learning repository https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones and from Kaggle:  https://www.kaggle.com/uciml/human-activity-recognition-with-smartphones

The dataset is comprised of un-calibrated accelerometer data from 30 different subjects ages ranging from 19-48 years, each performing 6 activities. Each person performed six activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) wearing a smartphone (Samsung Galaxy S II) on the waist. Using its embedded accelerometer and gyroscope, the data captured 3-axial linear acceleration and 3-axial angular velocity at a constant rate of 50Hz.  The obtained dataset has been randomly partitioned into two sets, "train.csv", where 70% of the volunteers was selected for generating the training data and 30% the test data "test.csv".

The sensor signals (accelerometer and gyroscope) were pre-processed by applying noise filters and then sampled in fixed-width sliding windows of 2.56 sec and 50% overlap (128 readings/window). The sensor acceleration signal, which has gravitational and body motion components, was separated using a Butterworth low-pass filter into body acceleration and gravity. The gravitational force is assumed to have only low frequency components, therefore a filter with 0.3 Hz cutoff frequency was used. From each window, a vector of features was obtained by calculating variables from the time and frequency domain.

For each record in the dataset the following is provided:

     -Triaxial acceleration from the accelerometer (total acceleration) and the estimated body acceleration.

     -Triaxial Angular velocity from the gyroscope.

     -A 561-feature vector with time and frequency domain variables.

     -Its activity label.

     -An identifier of the subject who carried out the experiment.

### Solution Statement
The solution is a classification model capable of predicting the human activity recognition first using traditional supervised learning models using classification algorithms such as logistic regression, random forests, support vector machines, then trying an artificial neural network (ANN). Models will be built and then used to infer the activity from test data.  The accuracy will be evaluated. 

### Benchmark Model
The benchmark model that will be used will be a logistic regression model for a classification to develop a baseline activity recognition model.

### Evaluation Metrics
The objective of this project is to predict the activity performed by the subject accurately, based on accelerometer and gyroscope readings, so misclassification rate is chosen as the assessment measure to compare the performance of different models. 

### Project Design
To develop an activity recognition model, we will make use of the  sensor data that will first be aggregated into examples that summarize the user activity. Supervised learning will be used with the training data and validation methods will be used to create a models first beginning with the benchmark statiscal model, logistic regression.  The results will be to accurately classify 6 different activity types: walking, walking upstairs, walking downstairs, sitting, standing, and laying.
Programming language: Python 3.6
Library: Pandas, Numpy, Scikit-learn, various classifiers.

The workflow for the solution is as follows:
#### Collect
1.	Problem Description
2.	Load Datasets and prepare training set and test set
- collect the labeled raw accelerometer and gyroscope data
#### Explore
- Establish the basic understanding of the dataset, perform basic cleaning and processing, transform the data into examples and explore the data to show the distribution information of the dataset
3.	Plot Traces for a Single Subject
4.	Plot Total Activity Durations
5.	Plot Traces for Each Subject
6.	Plot Histograms of Trace Observations.
7. Determine how to calibrate the data. 
#### Extract Features
- Devise new features based on the given features, some intuitions. 
7. Create a generator to break the data to be trained into batches. 
8.	Determine Approaches to Modeling the Problem
- divide the data into second segments and then generate features that were based on the readings.
#### Train and Classify with benchmark
- Work will be done to classify the activities using a classifier from sklearn such as logistic regression as a benchmark and fine tune the model's hyperparameters. 
9. Split the data into Training and Test sets.

#### Improve the classification with an ANN
The solution then proposes a Artificial neural network designed to take a
10. design a classifier that will be used for the Artificial Neural Network (ANN).

#### Test and Evaluate
11. Finally, use the Test data results to evaluate the performance of the classifier output.
The goal is to show a comparison with the traditional classification models to an ANN, and understanding any improvements in terms of computational costs while maintaining similar accuracy.

**Acknowledgements and References:**
[1] Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using Smartphones. https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones, 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013. 

[2] Jindong Wang, Yiqiang Chen, Shuji Hao, Xiaohui Peng, Lisha Hu, Deep Learning for Sensor-based Activity Recognition: A Survey, https://arxiv.org/abs/1707.03502, arXiv:1707.03502v2 [cs.CV],  14 Dec 2017

[3] Mohammad Abu Alsheikh and Ahmed Selim and Dusit Niyato Linda Doyle and Shaowei Lin and Hwee-Pink Tan, Deep Activity Recognition Models with Triaxial Accelerometers, https://arxiv.org/abs/1511.04664, arXiv:1511.04664v2 [cs.LG],  25 Oct 2016 https://arxiv.org/pdf/1511.04664.pdf

[4] Kwapisz J., Weiss G., Samuel A. Moore, S., Activity Recognition using Cell Phone Accelerometers, http://www.cis.fordham.edu/wisdm/includes/files/sensorKDD-2010.pdf, 2010

[5] Calatroni, Alberto; Roggen, Daniel; Tröster, Gerhard, A methodology to use unknown new sensors for activity
recognition by leveraging sporadic interactions with primitive sensors and behavioral assumptions,
https://doi.org/10.3929/ethz-a-006286309, 2010


**Links**
https://www.kaggle.com/uciml/human-activity-recognition-with-smartphones
https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
https://www.neuraldesigner.com/learning/examples/activity-recognition
https://www.youtube.com/watch?v=XOEN9W05_4A
