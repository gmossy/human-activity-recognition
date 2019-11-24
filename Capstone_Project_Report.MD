# Machine Learning Engineer Nanodegree
## Capstone Project
by Glenn Mossy
November 24, 2019

## 1 Definition

### 1.1 Project Overview
In this section, look to provide a high-level overview of the project in layman’s terms. Questions to ask yourself when writing this section:
- _Has an overview of the project been provided, such as the problem domain, project origin, and related datasets or input data?_
- _Has enough background information been given so that an uninformed reader would understand the problem domain and following problem statement?_

The goal of activity recognition is to recognize common human activities in real life settings. Human activity recognition or HAR for short is the problem of predicting what a person is doing based on a trace of their movement while doing a specific activity using sensors.  HAR plays an important role in people’s daily life for its competence in learning profound high-level knowledge about human activity from raw sensor inputs. [1]  

Human Activity Recognition appealing applications ranging from security-related applications, logistics support, location-based services and exploitation of Ambient Intelligence (AmI) by helping handicapped or elderly people to live more independently, rate health levels, activate household controls, appliances and provide safety. With knowledge from the model, results can be used to perform Ambient Assisted Living.  Better predicting of human behavior means we can proactively deliver specilized programs and targeted interventions to the populations that are most in need. HAR is very multifaceted, useful many fields and may be referred to as goal recognition, behavior recognition, location estimation, etc. 

Human Activity Recognition aims to identify the actions carried out by a person provided by a set of observations from the person's movements and surrounding environment, and then model a wide range of human activities. Recognition can be accomplished by exploiting the information retrieved from various sources such as body-worn sensors.  It is believed that by empowering computer monitoring to monitor the behavior of agents then these computers would become and suited to act on our behalf. 
 
This project is based on sensors that are a body mounted accelerometer and gyroscope but many additional sensors, such as microphones, motion video capture, human physiological signals or vital signs could be later added to this project in order to capture data to provide features to the model that would improve the accuracy, provide more data for the models, and increase the number of activities that can be understood, with the ultimate goal of providing a real-time prediction of each and every activity that the person is doing.  This project is based on the work "A Public Domain Dataset for Human Activity Recognition Using Smartphones." by Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. at the 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013. https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

My interest in this field stems from my work with AT&T Public sector where I am a Senior Solutions Architect providing technical leadership on Technology Innovation initiatives for our Nations Department of Defense, DISA, Army and Cyber programs.  In this role I support multi-vendor, high risk/reward integration projects, including rapid software development, enterprise networking and automation that enables business processes, and provides operations automation. I have participated in AT&T Software Symposium Hackathons in which the project was to provide assisted living support to wheelchair bound, severely disabled patients.  I am also an adjunct instruction at Frederick Community College teaching students how to program Arduino microcontrollers and sensors to interact with the physical environment. 

### 1.2 Problem Statement
In this section, you will want to clearly define the problem that you are trying to solve, including the strategy (outline of tasks) you will use to achieve the desired solution. You should also thoroughly discuss what the intended solution will be for this problem. Questions to ask yourself when writing this section:
- _Is the problem statement clearly defined? Will the reader understand what you are expecting to solve?_
- _Have you thoroughly discussed how you will attempt to solve the problem?_
- _Is an anticipated solution clearly defined? Will the reader understand what results you are looking for?_

Human Activity Recognition or HAR for short, is the problem of predicting what a person is doing based on a time series recording of their movement using sensors. The idea is that once the subject’s activity is recognized and known, it can be saved in a model that can then be used to recognize future those activities with high accuracy when applied, and then provide useful assistance with the result. 

Movements that we will study in this project will be normal indoor activities such as walking, standing, sitting, laying and walking up and downstairs. Additionaly, six postural transitional activities will be included these are moving from standing to sit, or laying down to sit, or laying to standing.  Sensors are body mounted and record accelerometer and gyroscope data in three dimensions (x, y, z).  The objective is to classify activities into one of the twelve activities performed. 

It is a challenging problem because there is no clear analytical way to relate the sensor data to specific actions in a general way. It is technically challenging because of the large volume of sensor data collected (e.g. tens or hundreds of observations per second) and the classical use of hand crafted features and heuristics from this data in developing predictive models.
Classical approaches to the problem involve hand crafting features from the time series data based on fixed-size windows and training machine learning models, such as ensembles of decision trees or random forest methods.

### 1.3 Metrics
In this section, you will need to clearly define the metrics or calculations you will use to measure performance of a model or result in your project. These calculations and metrics should be justified based on the characteristics of the problem and problem domain. Questions to ask yourself when writing this section:
- _Are the metrics you’ve chosen to measure the performance of your models clearly discussed and defined?_
- _Have you provided reasonable justification for the metrics chosen based on the problem and solution?_

Most of the time we use classification accuracy to measure the performance of our model, however it is not enough to truly judge our model. There are five other metrics that will be used to meaure performance of the models, the confusion matrix, the Precision-Recall Curve, F1 Score and Log Loss. 
(I removed the the Receiver Operating Characteristic,)
The classification Accuracy is what we usually mean, when we use the term accuracy. It is the ratio of number of correct predictions to the total number of activity input samples.

The confusion matrix, which is an output table that summarizes the number of true positives, true negatives, false positives, and false negatives, and describes the complete performance of the model. The confusion matrix is a straightforward evaluation metric if we assume that the results have balanced classes, in the number of true positives to true negatives.

The Precision-Recall evalution metrics can be used if we find that the dataset is imbalanced. 
Precision is the number of true positives over the number of total positive predictions.  A high precision means that there will be many true positives and a low false positive rate.  Recall is the number of true positives over the number of total actual positives in the dataset.  A high recall means that the model has captured most of the true positives, and has a low false negative rate.  An optimal solution needs to have high precision and high recall, rejecting only those activities that don't match the class, i.e. high precision, and matching the correct activity to the class in the dataset (high recall). 

F1 Score is the Harmonic Mean between precision and recall. The range for F1 Score is [0, 1]. It tells you how precise the classifier is (how many instances it classifies correctly), as well as how robust it is (it does not miss a significant number of instances).

The Logarithmic Loss metric can handle class imbalances and works well with for multi-class classification with artificial neural network implementations. The log loss works by penalising the false classifications, in other words minimizing Log Loss will indicate greater accuracy for the classifier.

Receiver Operating Characteristic is one of the most widely used metrics for evaluation, and used for binary classification problems. that plots the true positive rate on the Y axis and the false positive rate on the X axis.  The closer the curve is to the top-left corner of the plot, the better the solutionh,signifying a 0% false positive rate and a 100% true positive rate.  To evaluate the solution, we compute the area under the curve, so that the larger the auROC, the better the model. 

## 2 Analysis
_(approx. 2-4 pages)_

### 2.1 Data Exploration
In this section, you will be expected to analyze the data you are using for the problem. This data can either be in the form of a dataset (or datasets), input data (or input files), or even an environment. The type of data should be thoroughly described and, if possible, have basic statistics and information presented (such as discussion of input features or defining characteristics about the input or environment). Any abnormalities or interesting qualities about the data that may need to be addressed have been identified (such as features that need to be transformed or the possibility of outliers). Questions to ask yourself when writing this section:
- _If a dataset is present for this problem, have you thoroughly discussed certain features about the dataset? Has a data sample been provided to the reader?_
- _If a dataset is present for this problem, are statistics about the dataset calculated and reported? Have any relevant results from this calculation been discussed?_
- _If a dataset is **not** present for this problem, has discussion been made about the input space or input data for your problem?_
- _Are there any abnormalities or characteristics about the input space or dataset that need to be addressed? (categorical variables, missing values, outliers, etc.)_

A standard human activity recognition dataset is the ‘Activity Recognition Using Smart Phones Dataset’ made available in 2012. It was prepared and made available by Davide Anguita, et al. from the University of Genova, Italy and is described in full in their 2013 paper “A Public Domain Dataset for Human Activity Recognition Using Smartphones.” 
The dataset was made available and can be downloaded for free from the UCI Machine Learning Repository:
Human Activity Recognition Using Smartphones Data Set, UCI Machine Learning Repository

The data was collected from 30 subjects aged between 19 and 48 years old performing one of six standard activities while wearing a waist-mounted smartphone that recorded the movement data. Video was recorded of each subject performing the activities and the movement data was labeled manually from these videos. The HAR dataset was collected in laboratory conditions but volunteers were asked to perform freely the sequence of activities for a more naturalistic dataset.  The set of experiments in which each person were instructed to follow was a protocol of activities while wearing a waist-mounted Samsung Galaxy S II smartphone. The six selected activities were as follows: 1. Walking, 2. Walking Upstairs, 3. Walking Downstairs, 4. Sitting. 5. Standing, 6. Laying

The movement data recorded was the x, y, and z accelerometer data (linear acceleration) and gyroscopic data (angular velocity) from the smart phone, specifically a Samsung Galaxy S II. Observations were recorded at 50 Hz (i.e. 50 data points per second), that was consistent across the dataset. Each subject performed the sequence of activities twice; once with the device on their left-hand-side and once with the device on their right-hand side.
The raw data is not available. Instead, a pre-processed version of the dataset was made available. The pre-processing steps included:
Pre-processing accelerometer and gyroscope using noise filters.
Splitting data into fixed windows of 2.56 seconds (128 data points) with 50% overlap. Splitting of accelerometer data into gravitational (total) and body motion components.
Feature engineering was applied to the window data, and a copy of the data with these engineered features was made available.
A number of time and frequency features commonly used in the field of human activity recognition were extracted from each window. The result was a 561 element vector of features.
The dataset was split into train (70%) and test (30%) sets based on data for subjects, e.g. 21 subjects for train and nine for test.
The data is provided as a single zip file that is about 58 megabytes in size. 

### 2.1 Exploratory Visualization
In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:
- _Have you visualized a relevant characteristic or feature about the dataset or input data?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

Visualization of Feature Distribution of the activities by type.
Visual exploration of the data in 3 dimensions with principal component analysis will be done.
Plot Traces for a Single Subject, Plot Total Activity Durations, Plot Traces for Each Subject

### 2.2 Algorithms and Techniques
In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_

This stage of the activity recognition chain involves the training of a classification model that is able to discriminate the physical activity performed by a certain subject. Our objective is to eventually obtain a classifier that maximizes accuracy (i.e., the hit rate between the predicted and the actual class in a test set).
The resulting classifier is intended to be used in a real-time cross-person prediction system that must be able to accurately predict the activity that a new user is performing. 

First, to prototype a fast evaluation system,  Python’s machine learning library, using "scikit-learn" a baseline in performance will be done using a robust method such as Logistic Regression. Next, XGBoost and LightGBM models are good supervised learning approaches that will be explored next. 
Instead of picking just one of the machine learning solutions we have developed, we will build an ensemble of the models and evaluate if that leads to an improved classification rate.  An ensemble of the Logistic Regression Model, XGBoost, and LightGBM will be built.  An ensemble will be good to use because each of the standalone solutions have different strengths and weaknesses, and the compensation for the strengths and weaknesses between the models will generally lead to a better result of them all together in an ensemble. 
Finally an artificial neural network approach specifically suited to time series classification problem will be approached. 

The same data are input to different classifiers, and the results are stored for analysis. When selecting the classifiers, I used a wide set that includes classical [49] and state-of-the-art [50] techniques. I tested all of the classifiers with various parameters. The classifiers with lower performances were discarded. The following classifiers were finally selected to be studied in more detail.

Suggested:
•	The XGBoost and LightGBM models could be good supervised learning approaches to try here.
•	Since you're creating multiple supervised learning models, you could try combining them all together into a custom ensemble model:
http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/
https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python

Ensembles:  Instead of picking just one of the machine learning solutions we have developed for use in production, we can evaluate whether an ensemble of the models leans to an improved classification rate.   This is because each of the standalone solutions have different strengths and weaknesses, and the compensation for the strengths and weaknesses between the models will generally lead to a better result of them all together in an ensemble. 

Stacking:  We use stacking to determine whether or not we can get an improvement in performance compared to the standalone models.  In stacking, we take the predictions from the k-fold cross-validation from each of the standalone models (layer one predictions) and append them to the original training dataset.  We then train on this original features plus layer one predictions dataset using k-fold cross-validation, resulting in layer two predictions. And from this it will be evaluated to see if we have an improvement over any of the standalone models. 

### 2.4 Benchmark
In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_

The benchmark model that will be used will be a logistic regression model for a classification to develop a baseline activity recognition model.
The benchmark model will then be compared to the different standalone models, and then to the ensemble, then the stacking, and then the ANN models predictions.  

## 3 Methodology
_(approx. 3-5 pages)_

### 3.1 Data Preprocessing
In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_
There is no need for pre-processing of the samples, as the sampling rate of 50Hz that is consistent across the dataset.


### 3.2 Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_


For each individual run of all of the experiments conducted, I collected accuracy, precision, recall and F1 scores. The accuracy metric is simply the percentage of correctly-classified instances. Although the dataset used for this work is not perfectly balanced, because we are working with 12 classes, we consider that average accuracy metric to be both a simple and appropriate metric for representing the classifiers’ overall performances. For this reason, and to aid readability, we summarized only the accuracy metric plus variance for the preliminary evaluation. Of course, we also analyzed the confusion matrix, precision, recall, F1. Because this is a multiclass problem, these metrics are obtained by weighted averaging.

Another commonly-used quality metric is the receiver operating characteristic (ROC) metric; however, for this work, we did not analyze the ROC curves, because this is a multiclass problem. First, ROC curves, which generally work well for visualization, would be difficult to visualize for 12 classes, requiring the volume under the ROC surfaces metric (VUS) or Cobweb diagrams. Second, for each class, a pairwise comparison of the performance of that class versus all of the other classes would be required.
ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5298639/

#### 3.2.1 Distance-Based Methods
These algorithms assume that the data have some type of similarity and relations based on geometric properties and can be grouped according to these patterns.
The k-nearest neighbors algorithm (k-NN) is one of the simplest and most effective nonparametric machine learning algorithms for    classification and regression [51]. The k-NN classifier obtained better results for online recognition when compared to C4.5 decision tree. [52]. After testing several parameters, we eventually used the k-NN algorithm with its default scikit-learn parameters

#### 3.2.2  Statistical Methods
These techniques assume that data follow a probabilistic function that needs to be inferred.

#### 3.2.3 Kernel Methods
These models perform pattern analysis based on a kernel function, which is a similarity function over pairs of data points.

#### 3.2.4 Decision Tree-Based Methods
These techniques build a model that predicts a target variable by learning interpretable decision rules inferred from the training data [61]. Using a decision tree as a predictive model is widely used as a decision support tool because they are easily interpreted (i.e., they provide a sequence of decisions to obtain the final classification result).

#### 3.2.5 Ensemble Learners
These models combine the predictions of several base estimators built with a given learning algorithm to improve the results and robustness over a single estimator.

#### 3.2.6 Deep Learning
These techniques are an extension or a re-branding of neural networks [69]. They are based on the idea of modeling problems with high-level abstractions in data by using multiple processing layers that basically perform multiple non-linear transformations


### 3.3 Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## 4 Results
_(approx. 2-3 pages)_

### 4.1 Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_


### 4.2 Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## 5 Conclusion
_(approx. 1-2 pages)_

### 5.1 Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### 5.2 Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### 5.3 Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?




# References

Patel, A., "Hands-On Unsupervised Learning Using Python" How to Build Applied Machine Learning Solutions from Unlabeled Data, O'Reilly Media Inc., 3rd Release, May 03, 2019.



