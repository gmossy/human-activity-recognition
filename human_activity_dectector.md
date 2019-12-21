### Human Activity Recognition 
#### Udacity Machine Learning Nanodegree, Capstone Project
#### Glenn Mossy 11/10/2019

The data we will be using is the Human Activity Recognition dataset from the UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones


```python
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns
import os
import time
print(os.listdir('./data/UCI HAR Dataset'))
```

    ['activity_labels.txt', 'features.txt', 'features_info.txt', 'human_activity_predictions.csv', 'README.txt', 'test', 'train']
    

### Collect Dataset and Explore


```python
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, cross_val_score

import warnings
warnings.filterwarnings('ignore')
```


```python
# Importing the dataset
# 'train/X_train.txt': Training set
X_train = pd.read_csv('./data/UCI HAR Dataset/train/X_train.txt', sep='\s+', header=None)
#'test/X_test.txt': Test set.
X_test = pd.read_csv('./data/UCI HAR Dataset/test/X_test.txt', sep='\s+', header=None)
#'train/y_train.txt': Training labels
y_train = pd.read_csv('./data/UCI HAR Dataset/train/y_train.txt', sep='\s+', header=None, names='Y')
#'test/X_test.txt': Test set.
y_test = pd.read_csv('./data/UCI HAR Dataset/test/y_test.txt', sep='\s+', header=None, names='Y')

train_dataset = pd.read_csv('./data/kaggle-dataset/train.csv', sep=',')
test_dataset = pd.read_csv('./data/kaggle-dataset/test.csv', sep=',')

# combine the train/test data
X = pd.concat([X_train, X_test])
y = pd.concat([y_train, y_test])

# print(X.shape)
# print(y.shape)

# 'features.txt': List of all features.
features = pd.read_csv('./data/UCI HAR Dataset/features.txt', sep=' ', header=None, names=('ID','Sensor'))

#'features_info.txt': Shows information about the variables used on the feature vector.
features_info = pd.read_csv('./data/UCI HAR Dataset/features_info.txt', sep=' ', header=None, names=('ID','Sensor'))

# map the header row to identify the column parameter name
X_train.columns = features['Sensor']
X_test.columns = features['Sensor']

# 'activity_labels.txt': Links the class labels with their activity name.
activity_labels = pd.read_csv('./data/UCI HAR Dataset/activity_labels.txt', sep=' ', header=None, names=('ID','Activity'))

# The following files are available for the train and test data. Their descriptions are equivalent.
# - 'train/subject_train.txt': Each row identifies the subject who performed the activity for each window sample. Its range is from 1 to 30. 
# - 'train/Inertial Signals/total_acc_x_train.txt': The acceleration signal from the smartphone accelerometer X axis in standard gravity units 'g'. Every row shows a 128 element vector. The same description applies for the 'total_acc_x_train.txt' and 'total_acc_z_train.txt' files for the Y and Z axis. 
# - 'train/Inertial Signals/body_acc_x_train.txt': The body acceleration signal obtained by subtracting the gravity from the total acceleration. 
# - 'train/Inertial Signals/body_gyro_x_train.txt': The angular velocity vector measured by the gyroscope for each window sample. The units are radians/second. 

subject_train = pd.read_csv('./data/UCI HAR Dataset/train/subject_train.txt', sep='\s+', header=None)
# Inertial Signals
total_acc_x_train = pd.read_csv('./data/UCI HAR Dataset/train/Inertial Signals/total_acc_x_train.txt', sep='\s+', header=None)
body_acc_x_train = pd.read_csv('./data/UCI HAR Dataset/train/Inertial Signals/body_acc_x_train.txt', sep='\s+', header=None)
body_gyro_x_train = pd.read_csv('./data/UCI HAR Dataset/train/Inertial Signals/body_gyro_x_train.txt', sep='\s+', header=None)

INPUT_SIGNAL_TYPES = [
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_"
]
#y_train.describe()
#body_acc_x_train.describe()
```


```python
df = train_dataset
test = test_dataset
```


```python
df.T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>7342</th>
      <th>7343</th>
      <th>7344</th>
      <th>7345</th>
      <th>7346</th>
      <th>7347</th>
      <th>7348</th>
      <th>7349</th>
      <th>7350</th>
      <th>7351</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>tBodyAcc-mean()-X</td>
      <td>0.288585</td>
      <td>0.278419</td>
      <td>0.279653</td>
      <td>0.279174</td>
      <td>0.276629</td>
      <td>0.277199</td>
      <td>0.279454</td>
      <td>0.277432</td>
      <td>0.277293</td>
      <td>0.280586</td>
      <td>...</td>
      <td>0.276137</td>
      <td>0.29423</td>
      <td>0.221206</td>
      <td>0.207861</td>
      <td>0.237966</td>
      <td>0.299665</td>
      <td>0.273853</td>
      <td>0.273387</td>
      <td>0.289654</td>
      <td>0.351503</td>
    </tr>
    <tr>
      <td>tBodyAcc-mean()-Y</td>
      <td>-0.0202942</td>
      <td>-0.0164106</td>
      <td>-0.0194672</td>
      <td>-0.0262006</td>
      <td>-0.0165697</td>
      <td>-0.0100979</td>
      <td>-0.0196408</td>
      <td>-0.0304883</td>
      <td>-0.0217507</td>
      <td>-0.0099603</td>
      <td>...</td>
      <td>-0.108046</td>
      <td>-0.0399683</td>
      <td>-0.0363901</td>
      <td>0.0634229</td>
      <td>-0.00108781</td>
      <td>-0.0571934</td>
      <td>-0.00774933</td>
      <td>-0.0170106</td>
      <td>-0.018843</td>
      <td>-0.0124231</td>
    </tr>
    <tr>
      <td>tBodyAcc-mean()-Z</td>
      <td>-0.132905</td>
      <td>-0.12352</td>
      <td>-0.113462</td>
      <td>-0.123283</td>
      <td>-0.115362</td>
      <td>-0.105137</td>
      <td>-0.110022</td>
      <td>-0.12536</td>
      <td>-0.120751</td>
      <td>-0.106065</td>
      <td>...</td>
      <td>-0.056677</td>
      <td>-0.143397</td>
      <td>-0.167651</td>
      <td>-0.220567</td>
      <td>-0.148326</td>
      <td>-0.181233</td>
      <td>-0.147468</td>
      <td>-0.0450218</td>
      <td>-0.158281</td>
      <td>-0.203867</td>
    </tr>
    <tr>
      <td>tBodyAcc-std()-X</td>
      <td>-0.995279</td>
      <td>-0.998245</td>
      <td>-0.99538</td>
      <td>-0.996091</td>
      <td>-0.998139</td>
      <td>-0.997335</td>
      <td>-0.996921</td>
      <td>-0.996559</td>
      <td>-0.997328</td>
      <td>-0.994803</td>
      <td>...</td>
      <td>-0.230796</td>
      <td>-0.230396</td>
      <td>-0.176954</td>
      <td>-0.244758</td>
      <td>-0.218949</td>
      <td>-0.195387</td>
      <td>-0.235309</td>
      <td>-0.218218</td>
      <td>-0.219139</td>
      <td>-0.26927</td>
    </tr>
    <tr>
      <td>tBodyAcc-std()-Y</td>
      <td>-0.983111</td>
      <td>-0.9753</td>
      <td>-0.967187</td>
      <td>-0.983403</td>
      <td>-0.980817</td>
      <td>-0.990487</td>
      <td>-0.967186</td>
      <td>-0.966728</td>
      <td>-0.961245</td>
      <td>-0.972758</td>
      <td>...</td>
      <td>-0.140521</td>
      <td>-0.133669</td>
      <td>-0.0501467</td>
      <td>-0.0321591</td>
      <td>-0.0129267</td>
      <td>0.0399048</td>
      <td>0.00481628</td>
      <td>-0.103822</td>
      <td>-0.111412</td>
      <td>-0.0872115</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>angle(X,gravityMean)</td>
      <td>-0.841247</td>
      <td>-0.844788</td>
      <td>-0.848933</td>
      <td>-0.848649</td>
      <td>-0.847865</td>
      <td>-0.849632</td>
      <td>-0.85215</td>
      <td>-0.851017</td>
      <td>-0.847971</td>
      <td>-0.848294</td>
      <td>...</td>
      <td>-0.830575</td>
      <td>-0.799426</td>
      <td>-0.787935</td>
      <td>-0.780362</td>
      <td>-0.797272</td>
      <td>-0.791883</td>
      <td>-0.77184</td>
      <td>-0.779133</td>
      <td>-0.785181</td>
      <td>-0.783267</td>
    </tr>
    <tr>
      <td>angle(Y,gravityMean)</td>
      <td>0.179941</td>
      <td>0.180289</td>
      <td>0.180637</td>
      <td>0.181935</td>
      <td>0.185151</td>
      <td>0.184823</td>
      <td>0.18217</td>
      <td>0.183779</td>
      <td>0.188982</td>
      <td>0.19031</td>
      <td>...</td>
      <td>0.213174</td>
      <td>0.23549</td>
      <td>0.24449</td>
      <td>0.249624</td>
      <td>0.234996</td>
      <td>0.238604</td>
      <td>0.252676</td>
      <td>0.249145</td>
      <td>0.246432</td>
      <td>0.246809</td>
    </tr>
    <tr>
      <td>angle(Z,gravityMean)</td>
      <td>-0.0586269</td>
      <td>-0.0543167</td>
      <td>-0.0491178</td>
      <td>-0.0476632</td>
      <td>-0.0438923</td>
      <td>-0.0421264</td>
      <td>-0.04301</td>
      <td>-0.0419758</td>
      <td>-0.0373639</td>
      <td>-0.0344173</td>
      <td>...</td>
      <td>-0.00510524</td>
      <td>-0.00164732</td>
      <td>0.00953791</td>
      <td>0.0278779</td>
      <td>0.048907</td>
      <td>0.0498191</td>
      <td>0.0500526</td>
      <td>0.0408112</td>
      <td>0.0253395</td>
      <td>0.0366948</td>
    </tr>
    <tr>
      <td>subject</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>30</td>
      <td>30</td>
      <td>30</td>
      <td>30</td>
      <td>30</td>
      <td>30</td>
      <td>30</td>
      <td>30</td>
      <td>30</td>
      <td>30</td>
    </tr>
    <tr>
      <td>Activity</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>STANDING</td>
      <td>...</td>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING_UPSTAIRS</td>
      <td>WALKING_UPSTAIRS</td>
    </tr>
  </tbody>
</table>
<p>563 rows × 7352 columns</p>
</div>




```python

```


```python
print ('The training set has {} examples'.format(len(X_train)))
print ('The testing set has  {} examples'.format(len(X_test)))
print ('')
print ('The percentage of test to total examples is {:.2f}%'.format(len(X_test) / float(len(X_test) + len(X_train)) * 100))
```

    The training set has 7352 examples
    The testing set has  2947 examples
    
    The percentage of test to total examples is 28.61%
    


```python
df = X_train
test = X_test
```


```python
df.T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>7342</th>
      <th>7343</th>
      <th>7344</th>
      <th>7345</th>
      <th>7346</th>
      <th>7347</th>
      <th>7348</th>
      <th>7349</th>
      <th>7350</th>
      <th>7351</th>
    </tr>
    <tr>
      <th>Sensor</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>tBodyAcc-mean()-X</td>
      <td>0.288585</td>
      <td>0.278419</td>
      <td>0.279653</td>
      <td>0.279174</td>
      <td>0.276629</td>
      <td>0.277199</td>
      <td>0.279454</td>
      <td>0.277432</td>
      <td>0.277293</td>
      <td>0.280586</td>
      <td>...</td>
      <td>0.276137</td>
      <td>0.294230</td>
      <td>0.221206</td>
      <td>0.207861</td>
      <td>0.237966</td>
      <td>0.299665</td>
      <td>0.273853</td>
      <td>0.273387</td>
      <td>0.289654</td>
      <td>0.351503</td>
    </tr>
    <tr>
      <td>tBodyAcc-mean()-Y</td>
      <td>-0.020294</td>
      <td>-0.016411</td>
      <td>-0.019467</td>
      <td>-0.026201</td>
      <td>-0.016570</td>
      <td>-0.010098</td>
      <td>-0.019641</td>
      <td>-0.030488</td>
      <td>-0.021751</td>
      <td>-0.009960</td>
      <td>...</td>
      <td>-0.108046</td>
      <td>-0.039968</td>
      <td>-0.036390</td>
      <td>0.063423</td>
      <td>-0.001088</td>
      <td>-0.057193</td>
      <td>-0.007749</td>
      <td>-0.017011</td>
      <td>-0.018843</td>
      <td>-0.012423</td>
    </tr>
    <tr>
      <td>tBodyAcc-mean()-Z</td>
      <td>-0.132905</td>
      <td>-0.123520</td>
      <td>-0.113462</td>
      <td>-0.123283</td>
      <td>-0.115362</td>
      <td>-0.105137</td>
      <td>-0.110022</td>
      <td>-0.125360</td>
      <td>-0.120751</td>
      <td>-0.106065</td>
      <td>...</td>
      <td>-0.056677</td>
      <td>-0.143397</td>
      <td>-0.167651</td>
      <td>-0.220567</td>
      <td>-0.148326</td>
      <td>-0.181233</td>
      <td>-0.147468</td>
      <td>-0.045022</td>
      <td>-0.158281</td>
      <td>-0.203867</td>
    </tr>
    <tr>
      <td>tBodyAcc-std()-X</td>
      <td>-0.995279</td>
      <td>-0.998245</td>
      <td>-0.995380</td>
      <td>-0.996091</td>
      <td>-0.998139</td>
      <td>-0.997335</td>
      <td>-0.996921</td>
      <td>-0.996559</td>
      <td>-0.997328</td>
      <td>-0.994803</td>
      <td>...</td>
      <td>-0.230796</td>
      <td>-0.230396</td>
      <td>-0.176954</td>
      <td>-0.244758</td>
      <td>-0.218949</td>
      <td>-0.195387</td>
      <td>-0.235309</td>
      <td>-0.218218</td>
      <td>-0.219139</td>
      <td>-0.269270</td>
    </tr>
    <tr>
      <td>tBodyAcc-std()-Y</td>
      <td>-0.983111</td>
      <td>-0.975300</td>
      <td>-0.967187</td>
      <td>-0.983403</td>
      <td>-0.980817</td>
      <td>-0.990487</td>
      <td>-0.967186</td>
      <td>-0.966728</td>
      <td>-0.961245</td>
      <td>-0.972758</td>
      <td>...</td>
      <td>-0.140521</td>
      <td>-0.133669</td>
      <td>-0.050147</td>
      <td>-0.032159</td>
      <td>-0.012927</td>
      <td>0.039905</td>
      <td>0.004816</td>
      <td>-0.103822</td>
      <td>-0.111412</td>
      <td>-0.087212</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>angle(tBodyGyroMean,gravityMean)</td>
      <td>-0.464761</td>
      <td>-0.732626</td>
      <td>0.100699</td>
      <td>0.640011</td>
      <td>0.693578</td>
      <td>0.275041</td>
      <td>0.014637</td>
      <td>-0.561871</td>
      <td>-0.234313</td>
      <td>-0.482871</td>
      <td>...</td>
      <td>0.918171</td>
      <td>0.885558</td>
      <td>-0.546757</td>
      <td>-0.864127</td>
      <td>-0.774783</td>
      <td>0.206972</td>
      <td>-0.879033</td>
      <td>0.864404</td>
      <td>0.936674</td>
      <td>-0.056088</td>
    </tr>
    <tr>
      <td>angle(tBodyGyroJerkMean,gravityMean)</td>
      <td>-0.018446</td>
      <td>0.703511</td>
      <td>0.808529</td>
      <td>-0.485366</td>
      <td>-0.615971</td>
      <td>-0.368224</td>
      <td>-0.189512</td>
      <td>0.467383</td>
      <td>0.117797</td>
      <td>-0.070670</td>
      <td>...</td>
      <td>-0.609025</td>
      <td>-0.879032</td>
      <td>-0.950493</td>
      <td>0.591409</td>
      <td>0.730142</td>
      <td>-0.425619</td>
      <td>0.400219</td>
      <td>0.701169</td>
      <td>-0.589479</td>
      <td>-0.616956</td>
    </tr>
    <tr>
      <td>angle(X,gravityMean)</td>
      <td>-0.841247</td>
      <td>-0.844788</td>
      <td>-0.848933</td>
      <td>-0.848649</td>
      <td>-0.847865</td>
      <td>-0.849632</td>
      <td>-0.852150</td>
      <td>-0.851017</td>
      <td>-0.847971</td>
      <td>-0.848294</td>
      <td>...</td>
      <td>-0.830575</td>
      <td>-0.799426</td>
      <td>-0.787935</td>
      <td>-0.780362</td>
      <td>-0.797272</td>
      <td>-0.791883</td>
      <td>-0.771840</td>
      <td>-0.779133</td>
      <td>-0.785181</td>
      <td>-0.783267</td>
    </tr>
    <tr>
      <td>angle(Y,gravityMean)</td>
      <td>0.179941</td>
      <td>0.180289</td>
      <td>0.180637</td>
      <td>0.181935</td>
      <td>0.185151</td>
      <td>0.184823</td>
      <td>0.182170</td>
      <td>0.183779</td>
      <td>0.188982</td>
      <td>0.190310</td>
      <td>...</td>
      <td>0.213174</td>
      <td>0.235490</td>
      <td>0.244490</td>
      <td>0.249624</td>
      <td>0.234996</td>
      <td>0.238604</td>
      <td>0.252676</td>
      <td>0.249145</td>
      <td>0.246432</td>
      <td>0.246809</td>
    </tr>
    <tr>
      <td>angle(Z,gravityMean)</td>
      <td>-0.058627</td>
      <td>-0.054317</td>
      <td>-0.049118</td>
      <td>-0.047663</td>
      <td>-0.043892</td>
      <td>-0.042126</td>
      <td>-0.043010</td>
      <td>-0.041976</td>
      <td>-0.037364</td>
      <td>-0.034417</td>
      <td>...</td>
      <td>-0.005105</td>
      <td>-0.001647</td>
      <td>0.009538</td>
      <td>0.027878</td>
      <td>0.048907</td>
      <td>0.049819</td>
      <td>0.050053</td>
      <td>0.040811</td>
      <td>0.025339</td>
      <td>0.036695</td>
    </tr>
  </tbody>
</table>
<p>561 rows × 7352 columns</p>
</div>




```python
print(y_train.shape)
print(y_test.shape)
```

    (7352, 1)
    (2947, 1)
    


```python
print(activity_labels.Activity)
```

    0               WALKING
    1      WALKING_UPSTAIRS
    2    WALKING_DOWNSTAIRS
    3               SITTING
    4              STANDING
    5                LAYING
    Name: Activity, dtype: object
    


```python
#y_train.describe()
# print(features_info)
```


```python
# https://github.com/deadskull7/Human-Activity-Recognition-with-Neural-Network-using-Gyroscopic-and-Accelerometer-variables/blob/master/Human%20Activity%20Recognition%20(97.98%20%25).ipynb

# sns.set(rc={'figure.figsize':(15,5)})
# fig1 = sns.stripplot(x='Activity', y= df.loc[df['subject']==15].iloc[:,7], data= df.loc[df['subject']==15], jitter=True)
# plt.title("Feature Distribution")
# plt.grid(True)
# plt.show(fig1)
```


```python
activity_labels.head(6)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Activity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>WALKING</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>WALKING_UPSTAIRS</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>WALKING_DOWNSTAIRS</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>SITTING</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>STANDING</td>
    </tr>
    <tr>
      <td>5</td>
      <td>6</td>
      <td>LAYING</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = train_dataset
test = test_dataset
df.T
# print(df.Activity.unique())
print("----------------------------------------")
print(df.Activity.value_counts())

sns.set(rc={'figure.figsize':(14,6)})
sns.set_style('white')
fig = sns.countplot(x = "Activity" , data = df)
plt.xlabel("Activity", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.title("Activity Count", fontsize=16)
plt.grid(True)
plt.show(fig)
```

    ----------------------------------------
    LAYING                1407
    STANDING              1374
    SITTING               1286
    WALKING               1226
    WALKING_UPSTAIRS      1073
    WALKING_DOWNSTAIRS     986
    Name: Activity, dtype: int64
    


![png](human_activity_dectector_files/human_activity_dectector_17_1.png)



```python
# pd.crosstab(df.subject, df.Activity, margins=True).style.background_gradient(cmap='autumn_r')
```


```python
variables = ['fBodyAcc-Mean-1', 'fBodyAcc-Mean-2', 'fBodyAcc-Mean-3']

```


```python

```


```python
# https://github.com/deadskull7/Human-Activity-Recognition-with-Neural-Network-using-Gyroscopic-and-Accelerometer-variables/blob/master/Human%20Activity%20Recognition%20(97.98%20%25).ipynb
sns.set(rc={'figure.figsize':(20,7)})
colours = ["maroon","coral","darkorchid","goldenrod","purple","darkgreen","darkviolet","saddlebrown","aqua","olive"]
index = -1
for i in df.columns[0:10]:
    index = index + 1
    fig = sns.kdeplot(df[i] , shade=True, color=colours[index])
plt.xlabel("Features")
plt.ylabel("Value")
plt.title("Visualization of Feature Distribution")
plt.grid(True)
plt.show(fig)
```


![png](human_activity_dectector_files/human_activity_dectector_21_0.png)



```python
X_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Sensor</th>
      <th>tBodyAcc-mean()-X</th>
      <th>tBodyAcc-mean()-Y</th>
      <th>tBodyAcc-mean()-Z</th>
      <th>tBodyAcc-std()-X</th>
      <th>tBodyAcc-std()-Y</th>
      <th>tBodyAcc-std()-Z</th>
      <th>tBodyAcc-mad()-X</th>
      <th>tBodyAcc-mad()-Y</th>
      <th>tBodyAcc-mad()-Z</th>
      <th>tBodyAcc-max()-X</th>
      <th>...</th>
      <th>fBodyBodyGyroJerkMag-meanFreq()</th>
      <th>fBodyBodyGyroJerkMag-skewness()</th>
      <th>fBodyBodyGyroJerkMag-kurtosis()</th>
      <th>angle(tBodyAccMean,gravity)</th>
      <th>angle(tBodyAccJerkMean),gravityMean)</th>
      <th>angle(tBodyGyroMean,gravityMean)</th>
      <th>angle(tBodyGyroJerkMean,gravityMean)</th>
      <th>angle(X,gravityMean)</th>
      <th>angle(Y,gravityMean)</th>
      <th>angle(Z,gravityMean)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.288585</td>
      <td>-0.020294</td>
      <td>-0.132905</td>
      <td>-0.995279</td>
      <td>-0.983111</td>
      <td>-0.913526</td>
      <td>-0.995112</td>
      <td>-0.983185</td>
      <td>-0.923527</td>
      <td>-0.934724</td>
      <td>...</td>
      <td>-0.074323</td>
      <td>-0.298676</td>
      <td>-0.710304</td>
      <td>-0.112754</td>
      <td>0.030400</td>
      <td>-0.464761</td>
      <td>-0.018446</td>
      <td>-0.841247</td>
      <td>0.179941</td>
      <td>-0.058627</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.278419</td>
      <td>-0.016411</td>
      <td>-0.123520</td>
      <td>-0.998245</td>
      <td>-0.975300</td>
      <td>-0.960322</td>
      <td>-0.998807</td>
      <td>-0.974914</td>
      <td>-0.957686</td>
      <td>-0.943068</td>
      <td>...</td>
      <td>0.158075</td>
      <td>-0.595051</td>
      <td>-0.861499</td>
      <td>0.053477</td>
      <td>-0.007435</td>
      <td>-0.732626</td>
      <td>0.703511</td>
      <td>-0.844788</td>
      <td>0.180289</td>
      <td>-0.054317</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.279653</td>
      <td>-0.019467</td>
      <td>-0.113462</td>
      <td>-0.995380</td>
      <td>-0.967187</td>
      <td>-0.978944</td>
      <td>-0.996520</td>
      <td>-0.963668</td>
      <td>-0.977469</td>
      <td>-0.938692</td>
      <td>...</td>
      <td>0.414503</td>
      <td>-0.390748</td>
      <td>-0.760104</td>
      <td>-0.118559</td>
      <td>0.177899</td>
      <td>0.100699</td>
      <td>0.808529</td>
      <td>-0.848933</td>
      <td>0.180637</td>
      <td>-0.049118</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.279174</td>
      <td>-0.026201</td>
      <td>-0.123283</td>
      <td>-0.996091</td>
      <td>-0.983403</td>
      <td>-0.990675</td>
      <td>-0.997099</td>
      <td>-0.982750</td>
      <td>-0.989302</td>
      <td>-0.938692</td>
      <td>...</td>
      <td>0.404573</td>
      <td>-0.117290</td>
      <td>-0.482845</td>
      <td>-0.036788</td>
      <td>-0.012892</td>
      <td>0.640011</td>
      <td>-0.485366</td>
      <td>-0.848649</td>
      <td>0.181935</td>
      <td>-0.047663</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.276629</td>
      <td>-0.016570</td>
      <td>-0.115362</td>
      <td>-0.998139</td>
      <td>-0.980817</td>
      <td>-0.990482</td>
      <td>-0.998321</td>
      <td>-0.979672</td>
      <td>-0.990441</td>
      <td>-0.942469</td>
      <td>...</td>
      <td>0.087753</td>
      <td>-0.351471</td>
      <td>-0.699205</td>
      <td>0.123320</td>
      <td>0.122542</td>
      <td>0.693578</td>
      <td>-0.615971</td>
      <td>-0.847865</td>
      <td>0.185151</td>
      <td>-0.043892</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 561 columns</p>
</div>




```python
y_train['Y'].value_counts()
```




    6    1407
    5    1374
    4    1286
    1    1226
    2    1073
    3     986
    Name: Y, dtype: int64




```python
# look at the data through a correlation matrix
#correlations = X_train.corr()
# plot on a heatmap
#plt.figure(figsize=(25, 25))
#sns.heatmap(correlations, cmap='coolwarm')
#plt.show()
```


```python
# A visual exploration of 561 variables would be unreasonable. Let's try to visualise the data in 3 dimensions with principal
# components instead.

pca_3 = PCA(n_components=3)
X_pca_3 = pca_3.fit_transform(X_train)
y = y_train['Y']

labels = [('Walking', 1), ('Walking Upstairs', 2), ('Walking Downstairs', 3), 
          ('Sitting', 4), ('Standing', 5), ('Laying', 6)]

fig = plt.figure(figsize=(10, 8))
ax = Axes3D(fig, elev=-150, azim=130)
sc= ax.scatter(X_pca_3[:, 0], X_pca_3[:, 1], X_pca_3[:, 2], c=y, cmap='Set1', edgecolor='k', s=40)

ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

# create the events marking the x data points
colors = [sc.cmap(sc.norm(i)) for i in [1,2,3,4,5,6]]
custom_lines = [plt.Line2D([],[], ls="", marker='.', 
                mec='k', mfc=c, mew=.1, ms=20) for c in colors]
ax.legend(custom_lines, [l[0] for l in labels], 
          loc='center left', bbox_to_anchor=(1.0, 0.5))
ax.set_title('Visualise the data in 3 dimensions with PCA')

plt.show()
```


![png](human_activity_dectector_files/human_activity_dectector_25_0.png)



```python
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
    
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)
start = time.time()
classifier.fit(X_train, y_train)
end = time.time()
total_training_time = end - start
print ("Done!\nTraining time (secs): {:.3f}".format(total_training_time))

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,f1_score

# confusion matrix usage to evaluate the quality of the output of a classifier on the data set.
# https://github.com/todddangerfarr/mlnd-p5-capstone-wearables-activity-tracking/blob/master/human-activities-and-postural-transitions.ipyn
cm = confusion_matrix(y_test, y_pred)

Labels = ['Walking','Walking Upstairs','Walking Downstairs','Sitting','Standing','Laying']
colormap = plt.cm.RdBu
fig = plt.figure(figsize=(10, 7))
ax = plt.axes()
sns.set(style='whitegrid', font_scale=1.5)
sns.heatmap(cm, 
            #fmt='.0f',
            fmt ='g',
            annot=True, 
            annot_kws={"size": 10},
            cmap=colormap,
            square=True,
            linecolor='white',
            xticklabels =Labels,
            yticklabels =Labels,
            ax = ax)

ax.set_title('Testing Confusion Matrix ')
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

fig.subplots_adjust(wspace=0, hspace=0)
plt.show()

normalised_confusion_matrix = np.array(cm, dtype=np.float32)/np.sum(cm)*100
print("")
print("Confusion matrix (normalised to % of total test data):")
print(normalised_confusion_matrix)

# print classification report of Precision-Recall, f1-score accuracy scores
# The classification_report function builds a text report showing the main classification metrics.
#  precision is the ability of the classifier not to label as positive a sample that is negative, 
# and recall is the ability of the classifier to find all the positive sample
from sklearn.metrics import classification_report, accuracy_score
print(classification_report(y_test, y_pred, target_names=Labels))
```

    Done!
    Training time (secs): 11.214
    


![png](human_activity_dectector_files/human_activity_dectector_26_1.png)


    
    Confusion matrix (normalised to % of total test data):
    [[16.355616    0.2714625   0.13573125  0.06786563  0.          0.        ]
     [ 0.2375297  15.710892    0.03393281  0.          0.          0.        ]
     [ 0.03393281  0.06786563 14.116051    0.03393281  0.          0.        ]
     [ 0.          0.06786563  0.         14.692908    1.8663046   0.03393281]
     [ 0.          0.          0.          0.6107906  17.441467    0.        ]
     [ 0.          0.          0.          0.          0.10179844 18.120121  ]]
                        precision    recall  f1-score   support
    
               Walking       0.98      0.97      0.98       496
      Walking Upstairs       0.97      0.98      0.98       471
    Walking Downstairs       0.99      0.99      0.99       420
               Sitting       0.95      0.88      0.92       491
              Standing       0.90      0.97      0.93       532
                Laying       1.00      0.99      1.00       537
    
              accuracy                           0.96      2947
             macro avg       0.97      0.96      0.96      2947
          weighted avg       0.97      0.96      0.96      2947
    
    


```python
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
multilabel_confusion_matrix(y_test, y_pred, labels=[1,2,3,4,5,6])
```




    array([[[2443,    8],
            [  14,  482]],
    
           [[2464,   12],
            [   8,  463]],
    
           [[2522,    5],
            [   4,  416]],
    
           [[2435,   21],
            [  58,  433]],
    
           [[2357,   58],
            [  18,  514]],
    
           [[2409,    1],
            [   3,  534]]], dtype=int64)




```python
# “The Matthews correlation coefficient is used in machine learning as a measure of the quality of 
# binary (two-class) classifications. It takes into account true and false positives and negatives 
# and is generally regarded as a balanced measure which can be used even if the classes are of very different sizes. https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix
# https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix
from sklearn.metrics import matthews_corrcoef
print('matthews_corrcoef')
matthews_corrcoef(y_test, y_pred)
```

    matthews_corrcoef
    




    0.9573822820535877




```python

from sklearn.metrics import hamming_loss
print('hamming loss')
hamming_loss(y_test, y_pred)
```

    hamming loss
    




    0.035629453681710214




```python
from sklearn.metrics import zero_one_loss
# The zero_one_loss function computes the sum or the average of the 0-1 classification loss (L0−1) over nsamples
print('zero one loss')
zero_one_loss(y_test, y_pred)
```

    zero one loss
    




    0.03562945368171022




```python
from sklearn.metrics import max_error
max_error(y_test, y_pred)
```




    3




```python
# https://scikit-learn.org/stable/modules/model_evaluation.html
#The sklearn.metrics module implements several loss, score, and utility functions to measure regression performance. 
#Some of those have been enhanced to handle the multioutput case: mean_squared_error, mean_absolute_error,explained_variance_score and r2_score.
```


```python
# The r2_score function computes the coefficient of determination, usually denoted as R².
# It represents the proportion of variance (of y) that has been explained by the independent variables in the model. 
# It provides an indication of goodness of fit and therefore a measure of how well unseen samples are likely to be predicted
# by the model, through the proportion of explained variance.
# As such variance is dataset dependent, R² may not be meaningfully comparable across different datasets. 
# Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). 
# A constant model that always predicts the expected value of y, disregarding the input features, would get a R² score of 0.0.
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
```




    0.9837496368230267



The confusion matrix is plotted to get better insight of model performance using mlxted to refrain from extra code via scikit.

The model performance is evident from the diagonal concentration of the values



```python

from sklearn.metrics import confusion_matrix
#model.load_weights("HAR_weights.hdf5")
# pred = model.predict(X_test)
# pred = np.argmax(y_pred,axis = 1) 
# y_true = np.argmax(y_test,axis = 1)
```


```python
# CM = confusion_matrix(y_test, y_pred)
# from mlxtend.plotting import plot_confusion_matrix
# fig, ax = plot_confusion_matrix(conf_mat=CM ,  figsize=(10, 5))
# plt.show()
```


```python
pca = PCA()
X_pca = pca.fit_transform(X_train)

# function borrowed from course content
def scree_plot(pca):
    '''
    Creates a scree plot associated with the principal components and displays the cumulative variance explained
    
    INPUT: pca - the result of instantiating PCA in sklearn
            
    OUTPUT: None
    '''
    
    num_components=len(pca.explained_variance_ratio_)
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
 
    plt.figure(figsize=(8, 6))
    ax = plt.subplot(111)
    cumvals = np.cumsum(vals)

    ax.plot(ind, cumvals)

    ax.set_xlabel("No. Principal Components")
    ax.set_ylabel("Variance Explained (%)")
    plt.title('Explained Variance')
    
scree_plot(pca)
```


![png](human_activity_dectector_files/human_activity_dectector_37_0.png)


Model Build
The objective is to build a classifier that can classify the different activity types, even when extended to new data (i.e.: new participants). Accuracy will therefore be used as the evaluation metric.


```python

# fit a KNN to the data 
knn = KNeighborsClassifier()
knn.fit(X_train, y_train.values.ravel())

train_pred = knn.predict(X_train)
y_pred = knn.predict(X_test)

print('Accuracy score TRAIN: ', format(accuracy_score(y_train, train_pred)))
print('Accuracy score TEST: ', format(accuracy_score(y_test, y_pred)))
```


```python

# confusion matrix
cm = confusion_matrix(y_test, y_pred)

fig = plt.figure(figsize=(16, 6))
ax = plt.axes()
sns.heatmap(cm, 
            fmt='.0f', 
            annot=True, 
            cmap='Blues',
            xticklabels=['Walking','Walking Upstairs','Walking Downstairs','Sitting','Standing','Laying'],
            yticklabels=['Walking','Walking Upstairs','Walking Downstairs','Sitting','Standing','Laying'],
            ax = ax)

ax.set_title('Confusion Matrix for the Classifier')
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
plt.show()

from sklearn.metrics import classification_report, accuracy_score
print(classification_report(y_test, y_pred))
```


```python
d = { "Index":np.arange(2947) , "Activity":y_pred }
final = pd.DataFrame(d)
final.to_csv( './data/UCI HAR Dataset/human_activity_predictions.csv' , index = False)
HARpredictions = pd.read_csv('./data/UCI HAR Dataset/human_activity_predictions.csv')
print(HARpredictions)
```


```python
# Introduction to Ensembling/Stacking in Python
# https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python
```


```python
# Let's convert this notebook to a README automatically for the GitHub project's title page:
!jupyter nbconvert --to markdown human_activity_dectector.ipynb
!mv human_activity_dectector.md NOTEBOOK.md
```


```python

```
