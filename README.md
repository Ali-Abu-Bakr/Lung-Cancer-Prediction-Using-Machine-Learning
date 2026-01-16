Lung Cancer Prediction

Project Overview
This project aims to predict the presence of lung cancer in individuals based on various demographic data and symptoms. Using a survey-based dataset, we implemented several machine learning models to classify cases and evaluated their performance to find the most accurate prediction method.


Dataset Features
The dataset contains 309 records with 16 features:


Demographics: GENDER, AGE.


Lifestyle & Environment: SMOKING, PEER_PRESSURE, ALCOHOL_CONSUMING.


Symptoms: YELLOW_FINGERS, ANXIETY, CHRONIC_DISEASE, FATIGUE, ALLERGY, WHEEZING, COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN.


Target: LUNG_CANCER (YES/NO).

Technical Stack

Language: Python 

Libraries:


Data Handling: pandas, numpy 


Visualization: seaborn, matplotlib 


Machine Learning: scikit-learn


Below is a detailed README.md file format for the Lung Cancer Prediction project, based on the analysis and implementation details found in the provided notebook.

Lung Cancer Prediction
Project Overview
This project aims to predict the presence of lung cancer in individuals based on various demographic data and symptoms. Using a survey-based dataset, we implemented several machine learning models to classify cases and evaluated their performance to find the most accurate prediction method.

Dataset Features
The dataset contains 309 records with 16 features:


Demographics: GENDER, AGE.


Lifestyle & Environment: SMOKING, PEER_PRESSURE, ALCOHOL_CONSUMING.


Symptoms: YELLOW_FINGERS, ANXIETY, CHRONIC_DISEASE, FATIGUE, ALLERGY, WHEEZING, COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN.


Target: LUNG_CANCER (YES/NO).

Technical Stack

Language: Python 

Libraries:


Data Handling: pandas, numpy 


Visualization: seaborn, matplotlib 


Machine Learning: scikit-learn 

Data Preprocessing
Categorical Encoding:


GENDER was mapped as M:1, F:2.


LUNG_CANCER was mapped as YES:1, NO:2.

Exploratory Data Analysis:

Checked for null values (found 0 missing entries).

Generated a correlation heatmap which revealed a strong relationship (0.56) between ANXIETY and YELLOW_FINGERS


Data Splitting: Split the dataset into training (206 samples) and testing (103 samples) using a 1/3 test size ratio.


Below is a detailed README.md file format for the Lung Cancer Prediction project, based on the analysis and implementation details found in the provided notebook.

Lung Cancer Prediction
Project Overview
This project aims to predict the presence of lung cancer in individuals based on various demographic data and symptoms. Using a survey-based dataset, we implemented several machine learning models to classify cases and evaluated their performance to find the most accurate prediction method.

Dataset Features
The dataset contains 309 records with 16 features:


Demographics: GENDER, AGE.


Lifestyle & Environment: SMOKING, PEER_PRESSURE, ALCOHOL_CONSUMING.


Symptoms: YELLOW_FINGERS, ANXIETY, CHRONIC_DISEASE, FATIGUE, ALLERGY, WHEEZING, COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN.


Target: LUNG_CANCER (YES/NO).

Technical Stack

Language: Python 

Libraries:


Data Handling: pandas, numpy 


Visualization: seaborn, matplotlib 


Machine Learning: scikit-learn 

Data Preprocessing
Categorical Encoding:


GENDER was mapped as M:1, F:2.


LUNG_CANCER was mapped as YES:1, NO:2.

Exploratory Data Analysis:

Checked for null values (found 0 missing entries).

Generated a correlation heatmap which revealed a strong relationship (0.56) between ANXIETY and YELLOW_FINGERS.


Data Splitting: Split the dataset into training (206 samples) and testing (103 samples) using a 1/3 test size ratio.

Models and Performance
We implemented and compared six different classification algorithms. Below are the performance results on the test set:

Model	Accuracy	Precision	Recall	F1 Score
Random Forest	89.32%	0.9043	0.9770	0.9392
Logistic Regression	88.35%	0.8947	0.9770	0.9341
Naive Bayes	86.41%	0.9101	0.9310	0.9205
K-Nearest Neighbors	84.47%	0.8586	0.9770	0.9140
Decision Tree	84.47%	0.8586	0.9770	0.9140
SVM (RBF Kernel)	84.47%	0.8447	1.0000	0.9158
Key Insights

Best Overall Model: Random Forest achieved the highest accuracy and F1 score, making it the most balanced model for this dataset.


High Sensitivity: The Support Vector Machine (SVM) model achieved a perfect recall of 1.0, meaning it correctly identified every positive lung cancer case in the test set.


Data Imbalance: The dataset is imbalanced, with 270 "YES" cases and only 39 "NO" cases, which explains the high recall across most models.

How to Run
Ensure you have Python installed.

Install required libraries: pip install pandas scikit-learn seaborn matplotlib.

Load the survey lung cancer.csv dataset.

Execute the script to see model training and evaluation metrics.
