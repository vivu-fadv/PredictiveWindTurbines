## libraries for reading and manipulating data
import pandas as pd
import numpy as np

## libraries for data visualization
import matplotlib.pyplot as plt
import seaborn as sns

## libraries for splitting data
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

## libraries for imbalance dataset
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

## libraries for computing accuracy score
from sklearn.metrics import (make_scorer,f1_score, accuracy_score, recall_score, precision_score,
                            confusion_matrix, roc_auc_score)

## library for data scaling
from sklearn.preprocessing import StandardScaler

## library for hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV


## model building
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ( AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier, BaggingClassifier
)
from xgboost import XGBClassifier

## suppress warnings
import warnings
warnings.filterwarnings("ignore")

## loading dataset with pandas
train_generator_data = pd.read_csv("D:\Work\PredictiveWindTurbines\.venv\kaggle\input/Train.csv")
test_generator_data = pd.read_csv("D:\Work\PredictiveWindTurbines\.venv\kaggle\input/Test.csv")

## creating a copy of the data
df_train = train_generator_data.copy()
df_test= test_generator_data.copy()

