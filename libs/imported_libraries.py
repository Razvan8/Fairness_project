###IMPORTS
import joblib
import os
import sys
from ucimlrepo import fetch_ucirepo

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

from scipy.stats import  spearmanr
from scipy.stats import norm

import matplotlib.pyplot as plt

from itertools import product

from fairlearn.postprocessing import ThresholdOptimizer
