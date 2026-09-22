import pandas as pd     # use pandas for handling CSV files and working with DataFrames
import matplotlib.pyplot as plt     # use matplotlib for data visualization and graphs
from sklearn.model_selection import train_test_split    # use 'train_test_split' to split arrays or matrices into random train and test subsets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import seaborn as sns