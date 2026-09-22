import pandas as pd     # use pandas for handling CSV files and working with DataFrames
import matplotlib.pyplot as plt     # use matplotlib for data visualization and graphs
from sklearn.model_selection import train_test_split    # use 'train_test_split' to split arrays or matrices into random train and test subsets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE    #use SMOTE for resampling minority class
import seaborn as sns   # more data visualization

# data is already cleaned but I still checked for missing values in jupyter 

# load the data into a dataframe named data
data = pd.read_csv("creditcard_fraud_detection.csv")

# seperate features (x) from target value (y)
x = data.drop("Class", axis=1)
y = data["Class"]

# visualize class distribution using a bar graph
'''
plt.bar(y.value_counts().index, y.value_counts().values, color=['salmon','skyblue'])
plt.xticks([0,1], ['Non-Fraud', 'Fraud'])
plt.ylabel("Count")
plt.title("Original Class Distribution")
plt.show()
'''

# create test and training subsets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# train model on data BEFORE resampling with SMOTE
model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

