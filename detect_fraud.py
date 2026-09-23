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
feature = data.drop("Class", axis=1)
target = data["Class"]

# visualize class distribution using a bar graph
'''
plt.bar(y.value_counts().index, y.value_counts().values, color=['salmon','skyblue'])
plt.xticks([0,1], ['Non-Fraud', 'Fraud'])
plt.ylabel("Count")
plt.title("Original Class Distribution")
plt.show()
'''

# create test and training subsets
feature_train, feature_test, target_train, target_test = train_test_split(
    feature, target, test_size=0.2, random_state=42
)

# train model on data BEFORE resampling with SMOTE
model = LogisticRegression(max_iter=10000)
model.fit(feature_train, target_train)

# predict target values on unknown inputs from the test set
target_pred = model.predict(feature_test)

# accuracy of model on the testing set
print("Accuracy BEFORE SMOTE:", round(accuracy_score(target_test, target_pred)*100, 2), "%\n")

# classification report to measure how well the model is performing
print("Classification Report BEFORE SMOTE:\n", classification_report(target_test, target_pred))

