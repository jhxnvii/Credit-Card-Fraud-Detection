# Importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset as dataframe
df = pd.read_csv('creditcard.csv')
df = df[ ["Time","Amount","Class"] ]

# Viewing the dataset
df

# Sorting duplicate rows
df_no_duplicates=df.drop_duplicates()
print(df_no_duplicates)

# Exploring the data
df.describe()

# Missing values
df.isnull().sum()

# Separate identity and target (Data-Preprocessing)
X = df.drop('Class', axis=1)
y = df['Class']

X
y

# Split data into training and testing partitions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression() 
model.fit(X_train, y_train)
X_test

# Make predictions on the testing set
y_pred = model.predict(X_test)

# count predicted frauds & actual frauds
actual_count = len(y_test[y_test == 0])
pred_count = len(y_pred[y_pred == 0])

# Display number of actual frauds and no. of predicted frauds.
print("Actual fraud: ", actual_count)
print("Predicted fraud: ", pred_count)
print(len(y_pred))
Legit_transactions = len(y [y == 1])
print("Legit Transactions:", Legit_transactions)

# Calculate & display model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Classification Report
print(classification_report(y_test, y_pred))

# Plot Graph
plt.figure(figsize=(10, 6))
plt.bar(['Prediction', 'Actual'], [pred_count, actual_count], color = ['red', 'blue'])
plt.title('Accuracy for predicted frauds vs actual frauds.')
plt.xlabel('Prediction')
plt.ylabel('Actual')
plt.ylim(56800, 56900)
plt.grid(axis='y')
plt.show()

# Creating the confusion matirx
cm = confusion_matrix(y_test, y_pred)
cm

# Plotting the confusion Matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.xticks(ticks=[0.5, 1.5], labels=["Fraud Transactions", "Valid Transactions"])
plt.yticks(ticks=[0.5, 1.5], labels=["Fraud Transactions", "Valid Transactions"])
plt.show()

new_data = pd.read_csv("new_data.csv")
new_data

new_pred = model.predict(new_data)
print(new_pred)

if(new_pred == 0):
    print("This is a fraud transaction.")
else:
    print("This is a valid transaction.")