import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

delay = pd.read_csv("airlines_delay.csv")

# Checking the Data and EDA
# Looking at first 5 rows
delay.head()

# Looking at variable types
delay.info()

# Looking at shape of data
delay.shape

# Look at unique values of categorical variables
airline = delay["Airline"].unique()
print(airline)

airportfrom = delay["AirportFrom"].unique()
print(airportfrom)

airportto = delay["AirportTo"].unique()
print(airportto)

# Checking for NA values
print(delay.isna())

# Deleting column *Flight* because it is not needed
columns_delete = ["Flight"]
delay.drop(columns=columns_delete, inplace=True)
delay.info()

# Checking count of delays
plt.figure(figsize=(10,14))
sns.countplot(x="Class", data = delay)
plt.title("Delay Count")
plt.show()

# Now exploration by comparing *Airline* to other variables
plt.figure(figsize=(10, 14))
sns.countplot(x="Airline", hue="Class", data=delay)
plt.title("Airline Delay Count")
plt.show()

# It seems WN has more delays
count = len(delay[(delay['Class'] == 1) & (delay['Airline'] == "WN")])
print(count)
# More no delays

# Checking departure time and airline
plt.figure(figsize=(10,14))
sns.boxplot(x="Airline", y="Time", data=delay)
plt.title("Airline Departure Time")
plt.show()
# Departure time fairly the same

# Checking departure time distribution
plt.figure(figsize=(10, 14))
sns.histplot(x="Time", data=delay)
plt.title("Departure Time Distribution")
plt.show()

# Checking length of time in each airline
plt.figure(figsize=(10, 14))
sns.boxplot(x="Airline", y="Length", data=delay)
plt.title("Airline Length Time")
plt.show()

# Checking departure length distribution
plt.figure(figsize=(10, 14))
sns.histplot(x="Length", data=delay)
plt.title("Departure Time Distribution")
plt.show()
# Right skewed

# Checking which day of the week is most popular for flights
plt.figure(figsize=(10, 14))
sns.countplot(x="DayOfWeek", data=delay)
plt.title("Popular Days")
plt.show()
# Days in the middle of the week are more popular

# Now that we did some basic EDA, we can start to predict!

# Encode Categorical Variables
delay["Airline"] = LabelEncoder().fit_transform(delay["Airline"])
delay["AirportFrom"] = LabelEncoder().fit_transform(delay["AirportFrom"])
delay["AirportTo"] = LabelEncoder().fit_transform(delay["AirportTo"])

# Train and Test Data
# Since the dataset is pretty large, we can do a 80-20 split
X=delay.drop('Class',axis=1)
y=delay['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest - Model Building
# Building model with 100 decision trees
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Random Forest - Evaluation
pred = rf.predict(X_test)

# Creating confusion matrix
cm = confusion_matrix(y_test, pred)
print(f"Confusion Matrix: {cm}")

# Calculate accuracy and precision
accuracy = accuracy_score(y_test, pred)
print(f"Accuracy: {accuracy}")
precision = precision_score(y_test, pred, average="weighted")
print(f"Precision: {precision}")

# Random Forest - Visualizing
# Let's make a heatmap
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Now an importance graph
# Get importance of features
feat_imp = rf.feature_importances_
# Indices of features
indices = np.argsort(feat_imp)[::-1]
# Get names of features
feature_names = X_train.columns

# Now graphing
plt.figure(figsize=(10, 8))
plt.bar(range(X_train.shape[1]), feat_imp[indices], align="center")
plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=90)
plt.title("Feature Importances")
plt.show()