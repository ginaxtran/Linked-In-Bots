# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("linked_in_data.csv")

# Preprocess data
# Handle categorical variables
categorical_columns = ["Workplace", "Location", "Experiences", "Activities", "About", "Photo"]

# Ensure categorical columns are of type 'category'
for column in categorical_columns:
    df[column] = df[column].astype("category")

# Perform one-hot encoding for categorical variables
encoder = OneHotEncoder(handle_unknown='ignore')
encoded_data = encoder.fit_transform(df[categorical_columns])

# Combine encoded data with numerical features
numerical_features = [
    "Connections", "Followers", "Number of Experiences",
    "Number of Educations", "Number of Licenses",
    "Number of Volunteering", "Number of Skills",
    "Number of Recommendations", "Number of Projects",
    "Number of Publications", "Number of Courses",
    "Number of Honors", "Number of Scores",
    "Number of Languages", "Number of Organizations",
    "Number of Interests", "Number of Activities"
]

# Combine numerical features and encoded categorical data into one DataFrame
X = pd.concat([
    df[numerical_features], pd.DataFrame(encoded_data.toarray())
], axis=1)

# Convert feature names to strings (if necessary)
X.columns = X.columns.astype(str)

# Define labels
y = df["Label"].replace({
    "LLPs": 0,
    "FLPs": 1,
    "CLPs based on legitimate profiles' statistics": 10,
    "CLPs based on fake profiles' statistics": 11
})

# Split the data into training and temporary sets (80% training, 20% temporary)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Split the temporary set into testing and validation sets (50% testing, 50% validation of temporary set)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Initialize the Random Forest classifier with regularization (reduced max depth)
classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Train the classifier on the training set
classifier.fit(X_train, y_train)

# Evaluate the model using cross-validation on the training set
cross_val_scores = cross_val_score(classifier, X_train, y_train, cv=5)
print("Cross-Validation Scores:", cross_val_scores)
print("Mean Cross-Validation Score:", cross_val_scores.mean())

# Predict on the training set
y_train_pred = classifier.predict(X_train)

# Predict on the testing set
y_test_pred = classifier.predict(X_test)

# Evaluate the model's performance
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

# Generate classification reports
train_report = classification_report(y_train, y_train_pred)
test_report = classification_report(y_test, y_test_pred)

print("\nTraining Classification Report:\n", train_report)
print("\nTesting Classification Report:\n", test_report)