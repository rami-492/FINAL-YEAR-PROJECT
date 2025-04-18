import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
import time
import urllib.request
import telepot

# Initialize StandardScaler
scaler = StandardScaler()

# Initialize Telegram Bot
bot = telepot.Bot('8106749068:AAFaQHFdm1PH_436rGoMiwctR3EyFH-D9fg')
chat_id = '6361288664'
print("Telegram bot is ready")
bot.sendMessage(chat_id, 'BOT STARTED')
time.sleep(2)

# Function to fetch sensor data from ThingSpeak (mock example)
def takeInput():
    while True:
        try:
            # Fetching live data from ThingSpeak API
            r_link = 'https://api.thingspeak.com/channels/208729/fields/1/last?results=2'
            f = urllib.request.urlopen(r_link)
            pr1 = (f.readline()).decode()

            r_link = 'https://api.thingspeak.com/channels/208729/fields/2/last?results=2'
            f = urllib.request.urlopen(r_link)
            pr2 = (f.readline()).decode()

            r_link = 'https://api.thingspeak.com/channels/208729/fields/3/last?results=2'
            f = urllib.request.urlopen(r_link)
            pr3 = (f.readline()).decode()

            print(f'temperature: {pr1}, turbidity: {pr2}, pH: {pr3}')
            data = str(pr1) + ',' + str(pr2) + ',' + str(pr3)

            if data is not None:
                # Convert the string data into a 2D NumPy array for prediction
                X = np.array([data.split(',')], dtype=np.float32)

                # Normalize input data using the same scaler
                X = scaler.transform(X)  # Normalize the input features

                # Make a prediction using the trained KNN model
                y_pred = knn_classifier.predict(X)

                # Check the prediction and send appropriate message
                if y_pred == 2:
                    print('Abnormal Water quality')
                    bot.sendMessage(chat_id, 'Abnormal Water quality')
                elif y_pred == 1:
                    print('Bad Water quality')
                    bot.sendMessage(chat_id, 'Bad Water quality')
                elif y_pred == 0:
                    print('Good Water quality')
                    bot.sendMessage(chat_id, 'Good Water quality')

            time.sleep(15)  # Wait for 15 seconds before fetching the next data point

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(15)  # Wait for a while before trying again in case of error

# Load the dataset
data = pd.read_csv("water_quality_dataset.csv")

# Split the data into features (X) and target (y)
y = data['water_quality_index']
X = data.drop(['water_quality_index'], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# Normalize the features using StandardScaler
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the KNN Classifier
knn_classifier = KNeighborsClassifier(n_neighbors=4)
knn_classifier.fit(X_train, y_train)

# Train the SVC Classifier
svc_clf = SVC(gamma='scale')
svc_clf.fit(X_train, y_train)

# Train the Logistic Regression Classifier
logreg_classifier = LogisticRegression(solver="liblinear")
logreg_classifier.fit(X_train, y_train)

# Make predictions for each model
knn_preds = knn_classifier.predict(X_test)
svc_preds = svc_clf.predict(X_test)
logreg_preds = logreg_classifier.predict(X_test)

# Calculate accuracy for each model
knn_acc = accuracy_score(y_test, knn_preds)
svc_acc = accuracy_score(y_test, svc_preds)
logreg_acc = accuracy_score(y_test, logreg_preds)

print(f"Accuracy with KNN: {knn_acc:.2f}")
print(f"Accuracy with SVC: {svc_acc:.2f}")
print(f"Accuracy with Logistic Regression: {logreg_acc:.2f}")

# Plotting accuracy comparison
plt.figure(figsize=(8, 6))
plt.bar(['KNN', 'SVC', 'Logistic Regression'], [knn_acc, svc_acc, logreg_acc], color=['blue', 'green', 'red'])
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.ylim(0, 1)
plt.show()

# Precision vs Threshold plot for all models
thresholds = np.linspace(0, 1, num=101)
knn_precision = []
svc_precision = []
logreg_precision = []

for t in thresholds:
    knn_pred_t = (knn_preds >= t).astype(int)
    svc_pred_t = (svc_preds >= t).astype(int)
    logreg_pred_t = (logreg_preds >= t).astype(int)

    # Using 'macro' to handle multiclass precision
    knn_precision.append(precision_score(y_test, knn_pred_t, average='macro'))
    svc_precision.append(precision_score(y_test, svc_pred_t, average='macro'))
    logreg_precision.append(precision_score(y_test, logreg_pred_t, average='macro'))

# Plot Precision vs Threshold
plt.figure(figsize=(8, 6))
plt.plot(thresholds, knn_precision, label='KNN', color='blue')
plt.plot(thresholds, svc_precision, label='SVC', color='green')
plt.plot(thresholds, logreg_precision, label='Logistic Regression', color='red')
plt.xlabel('Threshold')
plt.ylabel('Precision')
plt.title('Precision vs Threshold')
plt.legend(loc='best')
plt.show()

# Confusion Matrix for each model
def plot_confusion_matrix(model, model_name):
    cm = confusion_matrix(y_test, model.predict(X_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()

# Plot confusion matrix for each model
plot_confusion_matrix(knn_classifier, 'KNN')
plot_confusion_matrix(svc_clf, 'SVC')
plot_confusion_matrix(logreg_classifier, 'Logistic Regression')

# Start Live Data Collection (Real-time data prediction)
takeInput()
