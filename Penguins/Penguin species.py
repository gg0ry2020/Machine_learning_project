import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"Data\Original Data\penguins.csv")
df.dropna(inplace=True)

label = LabelEncoder()
df["sex"] = label.fit_transform(df["sex"])

X = df.drop(columns=["sex"])
y = df["sex"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

features = X.columns.tolist()

scaler = StandardScaler()
scaler.fit(X_train[features])
X_train_scaled = scaler.transform(X_train[features])
X_test_scaled = scaler.transform(X_test[features])

pd.DataFrame(X_train_scaled, columns=features).to_csv("Data/Preprocessed data/X_train.csv", index=False)
pd.DataFrame(X_test_scaled, columns=features).to_csv("Data/Preprocessed data/X_test.csv", index=False)
y_train.to_csv("Data/Preprocessed data/Y_train.csv", index=False)
y_test.to_csv("Data/Preprocessed data/Y_test.csv", index=False)

models = {
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "NaiveBayes": GaussianNB(),
    "ANN": MLPClassifier(max_iter=1000, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
}

for name, model in models.items():
    print(f"\n== Training {name} ==")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    pd.DataFrame({"Prediction": y_pred}).to_csv(f"Data/Results/prediction_{name}.csv", index=False)

    print(classification_report(y_test, y_pred, zero_division=0))
    
    labels = [1, 2] 
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label.inverse_transform(labels),
            yticklabels=label.inverse_transform(labels))
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
