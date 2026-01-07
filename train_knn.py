import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv('./sign_dataset.csv')

print(df.head())
print("Total samples:", len(df))
print("Labels:", df["label"].unique())

x = df.drop("label", axis=1).values
y = df["label"].values

X_train, X_test, Y_train, Y_test = train_test_split(
  x, 
  y,
  test_size=0.2,
  random_state=42
)

knn = KNeighborsClassifier(
  n_neighbors=5,
  weights="distance"
)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)

joblib.dump(knn, "knn_model.pkl")
print("Model saved as knn_model.pkl")
