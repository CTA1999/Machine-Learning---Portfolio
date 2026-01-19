
import pandas as pd
from sktime.datatypes._panel._convert import from_2d_array_to_nested
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split



train = pd.read_csv('train.csv', header=None)
test = pd.read_csv('test.csv', header=None)

train = train.apply(pd.to_numeric, errors='coerce').fillna(0)
test = test.apply(pd.to_numeric, errors='coerce').fillna(0)

train_ft = train.iloc[:, 0:187]
train_label = train.iloc[:, 187]

scaler = MinMaxScaler()
train_ft_scaled = pd.DataFrame(scaler.fit_transform(train_ft))
test_scaled = pd.DataFrame(scaler.transform(test))

#train and validation split

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    train_ft_scaled, train_label, test_size=0.2, random_state=42, stratify=train_label
)

X_train_nested = from_2d_array_to_nested(X_train_split)
X_val_nested = from_2d_array_to_nested(X_val_split)

clf = TimeSeriesForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_train_nested, y_train_split)

val_pred = clf.predict(X_val_nested)

accuracy = accuracy_score(y_val_split, val_pred)
print(f"Validation Accuracy: {accuracy:.4f}")




#retrain on full data
X_train_nested = from_2d_array_to_nested(train_ft_scaled)
X_test_nested = from_2d_array_to_nested(test_scaled)

clf = TimeSeriesForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

clf.fit(X_train_nested, train_label)

pred = clf.predict(X_test_nested)

#load predictions to csv file
predictions = pd.DataFrame(pred).to_csv('predictions.csv', index=False, header=False)





