import os
import pandas as pd
import json
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Paso 1.

def load_and_clean(path):
    df = pd.read_csv(path, compression="zip")
    df = df.rename(columns = {'default payment next month':'default'})
    df = df.drop('ID', axis=1)
    df = df.loc[df["MARRIAGE"] != 0] 
    df = df.loc[df["EDUCATION"] != 0] 
    df = df.dropna()
    df['EDUCATION'] = df['EDUCATION'].map(lambda x: 4 if x>4 else x)

    return df

df_train = load_and_clean('files/input/train_data.csv.zip')
df_test = load_and_clean('files/input/test_data.csv.zip')

# Paso 2.

x_train = df_train.drop(columns='default')
y_train = df_train['default']

x_test = df_test.drop(columns='default')
y_test = df_test['default']

#
# Paso 3.


def make_pipeline():

    categoric_columns = ['SEX','EDUCATION','MARRIAGE']

    other_columns = [col for col in x_train.columns if col not in categoric_columns]

    preprocessor = ColumnTransformer(transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'), categoric_columns),
        ("scaler", StandardScaler(with_mean=True, with_std=True), other_columns)
        ],remainder='passthrough')


    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ('pca', PCA()),
            ('selectkbest', SelectKBest()),
            ('svm', SVC(kernel="rbf", random_state=12345, max_iter=-1) )
        ]
    )

    return pipeline

pipeline = make_pipeline()

#
# Paso 4.
#
def make_grid_search(estimator):

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid = {
            "pca__n_components": [20, x_train.shape[1] - 2],  
            'selectkbest__k': [12],           
            'svm__kernel': ["rbf"],           
            'svm__gamma': [0.1],                  
        },
        cv=10,
        scoring='balanced_accuracy',
        n_jobs=-1,
        verbose=3
    )

    return grid_search

estimator = make_grid_search(pipeline)
estimator.fit(x_train, y_train)
#
# Paso 5.


def save_estimator(estimator, path):

    import pickle
    import gzip
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True) 
    with gzip.open(path, "wb") as f:
        pickle.dump(estimator, f)

save_estimator(estimator,'files/models/model.pkl.gz')

#
# Paso 6.

def check_estimator(estimator, x, y, dataset):

    y_pred = estimator.predict(x)

    precision = round(precision_score(y, y_pred), 4)
    balanced_accuracy = round(balanced_accuracy_score(y, y_pred), 4)
    f1 = round(f1_score(y, y_pred), 4)
    recall = round(recall_score(y, y_pred), 4)

    metrics = {
        "type": "metrics",
        "dataset": dataset,
        "precision": precision,
        "balanced_accuracy": balanced_accuracy,
        "recall": recall,
        "f1_score": f1
    }
    
    return metrics, y_pred, y



metrics_train, y_pred_train, y_train = check_estimator(estimator, x_train, y_train, "train")
metrics_test, y_pred_test, y_test = check_estimator(estimator, x_test, y_test, "test")

#
# Paso 7.

def c_matrix(y_true, y_pred, dataset):
    cm = confusion_matrix(y_true, y_pred)
    return {
        "type": "cm_matrix", "dataset": dataset,
        "true_0": {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
        "true_1": {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])}
    }

os.makedirs("files/output", exist_ok=True)

c_train = c_matrix(y_train, y_pred_train, "train")
c_test = c_matrix(y_test, y_pred_test, "test")

with open("files/output/metrics.json", "w") as file:
        file.write(json.dumps(metrics_train) + "\n")
        file.write(json.dumps(metrics_test) + "\n")
        file.write(json.dumps(c_train) + "\n")
        file.write(json.dumps(c_test) + "\n")
