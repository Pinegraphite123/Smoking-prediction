# Using bio-signals to predict smoking status

## Pine Xie 

## Workflow
1.   [Exploratory Data Analysis](#first-bullet)
2.   [Classification imbalance](#second-bullet)
3.   [Feature selection](#third-bullet)
4.   [Supervised learning](#fourth-bullet)
5.   [Semi-supervised learning](#fifth-bullet)
6.   [Unsupervised learning](#sixth-bullet)

### Each ML learning section contains evaluation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, train_test_split
```

Download data from kaggle at https://www.kaggle.com/datasets/gauravduttakiit/smoker-status-prediction-using-biosignals/data?select=train_dataset.csv
Use train set only since test set does not have label so it cannot be evaluated

## Exploratory Data Analysis <a class="anchor" id="first-bullet"></a>

```python
train_df = pd.read_csv('/content/train_dataset.csv')
print(train_df.head())
```

<div>
  
| age | height(cm) | weight(kg) | waist(cm) | eyesight(left) | eyesight(right) | hearing(left) | hearing(right) | systolic | relaxation | ... | HDL | LDL | hemoglobin | Urine protein | serum creatinine | AST  | ALT  | Gtp | dental caries | smoking |
|-----|------------|------------|-----------|----------------|-----------------|---------------|----------------|----------|------------|-----|-----|-----|------------|---------------|-----------------|------|------|-----|---------------|---------|
|  35 |        170 |         85 |      97.0 |            0.9 |             0.9  |             1 |              1 |      118 |         78 | ... |  70 | 142 |       19.8 |             1 |             1.0  |   61 |  115 | 125 |             1 |       1 |
|  20 |        175 |        110 |     110.0 |            0.7 |             0.9  |             1 |              1 |      119 |         79 | ... |  71 | 114 |       15.9 |             1 |             1.1  |   19 |   25 |  30 |             1 |       0 |
|  45 |        155 |         65 |      86.0 |            0.9 |             0.9  |             1 |              1 |      110 |         80 | ... |  57 | 112 |       13.7 |             3 |             0.6  | 1090 | 1400 | 276 |             0 |       0 |
|  45 |        165 |         80 |      94.0 |            0.8 |             0.7  |             1 |              1 |      158 |         88 | ... |  46 |  91 |       16.9 |             1 |             0.9  |   32 |   36 |  36 |             0 |       0 |
|  20 |        165 |         60 |      81.0 |            1.5 |             0.1  |             1 |              1 |      109 |         64 | ... |  47 |  92 |       14.9 |             1 |             1.2  |   26 |   28 |  15 |             0 |       0 |
<p>5 rows x 23 columns</p>

</div>


### Check if there is any NA or unencoded values
```python
train_df.isnull().sum()
```

## Class imbalance handling <a class="anchor" id="second-bullet"></a>
### Check class imbalance
```python
ax = sns.countplot(x='smoking', data=(train_df))
ax.set_xticklabels(['non-smoking', 'smoking'])
```
![Class imbalance](https://github.com/Pinegraphite123/Smoking-prediction/blob/fd8aa0a4d16530bef762a4d832ed99dc42ba3a23/Graphs/Class%20imbalance.png))

There is a slight class imbalance.

Since there is enough number of samples to learn from, downsampling of non-smoking population should be a better choice versus upsampling of smoking population. Unlike oversampling or SMOTE, downsapling does not introduce synthetic data, which reduces the chance of overfitting because it might increase the instances of the minority class based on existing ones.

```python
### df1 is the smoking population, df1 is the non-smoking population

df1 = train_df[train_df.smoking == 1]
df0 = train_df[train_df.smoking == 0]

df0_downsampled = df0.sample(n=len(df1), random_state=123)

# Combine minority class with downsampled majority class
downsampled = pd.concat([df0_downsampled, df1])

# Display new class counts
ax = sns.countplot(x='smoking', data=downsampled)
ax.set_xticklabels(['non-smoking', 'smoking'])
```

![class imbalance handled](https://github.com/Pinegraphite123/Smoking-prediction/blob/main/Graphs/Class%20imbalance%20handled.png?raw=true)

```python
### downsampled is our new dataset to be used

train_df = downsampled
```

# Feature selection <a class="anchor" id="third-bullet"></a>
keep the important and meaningful feature and remove correlated features

```python
### Some basic research tells me height has the least relationship with smoking
train_df = train_df.drop(['height(cm)'], axis=1)
```

Checking if there is multicollnearity on the features. If two features are highly correlated, one of them can be dropped to save some computational resource and reduce confounding variable.

```python
# Using variance inflation factor (VIF) to check multicollinearity, a higher VIF indicates a stronger correlation compared to rest of the features.
# Generally a VIF > 5 is moderate to high in multicollnearity.

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.datasets import make_regression

X_train_vif = train_df.copy()

# Adding a constant column for the intercept
X_train_vif['Intercept'] = 1

# Calculating VIF for each feature
vif_data = pd.DataFrame()
vif_data["Feature"] = X_train_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_train_vif.values, i) for i in range(X_train_vif.shape[1])]

# Removing the intercept row for display
vif_data = vif_data[vif_data["Feature"] != "Intercept"]

plt.figure(figsize=(30, 6))
sns.barplot(x='Feature', y='VIF', data=vif_data)
plt.xticks(rotation=45)
plt.show()
```
![VIF](https://github.com/Pinegraphite123/Smoking-prediction/blob/main/Graphs/VIF.png?raw=true)

Using pearson correlation together with VIF, one of the correlated features can be dropped
```python
correlation_matrix = train_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()
```

![CorrMatrix](https://github.com/Pinegraphite123/Smoking-prediction/blob/main/Graphs/CorrMatrix.png?raw=true)

Features that to be dropped
1. weight: weight might be less representative than waist because waist is more indicative of organ fat, which is directly linked to smoking
2. Cholesterol: basically HDL + LDL, LDL, the bad cholesterol might be more direct than just cholesterol
3. AST: ALT is more indicative of liver injury, which is directly linked to smoking
4. One of the hearing: only a mild correlation. Just drop one of the two, the implication of right or left hearing isnt clear.

```python
train_df = train_df.drop(['weight(kg)', 'Cholesterol', 'AST', 'hearing(left)'], axis=1)
correlation_matrix = train_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()
```

![CorrMatrixCleaned](https://github.com/Pinegraphite123/Smoking-prediction/blob/main/Graphs/CorrMatrixCleaned.png?raw=true)


# Supervised learning <a class="anchor" id="fourth-bullet"></a>
Using Gridsearchcv to exhaustly find the best hyperparameter so you can tweek with other option to improve the model, such as adjusting train test split

Split train data into train data and test data

80% train data, 20% test data
```python
X = train_df.iloc[:,:-1]
y = train_df['smoking']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1234, stratify=y)
```


Chosen model:

Logistic regression

Random forest

Decision tree

These model is chosen because the goal is data classification.

```python
# Scale the train data
scaler = StandardScaler()

# fit on standardscaler computes mean and stv for each feature
scaler.fit(X_train)

# Apply transform to both the training set and the test set, so the data is scaled
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

Use GridSearchCv to find the optimal parameters
```python
def find_best_model(X, y):
    models = {
        'logistic_regression': { ### Penalty='elasticnet',both L1 and L2 regularization. Solver='saga', algorithm for big dataset for faster computation. Max_iter, Maximum number of iterations taken for gridsearch to reach best parameters. ###
            'pipe': Pipeline([('scaler', StandardScaler()), ('LR', LogisticRegression(penalty='elasticnet', random_state=1234, solver='saga', max_iter=2000))]),
            'param_grid': [{ ### C is a regularization parameter that controls the trade off between the achieving a low training error and a low testing error that is the ability to generalize your classifier to unseen data
                'LR__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100], ### The inverse of regularization strength for LogisticRegression. Smaller values specify stronger regularization.
                'LR__l1_ratio': [0, 0.2, 0.4, 0.6, 0.8, 1] ### allow the model to explore various combinations of L1 and L2 regularization
            }]
        },
        'decision_tree': {
            'pipe': Pipeline([('scaler', StandardScaler()), ('DT', DecisionTreeClassifier(class_weight='balanced', random_state=1234))]),
            'param_grid': [{
                'DT__criterion': ['gini', 'entropy', 'log_loss'], ### 3 similar formula used to calculate for decision tree spliting
                'DT__max_depth': list(range(5, 16, 2)) ### min, max, step
            }]
        },
        'random_forest': {
            'pipe': Pipeline([('scaler', StandardScaler()), ('RF', RandomForestClassifier(random_state=1234))]),
            'param_grid': [{
                'RF__criterion': ['gini', 'entropy'],
                'RF__max_depth': list(range(5, 16, 2)),  ### how deep is each tree
                'RF__n_estimators': list(range(100, 501, 100)) ### number of bootstrapped dataset
                }]
        },
        'xgb' : {
            'pipe' : Pipeline([('scaler', StandardScaler()), ('XGB', XGBClassifier(random_state=1234))]),
            'param_grid' : [{
                'XGB__learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.3],
                'XGB__max_depth' : list(range(5, 16, 2)),
                'XGB__n_estimators' : list(range(100, 501, 100))
        }]
    }
            }

    scores = []

    for model_name, model_params in models.items():
        gs = GridSearchCV(model_params['pipe'],
                  model_params['param_grid'],
                  scoring=['accuracy', 'roc_auc'],
                  cv=KFold(n_splits=10, shuffle=True, random_state=1234),
                  return_train_score=True,
                  refit='accuracy')

        gs.fit(X, y)
        scores.append({
            'model': model_name,
            'best_parameters': gs.best_params_,
            'score': gs.best_score_
        })

    return pd.DataFrame(scores, columns=['model', 'best_parameters', 'score'])
```

```python
best_model_train = find_best_model(X_train, y_train)
# best_model_test = find_best_model(X_test, y_test) ### not really needed but could use it to see the difference

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

print(best_model_train)
# print(best_model_test)
```

<div>
  model  \
0  logistic_regression   
1        decision_tree   
2        random_forest   
3                  xgb   

  best_parameters  \
0                                          {'LR__C': 0.1, 'LR__l1_ratio': 0.8}   
1                                {'DT__criterion': 'gini', 'DT__max_depth': 9}   
2   {'RF__criterion': 'entropy', 'RF__max_depth': 15, 'RF__n_estimators': 500}   
3  {'XGB__learning_rate': 0.1, 'XGB__max_depth': 15, 'XGB__n_estimators': 200}   

score  
0  0.720971  
1  0.722280  
2  0.772306  
3  0.772481  
</div>

These chosen model gives about 74% accuracy, not ideal or very reliable at predicting but it is obviously better than random guessing of 50% accuracy. The similar score on the train and test data, while being far away from 100% suggest that it is not a case of overfitting

## Evaluation on supervised models

```python
# CV score from random forest
from sklearn.model_selection import cross_val_score
scores = cross_val_score(RandomForestClassifier(criterion='entropy', max_depth=15, n_estimators=500, random_state=1234), X_train, y_train, cv=10)
print('10 fold CV avg Accuracy : {}'.format(scores.mean()))
```
10 fold CV avg Accuracy : 0.7757555616950178

```python
# CV score from XGB

from sklearn.model_selection import cross_val_score
scores = cross_val_score(XGBClassifier(n_estimators=200, max_depth=15, learning_rate=0.1, random_state=1234), X_train, y_train, cv=5)
print('5 fold CV avg Accuracy : {}'.format(scores.mean()))
```
5 fold CV avg Accuracy : 0.7656712279507208

Visualize decision tree structure
```python
### visualize decision trees
from sklearn.tree import plot_tree

tree_model = DecisionTreeClassifier(criterion='gini', max_depth=9, class_weight='balanced', random_state=1234)
tree_model.fit(X_train, y_train)

plt.figure(figsize=(20,10), dpi=1000)
plot_tree(tree_model, feature_names=X.columns, class_names=['0', '1'], filled=True)
plt.show()
```

Visualize random forest trees, but only a section
```python
from sklearn.tree import plot_tree

rf_model = RandomForestClassifier(criterion='entropy', max_depth=3, n_estimators=3, random_state=1234)
rf_model.fit(X, y)

plt.figure(figsize=(20,10), dpi=1000)
for i, tree_in_rf in enumerate(rf_model.estimators_):
    plt.subplot(1, len(rf_model.estimators_), i + 1)
    plot_tree(tree_in_rf, feature_names=X.columns, class_names=['0', '1'], filled=True)
    plt.title(f'Tree {i}')

plt.show()
```

### Confusion matrix for these two models
```python
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

rf = RandomForestClassifier(n_estimators=300, random_state=1234)
rf.fit(X_train, y_train)


y_pred = rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10,7))
p = sns.heatmap(cm, annot=True, cmap="Blues", fmt='g')
plt.title('Confusion matrix for Random Forest Classifier Model - Test Set')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()

report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print(report_df)
```

![CMRF](https://github.com/Pinegraphite123/Smoking-prediction/blob/main/Graphs/Confusion%20matrix%20for%20Random%20Forest%20Classifier%20Model%20-%20Test%20Set.png?raw=true)

```python
xgb = XGBClassifier(n_estimators=300, max_depth=15, learning_rate=0.05, random_state=1234)
xgb.fit(X_train, y_train)


y_pred = xgb.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10,7))
p = sns.heatmap(cm, annot=True, cmap="Blues", fmt='g')
plt.title('Confusion matrix for XGB Classifier Model - Test Set')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()
```

![CMXGB](https://github.com/Pinegraphite123/Smoking-prediction/blob/main/Graphs/Confusion%20matrix%20for%20XGB%20Classifier%20Model%20-%20Test%20Set.png?raw=true)

```python
classification_report(y_test, y_pred, output_dict=True)
```
<div>
{'0': {'precision': 0.8353708231458843,
  'recall': 0.7115584866365845,
  'f1-score': 0.7685098406747891,
  'support': 2881},
 '1': {'precision': 0.7461820403176542,
  'recall': 0.8580962416578855,
  'f1-score': 0.798235582421173,
  'support': 2847},
 'accuracy': 0.7843924581005587,
 'macro avg': {'precision': 0.7907764317317693,
  'recall': 0.784827364147235,
  'f1-score': 0.783372711547981,
  'support': 5728},
 'weighted avg': {'precision': 0.7910411330774537,
  'recall': 0.7843924581005587,
  'f1-score': 0.7832844891999209,
  'support': 5728}}
</div>
  
# Semi-supervised <a class="anchor" id="fifth-bullet"></a>
PCA -> Random forest, feeding the output of PCA onto random forest to see if it gives a better CM score
```python
from sklearn.decomposition import PCA

pca_xgb = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.9, random_state=1234)),
    ('xgb', XGBClassifier(n_estimators=200, max_depth=15, learning_rate=0.1, random_state=1234))
])

pca_xgb.fit(X_train, y_train)

print('Train Accuracy score: {}'.format(pca_xgb.score(X_train, y_train)))
print('Test Accuracy score: {}'.format(pca_xgb.score(X_test, y_test)))


### Overfitting, reason unclear, eigenvalue vector might not work well with tree models
```

Train Accuracy score: 1.0

Test Accuracy score: 0.7693925713113072

```python
from sklearn.decomposition import PCA

pca_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.9, random_state=1234)),
    ('rf', RandomForestClassifier(criterion='entropy', max_depth=15, n_estimators=500, random_state=1234))
])

pca_rf.fit(X_train, y_train)

print('Train Accuracy score: {}'.format(pca_rf.score(X_train, y_train)))
print('Test Accuracy score: {}'.format(pca_rf.score(X_test, y_test)))
```

See the transformed feature matrix from PCA
```python
from sklearn.decomposition import PCA

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Transformed Feature Matrix:\n", pd.DataFrame(X_pca))
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

principal_df = pd.DataFrame(data = X_pca, columns = ['Principal Component 1', 'Principal Component 2'])

plt.figure(figsize=(8, 6))
plt.scatter(principal_df['Principal Component 1'], principal_df['Principal Component 2'])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Dataset')
plt.grid(True)
plt.show()
```

### Delete one more correlated feature to see the if there is an improvement on score
```python
from sklearn.decomposition import PCA

train_df = train_df.drop(['relaxation'], axis=1) ### Comment or uncomment depending on steps
X = train_df.iloc[:,:-1]
y = train_df['smoking']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234, stratify=y)

pca_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.9, random_state=1234)),
    ('rf', RandomForestClassifier(criterion='entropy', max_depth=15, n_estimators=500, random_state=1234))
])

pca_rf.fit(X_train, y_train)

print('Train Accuracy score: {}'.format(pca_rf.score(X_train, y_train)))
print('Test Accuracy score: {}'.format(pca_rf.score(X_test, y_test)))
```

Train Accuracy score: 0.9686087151697823

Test Accuracy score: 0.7808131332563807


# Unsupervised <a class="anchor" id="sixth-bullet"></a>
Random Forest Feature ranking
```python
# random forest feature ranking
# import fresh dataset again

train_df = pd.read_csv('/content/train_dataset.csv')
X = train_df.iloc[:,:-1]
y = train_df['smoking']


rf = RandomForestClassifier(n_estimators=300, random_state=1234)
rf.fit(X, y)
```

```python
importances = rf.feature_importances_

sorted_indices = np.argsort(importances)[::-1]

plt.title('Feature Importance')
plt.bar(range(X.shape[1]), importances[sorted_indices], align='center')
plt.xticks(range(X.shape[1]), X.columns[sorted_indices], rotation=90)
plt.tight_layout()
plt.show()
```

![Importance](https://github.com/Pinegraphite123/Smoking-prediction/blob/main/Graphs/Feature%20importance.png?raw=true)

### See if selecting only the most important features will give a better score
```python
X = train_df.loc[:, ['hemoglobin', 'Gtp', 'height(cm)']]
y = train_df['smoking']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234, stratify=y)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(RandomForestClassifier(criterion='entropy', max_depth=15, n_estimators=500, random_state=1234), X_train, y_train, cv=5)
print('5 fold CV avg Accuracy : {}'.format(scores.mean()))

### Not an improvement
```

By only selecting the top 3 important features, the previous best performing model returns a lower score, which may suggest a case of underfitting because there probably isnt enough feature to build enough trees.

### Conclusion:
These models achieved very similar performance. Perhaps either smoking or not is not a very good indicator of smoking statues. On top of that, the casual relationship is not clear: is a high blood glucose level result in smoking, or is smoking a result of high blood glucose? Instead of yes or no, years of smoking could be a better indicator.





