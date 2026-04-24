#%%
import pandas as pd

pd.options.display.max_columns = 500
pd.options.display.max_rows = 500

df= pd.read_csv("../data/abt_churn.csv", sep=",")
df.head()
# %%
#OOT = out of time
oot = df[df["dtRef"] == df["dtRef"].max()].copy()
oot
# %%
df_train = df[df["dtRef"] < df["dtRef"].max()].copy()
df_train.shape
# %%
features = df_train.columns[2:-1]
target = 'flagChurn'

X, y = df_train[features], df_train[target]
# %%
#Sample
from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
                                                                    random_state=42,
                                                                    test_size=0.2,
                                                                    stratify=y
                                                                    )
# %%
print("taxa variável resposta:", y_train.mean())
print("taxa variável resposta:", y_test.mean())
# %%
#Aqui começa o Explore (Análise exploratória dos dados)
X_train.isna().sum()
# %%
#Analise bivariada
df_analise = X_train.copy()
df_analise[target]= y_train

sumario = df_analise.groupby(by=target).agg(['mean','median']).T
sumario
# %%
#Diferença absoluta
sumario['diff_abs'] = sumario[0] - sumario[1]
sumario['diff_rel'] = sumario[0] / sumario[1]
sumario.sort_values(by=['diff_rel'], ascending=True)

# %%
from sklearn import tree
import matplotlib.pyplot as plt

arvore = tree.DecisionTreeClassifier(random_state= 42, 
                                     max_depth=5)
arvore.fit(X_train, y_train)

plt.figure(dpi=300)
tree.plot_tree(arvore,
               feature_names=X_train.columns,
               filled=True,
               class_names=[str(i) for i in arvore.classes_]
               )
# %%
#Arvore completa
arvore = tree.DecisionTreeClassifier(random_state= 42)
arvore.fit(X_train, y_train)

feature_importances = (pd.Series(arvore.feature_importances_, 
                                index=X_train.columns)
                                .sort_values(ascending=False)
                                .reset_index()
                                )
feature_importances['acum.'] = feature_importances[0].cumsum()
#feature_importances[feature_importances[0] > 0.01]
feature_importances[feature_importances['acum.'] < 0.96]

# %%
best_features = (feature_importances[feature_importances['acum.'] < 0.96]['index']
                 .tolist())
best_features
# %%
#Modify (aqui foi uma versão simples do modify)
from feature_engine import discretisation

tree_discretisation = discretisation.DecisionTreeDiscretiser(
    variables=best_features,
    regression=False,
    bin_output='bin_number',
    cv=3
    )

tree_discretisation.fit(X_train[best_features], y_train)
# %%
#aplicar o que foi aprendido
X_train_transform = tree_discretisation.transform(X_train[best_features])
X_train_transform
# %%
#Model
from sklearn import linear_model

reg = linear_model.LogisticRegression(
    penalty=None,
    random_state=42,
    max_iter=1000000
)
reg.fit(X_train_transform, y_train)
# %%
from sklearn import metrics

y_train_predict = reg.predict(X_train_transform)
y_train_proba = reg.predict_proba(X_train_transform)[:,1]

acc_train = metrics.accuracy_score(y_train, y_train_predict)
auc_train = metrics.roc_auc_score(y_train, y_train_proba)

print("Acuracia treino:", acc_train)
print("AUC treino:", auc_train)

#Verificar dados do test
X_test_transform = tree_discretisation.transform(X_test[best_features])

y_test_predict = reg.predict(X_test_transform)
y_test_proba = reg.predict_proba(X_test_transform)[:,1]

acc_test = metrics.accuracy_score(y_test, y_test_predict)
auc_test = metrics.roc_auc_score(y_test, y_test_proba)

print("Acuracia teste:", acc_test)
print("AUC teste:", auc_test)
#Ver como fica na OOT

oot_transform = tree_discretisation.transform(oot[best_features])

y_oot_predict = reg.predict(oot_transform)
y_oot_proba = reg.predict_proba(oot_transform)[:,1]

acc_oot = metrics.accuracy_score(oot[target], y_oot_predict)
auc_oot = metrics.roc_auc_score(oot[target], y_oot_proba)

print("Acuracia oot:", acc_oot)
print("AUC oot:", auc_oot)
# %%
