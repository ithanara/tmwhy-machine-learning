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
from feature_engine import discretisation, encoding
from sklearn import pipeline

#Discretização
tree_discretisation = discretisation.DecisionTreeDiscretiser(
    variables=best_features,
    regression=False,
    bin_output='bin_number',
    cv=3
    )

tree_discretisation.fit(X_train[best_features], y_train)

#One hot encoding
onehot = encoding.OneHotEncoder(variables=best_features, ignore_format=True)

# X_train_transform = onehot.transform(X_train_transform)
# X_train_transform
# %%
#Model
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import ensemble

model = linear_model.LogisticRegression(penalty=None,random_state=42,max_iter=1000000)
#model = naive_bayes.BernoulliNB()
#model = ensemble.RandomForestClassifier(random_state=42,min_samples_leaf=20,n_jobs=-1,n_estimators=1000)
#model = tree.DecisionTreeClassifier(random_state=42, min_samples_leaf=20)
#model = ensemble.AdaBoostClassifier(random_state=42,n_estimators=500,learning_rate=0.01)

model_pipeline = pipeline.Pipeline(
    steps=[
        ('Discretizar', tree_discretisation),
        ('One Hot', onehot),
        ('Model', model)
    ]
)
from sklearn import metrics
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment(experiment_id=1)

with mlflow.start_run(run_name=model.__str__()):
    mlflow.sklearn.autolog()
    model_pipeline.fit(X_train, y_train)

    y_train_predict = model_pipeline.predict(X_train)
    y_train_proba = model_pipeline.predict_proba(X_train)[:,1]

    acc_train = metrics.accuracy_score(y_train, y_train_predict)
    auc_train = metrics.roc_auc_score(y_train, y_train_proba)
    roc_train = metrics.roc_curve(y_train, y_train_proba)
    print("Acuracia treino:", acc_train)
    print("AUC treino:", auc_train)

    #Verificar dados do test
    y_test_predict = model_pipeline.predict(X_test)
    y_test_proba = model_pipeline.predict_proba(X_test)[:,1]

    acc_test = metrics.accuracy_score(y_test, y_test_predict)
    auc_test = metrics.roc_auc_score(y_test, y_test_proba)
    roc_test = metrics.roc_curve(y_test, y_test_proba)
    print("Acuracia teste:", acc_test)
    print("AUC teste:", auc_test)

    #Ver como fica na OOT
    y_oot_predict = model_pipeline.predict(oot[features])
    y_oot_proba = model_pipeline.predict_proba(oot[features])[:,1]

    acc_oot = metrics.accuracy_score(oot[target], y_oot_predict)
    auc_oot = metrics.roc_auc_score(oot[target], y_oot_proba)
    roc_oot = metrics.roc_curve(oot[target], y_oot_proba)
    print("Acuracia oot:", acc_oot)
    print("AUC oot:", auc_oot)

    mlflow.log_metrics({
        "auc_train" : auc_train,
        "acc_test" : acc_test,
        "auc_test" : auc_test,
        "acc_oot" : acc_oot,
        "auc_oot" : auc_oot,
    })
# %%
#Plotar curva roc
plt.plot(roc_train[0], roc_train[1])
plt.plot(roc_test[0], roc_test[1])
plt.plot(roc_oot[0], roc_oot[1])
plt.grid(True)
plt.ylabel("Sensibilidade")
plt.xlabel("1 - Especificidade")
plt.title("Curva Roc")
plt.legend([
    f"Treino: {100*auc_train: .2f}",
    f"Teste: {100*auc_test: .2f}",
    f"OOT: {100*auc_oot: .2f}",
])
# %%
