#%%
import pandas as pd

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
