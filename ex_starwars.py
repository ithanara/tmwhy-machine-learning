#%%
import pandas as pd

df = pd.read_parquet("data/dados_clones.parquet")
df.head()
# %%
features = [ 'Massa(em kilos)',	'Estatura(cm)','General Jedi encarregado' ]
target = 'Status '

X = df[features]
y = df[target]
#%%
X = X.replace({
    "Yoda":0,
    "Shaak Ti":1,
    "Obi-Wan Kenobi":2, 
    "Aayla Secura":3,
    "Mace Windu":4
})
X
# %%
from sklearn import tree

model = tree.DecisionTreeClassifier()
model.fit(X=X, y=y)
# %%
import matplotlib.pyplot as plt

plt.figure(dpi=400)
tree.plot_tree(model,feature_names=features, 
               class_names=model.classes_,
               filled=True)
# %%
