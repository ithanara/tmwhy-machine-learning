#%%
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('data/dados_cerveja_nota.xlsx')
df.head()
# %%
#Criar coluna aprovado pra todo mundo com nota maior que 5
df['aprovado'] = (df['nota'] > 5).astype(int)
df
# %%
plt.plot(df['cerveja'],df['aprovado'], 'o', color='green')
plt.grid(True)
plt.title('Cerveja vs Aprovação')
plt.xlabel('cerveja')
plt.ylabel('aprovado')
# %%
#Regressão logística
from sklearn import linear_model
reg = linear_model.LogisticRegression(penalty=None, 
                                      fit_intercept=True)
reg.fit(df[['cerveja']], df['aprovado'])

reg_predict = reg.predict(df[['cerveja']].drop_duplicates())
reg_proba = reg.predict_proba(df[['cerveja']].drop_duplicates())[:,1]

#A linha é a predição da regressão logística
plt.plot(df['cerveja'],df['aprovado'], 'o', color='green')
plt.grid(True)
plt.title('Cerveja vs Aprovação')
plt.xlabel('cerveja')
plt.ylabel('aprovado')
plt.plot(df[['cerveja']].drop_duplicates(), reg_predict, color='purple')
plt.plot(df[['cerveja']].drop_duplicates(), reg_proba, color='orange')
plt.hlines(0.5, xmin=1, xmax=9, linestyles='--', colors='black')
plt.legend(['observação', "reg predict", "reg proba"])

# %%
#Arvore
from sklearn import tree

reg = linear_model.LogisticRegression(penalty=None, 
                                      fit_intercept=True)
reg.fit(df[['cerveja']], df['aprovado'])

reg_predict = reg.predict(df[['cerveja']].drop_duplicates())
reg_proba = reg.predict_proba(df[['cerveja']].drop_duplicates())[:,1]

arvore_full = tree.DecisionTreeClassifier(random_state=42)
arvore_full.fit(df[['cerveja']], df['aprovado'])
arvore_full_predict = arvore_full.predict((df[['cerveja']].drop_duplicates()))
arvore_full_proba = arvore_full.predict_proba(df[['cerveja']].drop_duplicates())[:,1]

arvore_d2 = tree.DecisionTreeClassifier(random_state=42, max_depth=2)
arvore_d2.fit(df[['cerveja']], df['aprovado'])
arvore_d2_predict = arvore_d2.predict((df[['cerveja']].drop_duplicates()))
arvore_d2_proba = arvore_d2.predict_proba(df[['cerveja']].drop_duplicates())[:,1]

plt.plot(df['cerveja'],df['aprovado'], 'o', color='green')
plt.grid(True)
plt.title('Cerveja vs Aprovação')
plt.xlabel('cerveja')
plt.ylabel('aprovado')
plt.plot(df[['cerveja']].drop_duplicates(), arvore_full_predict, color='blue')
plt.plot(df[['cerveja']].drop_duplicates(), arvore_full_proba, color='pink')
plt.plot(df[['cerveja']].drop_duplicates(), arvore_d2_predict, color='cyan')
plt.plot(df[['cerveja']].drop_duplicates(), arvore_d2_proba, color='red')
plt.hlines(0.5, xmin=1, xmax=9, linestyles='--', colors='black')
plt.legend(['observação', 
            "Arvore full predict", 
            "Arvore full proba",
            "Arvore d2 predict", 
            "Arvore d2 proba"])
# %%
#Naive Bayes
from sklearn import naive_bayes

nb = naive_bayes.GaussianNB()
nb.fit(df[['cerveja']], df['aprovado'])

nb_predict = nb.predict((df[['cerveja']].drop_duplicates()))
nb_proba = nb.predict_proba(df[['cerveja']].drop_duplicates())[:,1]

plt.plot(df['cerveja'],df['aprovado'], 'o', color='green')
plt.grid(True)
plt.title('Cerveja vs Aprovação')
plt.xlabel('cerveja')
plt.ylabel('aprovado')
plt.plot(df[['cerveja']].drop_duplicates(), nb_predict, color='brown')
plt.plot(df[['cerveja']].drop_duplicates(), nb_proba, color='orange')
plt.hlines(0.5, xmin=1, xmax=9, linestyles='--', colors='black')
plt.legend(['observação', 
            "NB predict", 
            "NB proba"])
# %%
#todos juntos
plt.plot(df['cerveja'],df['aprovado'], 'o', color='green')
plt.grid(True)
plt.title('Cerveja vs Aprovação')
plt.xlabel('cerveja')
plt.ylabel('aprovado')

plt.plot(df[['cerveja']].drop_duplicates(), reg_predict, color='purple')
plt.plot(df[['cerveja']].drop_duplicates(), reg_proba, color='yellow')

plt.plot(df[['cerveja']].drop_duplicates(), arvore_full_predict, color='blue')
plt.plot(df[['cerveja']].drop_duplicates(), arvore_full_proba, color='pink')
plt.plot(df[['cerveja']].drop_duplicates(), arvore_d2_predict, color='cyan')
plt.plot(df[['cerveja']].drop_duplicates(), arvore_d2_proba, color='red')

plt.plot(df[['cerveja']].drop_duplicates(), nb_predict, color='brown')
plt.plot(df[['cerveja']].drop_duplicates(), nb_proba, color='orange')

plt.hlines(0.5, xmin=1, xmax=9, linestyles='--', colors='black')
plt.legend(['observação', 
            "reg predict", 
            "reg proba",
            "Arvore full predict", 
            "Arvore full proba",
            "Arvore d2 predict", 
            "Arvore d2 proba", 
            "Naive Bayes predict", 
            "Naive Bayes proba"])
# %%
