# Importação das Bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as ltb
from sklearn.model_selection import KFold
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from IPython.display import display

import joblib

# Carregamento dos dados em arquivo csv
df_kme = pd.read_csv('df_kme.csv')

# Separação das variáveis preditoras da variável alvo
X = df_kme.drop(columns='cluster') 
Y = df_kme['cluster']

# Divisão do Dataset em treino e test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=18)

# Encontrar os melhores parâmetros com GridSearchCV
num_folds = 10
seed = 18
kfold = KFold(n_splits = num_folds, random_state = seed, shuffle=True)

# Criar o modelo LGBMClassifier
model = ltb.LGBMClassifier(random_state=18)

#GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.1, 0.05, 0.01],
    'max_depth': [3, 5, 7],
    'num_leaves': [31, 50, 100],
}

scoring = 'recall_weighted'
grind_ltb = GridSearchCV(model, param_grid = param_grid, cv = kfold, verbose=True, n_jobs=-1, scoring=scoring)
grind_ltb.fit(X_train, y_train)

# Avaliar o desempenho do modelo com os melhores parâmetros
best_model = grind_ltb.best_estimator_
predictions = best_model.predict(X_test)
recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
print('\n')
print("Recall com os melhores parâmetros: ", recall)
print('\n')
print('Melhores parametros:', grind_ltb.best_params_)

# Treinar o modelo final com os melhores parâmetros usando todos os dados de treinamento
ltb = best_model.fit(X_train, y_train)

# Fazer previsões usando o modelo treinado
y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

print('Confusion Matrix:')
display(pd.DataFrame(cm, columns=['0', '1', '2'], index=['0', '1', '2']))
print('\n')
print('Classification Report:')
print('\n')
display(pd.DataFrame.from_dict(classification_report(y_test, y_pred, output_dict=True)).transpose())

# Salvar o modelo
joblib.dump(ltb, 'modelo_lgbm.sav')








