from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Normalizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definir hiperparâmetros para testar
param_grid = {
    'n_estimators': [100, 200, 300],  # Número de árvores
    'max_depth': [5, 10, 15, None],  # Profundidade máxima
    'min_samples_split': [2, 5, 10],  # Mínimo de amostras para dividir um nó
    'min_samples_leaf': [1, 2, 4]  # Mínimo de amostras em cada folha
}

# Criar modelo e buscar melhores parâmetros
modelo = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(modelo, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Melhor modelo encontrado
melhor_modelo = grid_search.best_estimator_

# Fazer previsões e calcular precisão
y_pred = melhor_modelo.predict(X_test_scaled)
precisao = accuracy_score(y_test, y_pred)

print(f'Melhores parâmetros: {grid_search.best_params_}')
print(f'Precisão otimizada: {precisao:.4f}')
