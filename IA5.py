import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from xgboost import XGBRegressor
import joblib

def remover_outliers(medicoes):
    # Remoção de outliers pelo método IQR
    Q1 = np.percentile(medicoes, 25)
    Q3 = np.percentile(medicoes, 75)
    IQR = Q3 - Q1
    lim_inf = Q1 - 1.5 * IQR
    lim_sup = Q3 + 1.5 * IQR
    return medicoes[(medicoes >= lim_inf) & (medicoes <= lim_sup)]

def detectar_anomalias(medicoes):
    # Detecção de anomalias utilizando Isolation Forest
    modelo = IsolationForest(contamination=0.05, random_state=42)
    labels = modelo.fit_predict(medicoes.reshape(-1, 1))
    return medicoes[labels == 1]  # Retorna apenas os valores considerados normais

# Carregar os dados (neste exemplo, simulamos os dados; em produção, serão os 50 resultados de medição)
file_path = 'medicoes.xlsx'
sheet_name = 'Dados'
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Processamento individual de cada característica
resultados = []
grupos = df.groupby('Characteristic')
for caracteristica, grupo in grupos:
    medicoes = grupo['Value'].values
    LSL = grupo['LowerControlLimit'].iloc[0]
    USL = grupo['UpperControlLimit'].iloc[0]
    tipo = grupo['Tipo'].iloc[0].lower()
    
    # Remoção de outliers e detecção de anomalias
    medicoes_filtradas = remover_outliers(medicoes)
    medicoes_filtradas = detectar_anomalias(medicoes_filtradas)
    
    if len(medicoes_filtradas) < 2:
        continue  # Se houver poucas medições, pula a análise
    
    sigma = np.std(medicoes_filtradas, ddof=1)
    mu = np.mean(medicoes_filtradas)
    
    Cm, Cmk = np.nan, np.nan
    # Para características dimensionais, calcula Cm e Cmk
    if tipo not in ['batimento', 'simetria', 'circularidade', 'coaxialidade', 'concentricidade']:
        if sigma > 0:
            Cm = (USL - LSL) / (6 * sigma)
            Cmk = min((mu - LSL) / (3 * sigma), (USL - mu) / (3 * sigma))
    else:
        # Para características geométricas (batimento, circularidade, etc.), apenas o Cmk faz sentido
        if sigma > 0:
            Cmk = (USL - mu) / (3 * sigma) if mu > LSL else (mu - LSL) / (3 * sigma)
        Cmk = abs(Cmk) if not np.isnan(Cmk) else np.nan
    
    # Cálculo da tendência usando regressão linear
    x = np.arange(len(medicoes_filtradas))
    y = medicoes_filtradas
    coef = np.polyfit(x, y, 1)[0] if len(x) > 1 else 0
    
    # Definição da frequência de medição com base no Cmk e na tendência
    if Cmk > 1.33 and abs(coef) < 0.001:
        frequencia = '1 a cada 50 peças'
    elif 1.0 < Cmk <= 1.33 and abs(coef) < 0.002:
        frequencia = '1 a cada 20 peças'
    elif Cmk <= 1.0 or abs(coef) >= 0.002:
        frequencia = '1 a cada 10 peças'
    else:
        frequencia = '1 a cada 5 peças'
    
    resultados.append({
        'Característica': caracteristica,
        'Tipo': tipo,
        'Cm': Cm,
        'Cmk': Cmk,
        'Tendência': coef,
        'Frequência de Medição': frequencia
    })

resultados_df = pd.DataFrame(resultados)
print("### Resultados de Capabilidade e Frequência de Medição")
print(resultados_df)

# Avaliação de correlação entre características (apenas entre medições do mesmo tipo)
correlacoes = []
caracteristicas = list(resultados_df['Característica'])
for i in range(len(caracteristicas)):
    for j in range(i + 1, len(caracteristicas)):
        carac1 = caracteristicas[i]
        carac2 = caracteristicas[j]
        tipo1 = resultados_df.loc[resultados_df['Característica'] == carac1, 'Tipo'].values[0]
        tipo2 = resultados_df.loc[resultados_df['Característica'] == carac2, 'Tipo'].values[0]
        
        # Somente correlaciona se as características forem do mesmo tipo
        if tipo1 == tipo2 and tipo1 is not None:
            valores1 = df[df['Characteristic'] == carac1]['Value'].values
            valores2 = df[df['Characteristic'] == carac2]['Value'].values
            
            if len(valores1) > 1 and len(valores2) > 1:
                correlacao = np.corrcoef(valores1, valores2)[0, 1]
                observacao = ''
                # Para medições dimensionais, correlação alta pode indicar influência de variação do eixo ou desgaste de ferramenta.
                if tipo1 == 'dimensional':
                    if abs(correlacao) > 0.7:
                        observacao = 'Possível variação do eixo ou desgaste de ferramenta'
                    elif abs(correlacao) < 0.3:
                        observacao = 'Processo estável'
                # Para características de fixação (batimento, circularidade, etc.), a correlação pode não ser indicativa
                elif tipo1 in ['batimento', 'circularidade', 'simetria', 'coaxialidade', 'concentricidade']:
                    observacao = 'Medição dependente de fixação; correl. pode não ser indicativa'
                
                correlacoes.append({
                    'Pares de Características': f"{carac1} - {carac2}",
                    'Tipo': tipo1,
                    'Correlação': correlacao,
                    'Observação': observacao
                })

correlacoes_df = pd.DataFrame(correlacoes)
print("\n### Correlações Entre Características")
print(correlacoes_df)

# Treinamento de modelo preditivo (usando XGBoost) para estimar a próxima medição
previsoes = []
for caracteristica, grupo in grupos:
    medicoes = grupo['Value'].values
    # Para treinamento, utiliza-se somente se houver pelo menos 10 medições (caso dos 50 resultados, isso é atendido)
    if len(medicoes) < 10:
        continue
    
    X = np.arange(len(medicoes)).reshape(-1, 1)
    y = medicoes
    
    modelo = XGBRegressor(objective='reg:squarederror', n_estimators=50)
    modelo.fit(X, y)
    
    joblib.dump(modelo, f'modelo_{caracteristica}.pkl')  # Salva o modelo para uso futuro
    previsao = modelo.predict(np.array([[len(medicoes) + 1]]))[0]
    previsoes.append({'Característica': caracteristica, 'Previsão': previsao})

previsoes_df = pd.DataFrame(previsoes)
print("\n### Previsão para a Próxima Peça (Modelo Preditivo)")
print(previsoes_df)
