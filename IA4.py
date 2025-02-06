import numpy as np
import pandas as pd

def remover_outliers(medicoes):
    Q1 = np.percentile(medicoes, 25)
    Q3 = np.percentile(medicoes, 75)
    IQR = Q3 - Q1
    lim_inf = Q1 - 1.5 * IQR
    lim_sup = Q3 + 1.5 * IQR
    return medicoes[(medicoes >= lim_inf) & (medicoes <= lim_sup)]

file_path = 'medicoes.xlsx'  # Nome do arquivo Excel
sheet_name = 'Dados'  # Nome da aba
df = pd.read_excel(file_path, sheet_name=sheet_name)

resultados = []
grupos = df.groupby('Characteristic')
for caracteristica, grupo in grupos:
    medicoes = grupo['Value'].values
    LSL = grupo['LowerControlLimit'].iloc[0]
    USL = grupo['UpperControlLimit'].iloc[0]
    tipo = grupo['Tipo'].iloc[0].lower()
    
    # Remover outliers antes do cálculo
    medicoes_filtradas = remover_outliers(medicoes)
    
    if len(medicoes_filtradas) < 2:
        continue  # Se restar poucos dados, pula a característica
    
    sigma = np.std(medicoes_filtradas, ddof=1)
    mu = np.mean(medicoes_filtradas)
    
    Cm, Cmk = np.nan, np.nan
    
    if tipo not in ['batimento', 'simetria', 'circularidade', 'coaxialidade', 'concentricidade']:
        if sigma > 0:
            Cm = (USL - LSL) / (6 * sigma)
            Cmk = min((mu - LSL) / (3 * sigma), (USL - mu) / (3 * sigma))
    else:
        if sigma > 0:
            Cmk = (USL - mu) / (3 * sigma) if mu > LSL else (mu - LSL) / (3 * sigma)
        Cmk = abs(Cmk) if Cmk is not np.nan else np.nan
    
    # Avaliação de tendência (coeficiente de regressão linear)
    x = np.arange(len(medicoes_filtradas))
    y = medicoes_filtradas
    coef = np.polyfit(x, y, 1)[0] if len(x) > 1 else 0
    
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
print(resultados_df)

# Avaliação da correlação entre características para detectar padrões de variação
correlacoes = []
caracteristicas = list(resultados_df['Característica'])
for i in range(len(caracteristicas)):
    for j in range(i + 1, len(caracteristicas)):
        carac1 = caracteristicas[i]
        carac2 = caracteristicas[j]
        tipo1 = resultados_df.loc[resultados_df['Característica'] == carac1, 'Tipo'].values[0]
        tipo2 = resultados_df.loc[resultados_df['Característica'] == carac2, 'Tipo'].values[0]
        
        # Comparar apenas características do mesmo tipo
        if tipo1 == tipo2 and tipo1 is not None:
            valores1 = df[df['Characteristic'] == carac1]['Value'].values
            valores2 = df[df['Characteristic'] == carac2]['Value'].values
            
            if len(valores1) > 1 and len(valores2) > 1:
                correlacao = np.corrcoef(valores1, valores2)[0, 1]
                observacao = ''
                if abs(correlacao) > 0.7:
                    observacao = 'Possível variação do eixo da máquina'
                elif abs(correlacao) < 0.3 and tipo1 not in ['batimento', 'circularidade', 'coaxialidade']:
                    observacao = 'Possível desgaste de ferramenta'
                
                correlacoes.append({
                    'Pares de Características': f"{carac1} - {carac2}",
                    'Tipo': tipo1,
                    'Correlação': correlacao,
                    'Observação': observacao
                })

correlacoes_df = pd.DataFrame(correlacoes)
print(correlacoes_df)
