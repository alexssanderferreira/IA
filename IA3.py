import numpy as np
import pandas as pd

# Carregar medições de um arquivo Excel
file_path = 'medicoes.xlsx'  # Nome do arquivo Excel
sheet_name = 'Dados'  # Nome da aba

df = pd.read_excel(file_path, sheet_name=sheet_name)

# Criar um dicionário para armazenar os resultados por característica
resultados = []

# Processar cada característica individualmente
grupos = df.groupby('Characteristic')
for caracteristica, grupo in grupos:
    medicoes = grupo['Value'].values
    LSL = grupo['LowerControlLimit'].iloc[0]  # Limite inferior
    USL = grupo['UpperControlLimit'].iloc[0]  # Limite superior
    tipo = grupo['Tipo'].iloc[0]  # Identificar tipo da característica (X, Z ou Batimento)
    
    # Cálculo de Cm e Cmk
    sigma = np.std(medicoes, ddof=1)
    mu = np.mean(medicoes)
    Cm = (USL - LSL) / (6 * sigma)
    Cmk = min((mu - LSL) / (3 * sigma), (USL - mu) / (3 * sigma))

    # Avaliação de tendência (coeficiente de inclinação da regressão linear)
    x = np.arange(len(medicoes))
    y = medicoes
    coef = np.polyfit(x, y, 1)[0]  # Inclinação da reta de tendência
    
    # Definição da frequência de medição considerando Cm, Cmk e tendências
    if Cmk > 1.33 and abs(coef) < 0.001:
        frequencia = '1 a cada 50 peças'
    elif 1.0 < Cmk <= 1.33 and abs(coef) < 0.002:
        frequencia = '1 a cada 20 peças'
    elif Cmk <= 1.0 or abs(coef) >= 0.002:
        frequencia = '1 a cada 10 peças'
    else:
        frequencia = '1 a cada 5 peças'

    # Armazenar os resultados
    resultados.append({
        'Característica': caracteristica,
        'Tipo': tipo,
        'Cm': Cm,
        'Cmk': Cmk,
        'Tendência': coef,
        'Frequência de Medição': frequencia
    })

# Criar DataFrame com os resultados
resultados_df = pd.DataFrame(resultados)

# Exibir a tabela de resultados
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
        
        # Só comparar características do mesmo tipo
        if tipo1 == tipo2 and tipo1 is not None:
            correlacao = np.corrcoef(df[df['Characteristic'] == carac1]['Value'], df[df['Characteristic'] == carac2]['Value'])[0, 1]
            correlacoes.append({
                'Pares de Características': f"{carac1} - {carac2}",
                'Tipo': tipo1,
                'Correlação': correlacao,
                'Observação': 'Possível variação do eixo da máquina' if abs(correlacao) > 0.7 else 'Possível desgaste de ferramenta' if abs(correlacao) < 0.3 else ''
            })

# Criar DataFrame com as correlações
correlacoes_df = pd.DataFrame(correlacoes)

# Exibir a tabela de correlações
print(correlacoes_df)
