import numpy as np
import pandas as pd

# Carregar medições de um arquivo Excel
file_path = 'medicoes.xlsx'  # Nome do arquivo Excel
sheet_name = 'Dados'  # Nome da aba

df = pd.read_excel(file_path, sheet_name=sheet_name)

# Criar um dicionário para armazenar os resultados por característica
resultados = {}

def identificar_eixo(caracteristica):
    """Identifica se a característica pertence ao eixo X (diâmetro) ou Z (comprimento)."""
    if 'Diâmetro' in caracteristica:
        return 'X'
    elif 'Comprimento' in caracteristica or 'Distância' in caracteristica:
        return 'Z'
    return None

# Processar cada característica individualmente
grupos = df.groupby('Characteristic')
for caracteristica, grupo in grupos:
    medicoes = grupo['Value'].values
    LSL = grupo['LowerControlLimit'].iloc[0]  # Limite inferior
    USL = grupo['UpperControlLimit'].iloc[0]  # Limite superior

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

    # Identificar eixo da característica
    eixo = identificar_eixo(caracteristica)
    
    # Armazenar os resultados
    resultados[caracteristica] = {'Cm': Cm, 'Cmk': Cmk, 'Tendência': coef, 'Frequência de Medição': frequencia, 'Eixo': eixo}

# Avaliação da correlação entre os diâmetros para detectar padrões de variação
caracteristicas = list(resultados.keys())
correlacoes = {}
for i in range(len(caracteristicas)):
    for j in range(i + 1, len(caracteristicas)):
        carac1 = caracteristicas[i]
        carac2 = caracteristicas[j]
        eixo1 = resultados[carac1]['Eixo']
        eixo2 = resultados[carac2]['Eixo']
        
        # Só comparar características do mesmo eixo
        if eixo1 == eixo2 and eixo1 is not None:
            tendencia1 = resultados[carac1]['Tendência']
            tendencia2 = resultados[carac2]['Tendência']
            correlacao = np.corrcoef(df[df['Characteristic'] == carac1]['Value'], df[df['Characteristic'] == carac2]['Value'])[0, 1]
            correlacoes[f"{carac1} - {carac2} ({eixo1})"] = correlacao

# Exibir resultados
for caract, res in resultados.items():
    print(f"Característica: {caract} (Eixo {res['Eixo']})")
    print(f"Cm: {res['Cm']:.2f}, Cmk: {res['Cmk']:.2f}, Tendência: {res['Tendência']:.6f}, Frequência de Medição: {res['Frequência de Medição']}")
    print("-")

# Exibir análise de correlação entre características do mesmo eixo
print("Análise de Correlação entre Características do Mesmo Eixo:")
for pares, corr in correlacoes.items():
    print(f"{pares}: Correlação = {corr:.4f}")
    if abs(corr) > 0.7:
        print("Possível variação do eixo da máquina identificada.")
    elif abs(corr) < 0.3:
        print("Possível desgaste de ferramenta identificado.")
    print("-")