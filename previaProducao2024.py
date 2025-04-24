#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 11:52:43 2025

@author: costa
"""
import locale
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import pymysql

# tradução para o português brasil
locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')

# Função para calcular o RMSE
def calcular_rmse(y_real, y_previsto):
    return np.sqrt(mean_squared_error(y_real, y_previsto))

# Função para calcular o MAE
def calcular_mae(y_real, y_previsto):
    return mean_absolute_error(y_real, y_previsto)

# Função para calcular o MAPE
def calcular_mape(y_real, y_previsto):
    return mean_absolute_percentage_error(y_real, y_previsto) * 100  # Em percentual

# Conectando ao banco de dados MySQL
db_name = "db_demo"
db_host = "localhost"
db_user = "grauzy_user"
db_pass = "@K41sal1s"

try:
    conn = pymysql.connect(
        host=db_host,
        port=int(3306),
        user=db_user,
        password=db_pass,
        db=db_name
    )
    print("Conexão realizada com sucesso")
except Exception as e:
    print(f"Erro na conexão: {e}")
    exit(1)  # Encerra o script com código de erro

# Verificar se a conexão foi estabelecida
if 'conn' not in locals():
    print("Conexão com o banco de dados não foi estabelecida.")
    exit(1)

# Carregando os dados da tabela tb_lancamentos_prod_recebida
query = """
    SELECT 
        DT_AVISO_PROD, 
        TIPO_AVISO_PROD,  
        VL_AVISO_PROD 
    FROM tb_lancamentos_prod_recebida
    WHERE FL_TIPO_REDE = 'C'
"""
df = pd.read_sql(query, conn)

# Fechar a conexão
conn.close()

# Converter DT_AVISO_PROD para datetime
df['DT_AVISO_PROD'] = pd.to_datetime(df['DT_AVISO_PROD'])

# Extraindo o mês e o ano
df['MES'] = df['DT_AVISO_PROD'].dt.month
df['ANO'] = df['DT_AVISO_PROD'].dt.year

# Filtrando os dados de 2021 a 2023 da produção recebida para treinamento
df_treino = df[(df['ANO'] >= 2021) & (df['ANO'] <= 2023)]

# Agrupando dados por TIPO_AVISO_PROD
grupos = df_treino.groupby(['TIPO_AVISO_PROD'])

# Dicionário para armazenar previsões
previsoes = {}

# Linha do gráfico
# Definindo paleta de cores para os anos
cores_anos = {
    2021: '#fbcc8a',
    2022: '#df8492',
    2023: '#c660c5',
}

# Cor da previsão
cor_previsao = '#00ff0b'

# Loop para iterar sobre cada grupo
for (tipo_aviso), grupo in grupos:
    # Criando série temporal
    serie_temporal = grupo.set_index('DT_AVISO_PROD')['VL_AVISO_PROD'].asfreq('MS')

    # Verificar se a série temporal é válida
    if serie_temporal.isnull().any() or len(serie_temporal) < 12:
        print(f"Atenção: Série temporal inválida para {tipo_aviso}")
        continue

    # Preenchendo valores nulos, se necessário
    serie_temporal = serie_temporal.ffill()

    # Treinando o modelo SARIMA com SARIMAX
    # (p, d, q) P = ordem do componente AR (auto-regressão sazonal), d = número de diferenciações necessárias para
    #               tornar a série estacionária, q = ordem do componente MA (média móvel sazonal)
    #
    # (P, D, Q, S) P = Ordem sazonal AR (auto-regressão sazonal), D = Diferenciação sazonal,
    #                  Q = Ordem sazonal MA (média móvel sazonal), s = Período sazonal
    #
    # Padrão
    # (1,1,1) para a parte não sazonal
    # (1, 1, 1, 12) para a parte sazonal

    # Ordens não sazonais  Ordens sazonais:
    # (0,1,1)              (0,1,1,12)
    # (1,1,0)              (1,0,1,12)
    # (2,1,2)              (1,1,0,12)
    # (1,1,2)              (2,1,1,12)
    try:
        modelo_sarima = SARIMAX( serie_temporal, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        resultado = modelo_sarima.fit(disp=False)

        # Prevendo os próximos 12 meses (ano de 2024) com intervalo de confiança
        previsao = resultado.get_forecast(steps=12)
        previsao_media = previsao.predicted_mean

        # Armazenando as previsões
        previsoes[(tipo_aviso)] = previsao_media

        # Calculando as métricas de avaliação (usando últimos 12 meses de 2023 para validação)
        y_real = serie_temporal[-12:]  # Últimos 12 meses de 2023
        y_previsto = resultado.predict(start=serie_temporal.index[-12], end=serie_temporal.index[-1])

        # Calculando métricas
        rmse = calcular_rmse(y_real, y_previsto)
        mae = calcular_mae(y_real, y_previsto)
        mape = calcular_mape(y_real, y_previsto)

        # Exibindo as métricas
        print(f"Grupo: TIPO = {tipo_aviso}")
        print(f"  - RMSE (Erro Quadrático Médio Raiz): {rmse:.2f}")
        print(f"  - MAE (Erro Absoluto Médio): {mae:.2f}")
        print(f"  - MAPE (Erro Percentual Absoluto Médio): {mape:.2f}%")
        print("\n")

        # Exibindo previsões para todos os meses de 2024
        datas_previsao = pd.date_range(start='2024-01-01', periods=12, freq='MS')
        for data, valor in zip(datas_previsao, previsao_media):
            print(f"  - Previsão para {data.strftime('%B %Y')}: R$ {valor:.2f}")
        print("\n")

        # Preparando dados para o gráfico
        dados_grafico = serie_temporal.reset_index()
        dados_grafico['MES'] = dados_grafico['DT_AVISO_PROD'].dt.month
        dados_grafico['ANO'] = dados_grafico['DT_AVISO_PROD'].dt.year

        # Plotar gráfico da série temporal e previsões
        plt.figure(figsize=(12, 6))

        # Plotar cada ano em uma linha separada
        for ano in dados_grafico['ANO'].unique():
            dados_ano = dados_grafico[dados_grafico['ANO'] == ano]
            cor = cores_anos[ano]  # Cor do ano
            plt.plot(dados_ano['MES'], dados_ano['VL_AVISO_PROD'], label=f'Dados {ano}', marker='o', color=cor)

            # Adicionar rótulos aos pontos da série temporal
            for mes, valor in zip(dados_ano['MES'], dados_ano['VL_AVISO_PROD']):
                plt.text(mes, valor, f'{valor:.2f}', fontsize=9, ha='center', va='bottom')

        # Plotando previsões para 2024
        plt.plot(range(1, 13), previsao_media, label='Previsão 2024', marker='o', color=cor_previsao)

        # Adicionar rótulos aos pontos da previsão
        for mes, valor in zip(range(1, 13), previsao_media):
            plt.text(mes, valor, f'{valor:.2f}', fontsize=9, ha='center', va='bottom')

        # Configurando eixo X para exibir apenas os meses
        plt.xticks(range(1, 13), ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dec'])
        plt.title(f'Previsão SARIMA - {tipo_aviso}')
        plt.xlabel('Mês')
        plt.ylabel('Valor da Produção Recebida')
        plt.legend()
        plt.grid()
        plt.show()

    except Exception as e:
        print(f"Erro ao modelar {tipo_aviso} : {e}")