#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 09:27:55 2025

@author: costa
"""
import locale
import pandas as pd
import pymysql
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX


# tradução para o português brasil
locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')

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

# SQL
query = """
        SELECT
            NU_CGC_CPF,
        	SUBSTR(NM_PESSOA_RAZAO_SOCIAL,1,25) AS NM_PESSOA_RAZAO_SOCIAL,
            DATE(DT_PRODUCAO) AS DT_PRODUCAO,
            DATE(DT_ENTREGA) AS DT_ENTREGA,
            DATE(DATA_REALIZACAO) AS DATA_REALIZACAO,
            CD_TIPO_SERVICO_SAUDE,
            CD_TIPO_TRATAMENTO,
            CASE
                WHEN CD_TIPO_TRATAMENTO = 1 THEN 'CLINICO'
                WHEN CD_TIPO_TRATAMENTO = 2 THEN 'CIRURGICO'
                WHEN CD_TIPO_TRATAMENTO = 3 THEN 'OBSTETRICO'
                WHEN CD_TIPO_TRATAMENTO = 4 THEN 'PEDIATRICA'
                WHEN CD_TIPO_TRATAMENTO = 5 THEN 'PSIQUIATRICA'
                WHEN CD_TIPO_TRATAMENTO = 6 THEN ' '
                WHEN CD_TIPO_TRATAMENTO = 7 THEN 'ODONTOLOGICA'
            END  as DS_TIPO_TRATAMENTO,
            CD_SERVICO,
            CD_SENHA_PRINCIPAL,
            CD_SENHA_AUTORIZACAO,
            PROCESSO,
            GUIA,
            CD_BENEFICIARIO,
            DS_SITUACAO_BENEFICIARIO,
            CD_TAB,
            CD_PROCEDIMENTO,
        	SUBSTR(NM_PROCEDIMENTO,1,25) AS NM_PROCEDIMENTO,
            QT_PROCEDIMENTO,
            VL_PROCEDIMENTO,
            QT_PROCEDIMENTO_P,
            VL_PROCEDIMENTO_P,
            CD_TIPO_ATENDIMENTO_TISS,
            DS_TIPO_ATENDIMENTO_TISS,
            DS_TIPO_EVENTO
        FROM
            db_demo.tb_demost_analise_conta AS ac
        WHERE
            ac.CD_PROCEDIMENTO IN (
                SELECT
                    x.CD_PROCEDIMENTO
                FROM
                    tb_procedimentos_onco x
            )
            AND ac.VL_PROCEDIMENTO_P <> 0;
"""

# Carregar os dados em um DataFrame
df = pd.read_sql(query, conn)

# Fechar a conexão
conn.close()

print("Iniciando a analise dos dados")
print("\n")

# Inicio -- Beneficiários
# Agrupar por beneficiário e contar procedimentos
utilizacao_beneficiario = df['CD_BENEFICIARIO'].value_counts().reset_index()
utilizacao_beneficiario.columns = ['CD_BENEFICIARIO', 'QUANTIDADE_PROCEDIMENTOS']

# Histograma - Distribuição de utilização por beneficiários (Procedimento Oncológicos)
plt.figure(figsize=(10, 6))
ax = sns.histplot(utilizacao_beneficiario['QUANTIDADE_PROCEDIMENTOS'], bins=30, kde=True)
plt.title('Histograma - Distribuição de Utilização por Beneficiário (Procedimentos Oncológicos)')
plt.xlabel('Quantidade de Procedimentos')
plt.ylabel('Frequência')

# Adicionar rótulos de dados para as barras do histograma
for p in ax.patches:
    if p.get_height() > 0:  # Só adiciona rótulo se a barra tiver altura > 0
        ax.annotate(f'{int(p.get_height())}',
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center',
                   xytext=(0, 5),
                   textcoords='offset points',
                   fontsize=8)

plt.show()
print("\n")

# Ordenar por quantidade de procedimentos (do maior para o menor)
top_beneficiarios = utilizacao_beneficiario.sort_values(by='QUANTIDADE_PROCEDIMENTOS', ascending=False)

# Exibir os top 10 beneficiários
print("Top 10 Beneficiários com Mais Procedimentos:")
print(top_beneficiarios.head(10))

# Top 10 Beneficiários com mais procedimentos oncológicos
plt.figure(figsize=(12, 6))
ax = sns.barplot(x='CD_BENEFICIARIO', y='QUANTIDADE_PROCEDIMENTOS',
                 data=top_beneficiarios.head(10), palette='viridis')
plt.title('Top 10 Beneficiários com Mais Procedimentos Oncológicos')
plt.xlabel('Beneficiário')
plt.ylabel('Quantidade de Procedimentos')
plt.xticks(rotation=45)

# Adicionar rótulos de dados
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 5),
                textcoords='offset points',
                fontsize=8)

plt.show()
print("\n")
# Fim -- Beneficiários

# Inicio -- Prestadores
# Agrupar por prestador e contar procedimentos
utilizacao_prestador = df.groupby(['NU_CGC_CPF', 'NM_PESSOA_RAZAO_SOCIAL']).size().reset_index()
utilizacao_prestador.columns = ['NU_CGC_CPF', 'NM_PESSOA_RAZAO_SOCIAL', 'QUANTIDADE_PROCEDIMENTOS']

# Ordenar por quantidade de procedimentos
utilizacao_prestador = utilizacao_prestador.sort_values(by='QUANTIDADE_PROCEDIMENTOS', ascending=False)

# Visualizar os top 10 prestadores com mais procedimentos
print(utilizacao_prestador.head(10))

# Top 10 prestadores por quantidade de procedimentos oncológicos
plt.figure(figsize=(12, 6))
ax = sns.barplot(x='NM_PESSOA_RAZAO_SOCIAL', y='QUANTIDADE_PROCEDIMENTOS',
                 data=utilizacao_prestador.head(10))
plt.title('Top 10 Prestadores por Quantidade de Procedimentos Oncológicos')
plt.xlabel('Prestador')
plt.ylabel('Quantidade de Procedimentos')
plt.xticks(rotation=45)

# Adicionar rótulos de dados
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 5),
                textcoords='offset points',
                fontsize=8)

plt.show()
print("\n")

# fim -- Prestadores

# Inicio -- Procedimentos
# Converter a coluna de data para o tipo datetime
df['DT_PRODUCAO'] = pd.to_datetime(df['DT_PRODUCAO'])

# Extrair o mês e o ano
df['MES'] = df['DT_PRODUCAO'].dt.month
df['ANO'] = df['DT_PRODUCAO'].dt.year

# Agrupar por mês e ano e contar procedimentos
sazonalidade = df.groupby(['ANO', 'MES']).size().reset_index()
sazonalidade.columns = ['ANO', 'MES', 'QUANTIDADE_PROCEDIMENTOS']

# Gráfico de sazonalidade
plt.figure(figsize=(12, 6))
ax = sns.lineplot(x='MES', y='QUANTIDADE_PROCEDIMENTOS', hue='ANO', data=sazonalidade, marker='o')

# Adicionar rótulos de dados
for line in range(len(sazonalidade)):
    ax.text(sazonalidade.MES[line], sazonalidade.QUANTIDADE_PROCEDIMENTOS[line],
            sazonalidade.QUANTIDADE_PROCEDIMENTOS[line],
            horizontalalignment='center', size=8, color='black')

# Configurações do gráfico
plt.title('Sazonalidade de Procedimentos Oncológicos', fontsize=14)
plt.xlabel('Mês', fontsize=9)
plt.ylabel('Quantidade de Procedimentos', fontsize=9)
plt.xticks(range(1, 13), ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'],
           fontsize=9)
plt.yticks(fontsize=9)
plt.legend(title='Ano', fontsize=9)
plt.grid(True)
plt.show()
print("\n")

# Calcular o valor total pago por procedimento
df['VL_TOTAL_PAGO'] = df['QT_PROCEDIMENTO_P'] * df['VL_PROCEDIMENTO_P']

# Agrupar por procedimento e somar os valores pagos
valores_pagos = df.groupby(['CD_PROCEDIMENTO', 'NM_PROCEDIMENTO'])['VL_TOTAL_PAGO'].sum().reset_index()

# Ordenar por valor total pago
valores_pagos = valores_pagos.sort_values(by='VL_TOTAL_PAGO', ascending=False)

# Visualizar os top 10 procedimentos com maior valor pago
print("Top 10 Procedimentos com Maior Valor Pago:")
print(valores_pagos.head(10))

# Plotar os top 10 procedimentos
plt.figure(figsize=(12, 6))
ax = sns.barplot(x='NM_PROCEDIMENTO', y='VL_TOTAL_PAGO', data=valores_pagos.head(10), palette='viridis')

# Adicionar rótulos de dados
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 9),
                textcoords='offset points',
                fontsize=8)

# Configurações do gráfico
plt.title('Top 10 Procedimentos Oncológicos com Maior Valor Pago', fontsize=14)
plt.xlabel('Procedimento', fontsize=9)
plt.ylabel('Valor Total Pago (R$)', fontsize=9)
plt.xticks(rotation=45, fontsize=9)
plt.yticks(fontsize=9)
plt.grid(True)
plt.show()
print("\n")

# procedimentos mês a mês

# Agrupar por procedimento, ano e mês, e somar os valores pagos
valores_pagos_mensais = df.groupby(
    ['CD_PROCEDIMENTO', 'NM_PROCEDIMENTO', 'ANO', 'MES']
)['VL_TOTAL_PAGO'].sum().reset_index()

# Definir uma paleta de cores mais vibrante
cores = sns.color_palette("dark", n_colors=len(valores_pagos_mensais['NM_PROCEDIMENTO'].unique()))

# Criar um gráfico de linha para cada procedimento
plt.figure(figsize=(14, 8))
for idx, procedimento in enumerate(valores_pagos_mensais['NM_PROCEDIMENTO'].unique()):
    dados_procedimento = valores_pagos_mensais[valores_pagos_mensais['NM_PROCEDIMENTO'] == procedimento]
    linha = plt.plot(dados_procedimento['MES'], dados_procedimento['VL_TOTAL_PAGO'],
                     label=procedimento, marker='o', color=cores[idx], linewidth=2)

    # Adicionar rótulos de dados
    for x, y in zip(dados_procedimento['MES'], dados_procedimento['VL_TOTAL_PAGO']):
        plt.text(x, y, f'{y:.2f}', color=linha[0].get_color(), fontsize=8, ha='center', va='bottom')

# Configurações do gráfico
plt.title('Valores Pagos Mês a Mês por Procedimento', fontsize=14)
plt.xlabel('Mês', fontsize=9)
plt.ylabel('Valor Total Pago (R$)', fontsize=9)
plt.xticks(range(1, 13), ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'],
           fontsize=9)
plt.yticks(fontsize=9)
plt.legend(title='Procedimento', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.show()
# Fim -- Procedimentos

# Inicio -- Modelo Sarima
# 1. Preparar os Dados
# Agrupar por mês e ano e somar os valores pagos
dados_agrupados = df.groupby(['ANO', 'MES'])['VL_TOTAL_PAGO'].sum().reset_index()

# Criar uma coluna de data para a série temporal
dados_agrupados['DATA'] = pd.to_datetime(dados_agrupados['ANO'].astype(str) + '-' + dados_agrupados['MES'].astype(str))

# Ordenar por data
dados_agrupados = dados_agrupados.sort_values('DATA')

# Criar a série temporal
serie_temporal = dados_agrupados.set_index('DATA')['VL_TOTAL_PAGO']

# 2. Análise Exploratória com Rótulos de Dados
plt.figure(figsize=(12, 6))
plt.plot(serie_temporal, marker='o', label='Custos Totais')

# Adicionar rótulos de dados para cada ponto
for x, y in zip(serie_temporal.index, serie_temporal.values):
    plt.text(x, y, f'R$ {y:.2f}', color='black', fontsize=8, ha='center', va='bottom')
# Configurações do gráfico
plt.title('Custos Totais Mês a Mês (Procedimentos Oncológicos)', fontsize=14)
plt.xlabel('Data', fontsize=9)
plt.ylabel('Custo Total (R$)', fontsize=9)
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.grid(True)
plt.legend(fontsize=9)
plt.tight_layout()
plt.show()
print("\n")

# Exibir os dados agrupados
print("Dados Agrupados por Mês e Ano:")
print(dados_agrupados)

# 3. Modelagem com SARIMA
# Dividir os dados em treino (até dezembro de 2024) e teste (2025)
train = serie_temporal[:'2024-12']  # Dados até dezembro de 2024
test = serie_temporal['2025-01':]  # Dados a partir de janeiro de 2025 (se houver)

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
modelo = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))  # (p, d, q) x (P, D, Q, S)
resultado = modelo.fit(disp=False)

# Fazer previsões para o período de teste (2025)
previsoes_2025 = resultado.get_forecast(steps=12)  # 12 meses de 2025
previsoes_2025_intervalo = previsoes_2025.conf_int()
previsoes_2025_media = previsoes_2025.predicted_mean

# 4. Visualização dos Resultados com Eixo X de 12 Meses
plt.figure(figsize=(14, 8))

# Criar um eixo x com 12 meses (1 a 12)
meses = range(1, 13)

# Plotar os dados históricos (2024)
plt.plot(meses, serie_temporal.values, label='Dados Históricos (2024)', marker='o', color='blue')

# Adicionar rótulos de dados para os dados históricos
for x, y in zip(meses, serie_temporal.values):
    plt.text(x, y, f'R$ {y:.2f}', color='blue', fontsize=8, ha='center', va='bottom')

# Plotar as previsões para 2025 acima dos dados históricos
# Adicionar um deslocamento vertical para destacar a diferença
deslocamento_vertical = max(serie_temporal.values) * 0.1  # 10% do valor máximo histórico
previsoes_2025_deslocadas = previsoes_2025_media.values + deslocamento_vertical

plt.plot(meses, previsoes_2025_deslocadas, label='Previsões 2025', color='red', linestyle='--', marker='o')

# Adicionar rótulos de dados para as previsões de 2025
for x, y in zip(meses, previsoes_2025_deslocadas):
    plt.text(x, y, f'R$ {y:.2f}', color='red', fontsize=8, ha='center', va='bottom')

# Adicionar intervalo de confiança para as previsões de 2025 (deslocado verticalmente)
previsoes_2025_intervalo_deslocado = previsoes_2025_intervalo + deslocamento_vertical
plt.fill_between(meses,
                 previsoes_2025_intervalo_deslocado.iloc[:, 0],
                 previsoes_2025_intervalo_deslocado.iloc[:, 1],
                 color='red', alpha=0.1)

# Configurações do gráfico
plt.title('Comparação de Custos Totais Mês a Mês (2024 vs 2025)', fontsize=14)
plt.xlabel('Mês', fontsize=9)
plt.ylabel('Custo Total (R$)', fontsize=9)

# Formatar eixo x com meses (Jan, Fev, Mar, etc.)
plt.xticks(ticks=meses,
           labels=['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'],
           fontsize=9, rotation=45)

plt.yticks(fontsize=9)
plt.legend(fontsize=9)
plt.grid(True)
plt.tight_layout()
plt.show()
print("\n")

# Exibir as previsões para 2025
print("Previsões de Custos Totais para 2025:")
print(previsoes_2025)
# Fim -- Modelo Sarima