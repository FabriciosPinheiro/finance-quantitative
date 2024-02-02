import fundamentus
import pandas as pd 
import matplotlib.pyplot as plt 
from datetime import datetime, timedelta
import yfinance as yf 
import numpy as np 
import riskfolio as rp
import vectorbt as vbt

df = fundamentus.get_resultado()
df.head()

#Filtra o PL
#df.loc[['GRND3', 'ABEV3'], ['pl']] #Linhas X Colunas

#df[(df.pl > 0) & (df.dy > 0.05)]

# Ordena pelo DY do Maior para o menor
df[(df.pl > 0) & (df.dy) > 0.1].sort_values('dy', ascending = False)

#Cria uma nova coluna
df['Ticket'] = df.index.str[:4]
df.head()

#Remove os Tickets duplicados
df = df.drop_duplicates(subset=['Ticket'])
df.head()

#Filtra empresas com o PL positivo
filtrado = df[(df.patrliq > 0) & (df.divbpatr < 0.5)] #50%

#Ordena com base na Dívida Bruta Sobre o Patrimônio
div_bruta_patr = filtrado.sort_values('divbpatr', ascending = True)
div_bruta_patr = div_bruta_patr.head(50)

div_bruta_patr['ACAO'] = div_bruta_patr.index + '.SA'

def is_traded_recently(symbol): #Verifica se a Ação foi negociada recentemente
    try:
        #Traz os dados históricos de cotação para aquela ação
        stock = yf.Ticker(symbol)
        historical_data = stock.history(period='1mo')

        #Traz a cotação mais recente e obtem a data correspondente
        last_trading_day = historical_data.index[-1].date()

        #Verifica o dia de negociação mais recente
        today = datetime.now().date()
        window_start = today - timedelta(days=5)

        return window_start <= last_trading_day <= today
    except Exception:

        return False

#is_traded_recently('PETR4.SA')

#Aplica para cada Ação a Função
div_bruta_patr['Negociada'] = div_bruta_patr['ACAO'].apply(is_traded_recently)

#Filtra somente as Ações que são negociadas
empresas_filtradas = div_bruta_patr[div_bruta_patr['Negociada'] == True]

#Ordena pelo maior ROE e retorna somente as 10 primeiras ações
empresas_filtradas = empresas_filtradas.sort_values('roe', ascending = False).head(10)


plt.figure(figsize=(10,6));
plt.bar(empresas_filtradas.tail(10).index, empresas_filtradas.tail(10).roe);
plt.xticks(rotation=45);
plt.title("Ações selecionadas por Divida Bruta/Pat Liquido e ROE");

#ativos = [empresas_filtradas.index + '.SA']



#Modelos de Otimização

#Parâmetros

inicio='2023-01-01'
fim='2023-09-30'

#Seleção dos ativos

ativos = ['CORR3.SA', 'ODPV3.SA', 'CXSE3.SA', 'BRAP3.SA', 'CEBR3.SA', 'PSSA3.SA','ITUB3.SA', 'PINE3.SA', 'BSLI3.SA', 'LIPR3.SA']
#ativos = ['BBSE3.SA', 'ODPV3.SA', 'ALOS3.SA', 'BBAS3.SA', 'BAZA3.SA', 'PSSA3.SA', 'ITUB3.SA', 'BEES3.SA', 'PINE3.SA', 'LIPR3.SA']

#Extrair os dados de preço
carteira = yf.download(ativos, start=inicio, end=fim)['Adj Close']

carteira.head()

#Calcular os retornos dos ativos
retornos = carteira.pct_change().dropna()

retornos.cov()

#Modelo de otimização - Markowitz MAX SHARPE
#Sharpe Ratio = (Retorno Esperado - Taxa Livre de Risco)/Volatilidade da Carteira

portfolio = rp.Portfolio(returns=retornos)

mu = 'hist' #estimando os retornos com base no histórico
cov = 'hist' #estimar a matriz de covariância através do histórico

portfolio.assets_stats(method_mu=mu, method_cov=cov, d=0.94)

#Construindo o modelo

model='Classic'
rm='MV'
obj='Sharpe'
hist=True
rf=0
l=0

w = portfolio.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)

display(w.T)

ax = rp.plot_pie(w=w,
                 title= 'Otimização de Portfólio',
                 others=0.01,
                 nrow=25,
                 cmap='tab20',
                 height=8,
                 width=10,
                 ax=None)

#Backtest

start = '2023-09-30'
end = '2024-01-12'

vbt.settings.array_wrapper['freq'] = 'days'
vbt.settings.returns['year_freq'] = '252 days'
vbt.settings.portfolio.stats['incl_unrealized'] = True

ativos = ['ODPV3.SA', 'CXSE3.SA', 'BRAP3.SA', 'CEBR3.SA', 'PSSA3.SA', 'ITUB3.SA','PINE3.SA', 'BSLI3.SA', 'LPSB3.SA', 'CGRA3.SA']

precos = yf.download(ativos, start=start, end=end)['Close']

#Calcular os retornos diários
retornos = precos.pct_change().dropna()

#Backtesting Markowitz

pesos_sharpe = np.array([1.801258e-10,	2.345396e-10,	0.283559,	3.361861e-11,	0.527423,	4.185716e-10,	0.049914,	0.09117,	0.015732,	0.032201])

tamanho_sharpe= np.full_like(precos,np.nan)

tamanho_sharpe[0,:] = pesos_sharpe

#Simulação de backtesting para markowitz

pf_sharpe = vbt.Portfolio.from_orders(
    close=precos,
    size=tamanho_sharpe,
    size_type='targetpercent',
    group_by=True,
    cash_sharing=True,
)

pf_sharpe.plot().show()

pf_sharpe.stats()
