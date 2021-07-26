# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
import FinanceDataReader as fdr
from model import *


samsungElec = fdr.DataReader('005930', '2018-07-13', '2019-12-31')
samsungElec['log_return'] = np.log(samsungElec.Close) - np.log(samsungElec.Close.shift(1))

KGMobil = fdr.DataReader('046440', '2018-07-13', '2019-12-31')
KGMobil['log_return'] = np.log(KGMobil.Close) - np.log(KGMobil.Close.shift(1))

hyosungChem = fdr.DataReader('298000', '2017-01-01', '2019-12-31')
hyosungChem['log_return'] = np.log(hyosungChem.Close) - np.log(hyosungChem.Close.shift(1))

celltrion = fdr.DataReader('068270', '2017-01-01')
celltrion['log_return'] = np.log(celltrion.Close) - np.log(celltrion.Close.shift(1))

dfReturns = pd.concat([samsungElec['log_return'], KGMobil['log_return'], hyosungChem['log_return'], celltrion['log_return']], axis=1, join='inner')
dfReturns.columns = ['삼성전자', 'KG 모빌리언스', '효성화학', '셀트리온']
dfReturns = dfReturns[1:]

correlation = dfReturns.corr()
volatilities = dfReturns.std() * np.sqrt(252)

capSE = 5561791
capKG = 4290
capHS = 12761
capCell = 396586
capSum = capSE + capKG + capHS + capCell
cap_weights = [capSE/capSum, capKG/capSum, capHS/capSum, capCell/capSum]
dfCW = pd.DataFrame(cap_weights).T
dfCW.columns = ['삼성전자', 'KG 모빌리언스', '효성화학', '셀트리온']
dfCW.index = ['cap_weights']

# Q
views = [0.079, 0.049]
# P
pick_list = [
        {
            "효성화학": 1
        },
        {
            "KG 모빌리언스": 1
        }
]

stocks = ['삼성전자', 'KG 모빌리언스', '효성화학', '셀트리온']
correlation = pd.DataFrame(correlation, index=stocks, columns=stocks)
volatilities = pd.DataFrame(volatilities,
                            index=stocks, columns=["vol"])
covariance = volatilities.dot(volatilities.T) * correlation
cap_weights = pd.DataFrame(cap_weights,
                              index=stocks, columns=["CapWeight"])

bl = BlackLitterman()
bl.allocate(covariance=covariance,
            market_capitalised_weights=cap_weights,
            investor_views=views,
            pick_list=pick_list,
            asset_names=covariance.columns,
            tau=0.05,
            risk_aversion=2.5)
print(bl.weights)

