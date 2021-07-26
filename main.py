from portfoliolab.bayesian import VanillaBlackLitterman
from model import BlackLitterman
import pandas as pd
countries = ['AU', 'CA', 'FR', 'DE', 'JP', 'UK', 'US']
# Table 1 of the He-Litterman paper: Correlation matrix
correlation = pd.DataFrame([
    [1.000, 0.488, 0.478, 0.515, 0.439, 0.512, 0.491],
    [0.488, 1.000, 0.664, 0.655, 0.310, 0.608, 0.779],
    [0.478, 0.664, 1.000, 0.861, 0.355, 0.783, 0.668],
    [0.515, 0.655, 0.861, 1.000, 0.354, 0.777, 0.653],
    [0.439, 0.310, 0.355, 0.354, 1.000, 0.405, 0.306],
    [0.512, 0.608, 0.783, 0.777, 0.405, 1.000, 0.652],
    [0.491, 0.779, 0.668, 0.653, 0.306, 0.652, 1.000]
], index=countries, columns=countries)
# Table 2 of the He-Litterman paper: Volatilities
volatilities = pd.DataFrame([0.160, 0.203, 0.248, 0.271, 0.210, 0.200, 0.187],
                            index=countries, columns=["vol"])
covariance = volatilities.dot(volatilities.T) * correlation

# Table 2 of the He-Litterman paper: Market-capitalised weights
market_weights = pd.DataFrame([0.016, 0.022, 0.052, 0.055, 0.116, 0.124, 0.615],
                              index=countries, columns=["CapWeight"])

# Q
views = [0.05]
# P
pick_list = [
        {
            "DE": 1.0,
            "FR": -market_weights.loc["FR"]/(market_weights.loc["FR"] + \
                                            market_weights.loc["UK"]),
            "UK": -market_weights.loc["UK"] / (market_weights.loc["FR"] + \
                                                market_weights.loc["UK"])
        }
]
bl = BlackLitterman()
bl.allocate(covariance=covariance,
            market_capitalised_weights=market_weights,
            investor_views=views,
            pick_list=pick_list,
            asset_names=covariance.columns,
            tau=0.05,
            risk_aversion=2.5)
print(bl.weights)

# Q
views2 = [0.05, 0.03]
# P
pick_list2 = [
    {
        "DE": 1.0,
        "FR": -market_weights.loc["FR"]/(market_weights.loc["FR"] + market_weights.loc["UK"]),
        "UK": -market_weights.loc["UK"] / (market_weights.loc["FR"] + market_weights.loc["UK"])
    },
    {
        "CA": 1,
        "US": -1
    }
]
# Allocate
bl2 = BlackLitterman()
bl2.allocate(covariance=covariance,
            market_capitalised_weights=market_weights,
            investor_views=views2,
            pick_list=pick_list2,
            asset_names=covariance.columns,
            tau=0.05,
            risk_aversion=2.5)
print(bl2.weights)

# Q
views3 = [0.05, 0.04]
# P
pick_list3 = [
    {
        "DE": 1.0,
        "FR": -market_weights.loc["FR"]/(market_weights.loc["FR"] + market_weights.loc["UK"]),
        "UK": -market_weights.loc["UK"] / (market_weights.loc["FR"] + market_weights.loc["UK"])
    },
    {
        "CA": 1,
        "US": -1
    }
]
# Allocate
bl3 = BlackLitterman()
bl3.allocate(covariance=covariance,
            market_capitalised_weights=market_weights,
            investor_views=views3,
            pick_list=pick_list3,
            asset_names=covariance.columns,
            tau=0.05,
            risk_aversion=2.5)
print(bl3.weights)