import holidays
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import adfuller
import io

class Shell_EDA:
    def __init__(self,filepath, train_set_name, column_name, start_date, end_date,output_path):
        self.filepath = filepath
        self.output_path = output_path
        self.train_set_name = train_set_name
        self.column_name = column_name
        self.start_date = start_date
        self.end_date = end_date
        self.dataset = None

    def read_data(self):
        train_set = pd.read_csv(f'{self.filepath}/{self.train_set_name}', index_col="Date")
        self.dataset = pd.DataFrame(train_set[self.column_name], columns=[self.column_name])
        self.dataset.index = pd.to_datetime(self.dataset.index)
        self.dataset = self.dataset.asfreq('D').fillna(0)

    def acf_pacf(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        plot_acf(self.dataset["Net Cashflow from Operations"], ax=ax1, lags=50)
        plot_pacf(self.dataset["Net Cashflow from Operations"], ax=ax2, lags=50)
        plt.savefig(f'{self.output_path}/acf_pacf_plot.png')

        return f'{self.output_path}/acf_pacf_plot.png'

    def decompose(self):
        # Seasonal Decomposition
        self.dataset.index = pd.to_datetime(self.dataset.index)
        self.dataset = self.dataset.asfreq('D').fillna(0)
        res = seasonal_decompose(self.dataset[self.dataset.index >= self.start_date], model='additive')
        # Extract the seasonality component
        seasonality = res.seasonal
        # %%
        seasonal_periods = []
        for i in range(2, len(seasonality) // 2):
            autocorr = np.abs(acf(seasonality, nlags=i, fft=True))
            if autocorr[-1] > 2 / np.sqrt(len(seasonality)):
                seasonal_periods.append(i)

        Seasonal_Decompose = f"Dominant seasonal periods found:{seasonal_periods}"

        return Seasonal_Decompose

    def information(self):
        buffer = io.StringIO()
        self.dataset.info(buf=buffer)
        pandas_info_info = buffer.getvalue()
        pandas_info_describe = self.dataset.describe()

        return pandas_info_info+"\n"+str(pandas_info_describe)

    def mean_std_plot(self):
        plt.figure(figsize=(15, 7))
        plt.plot(self.dataset[-84:], label='Original', marker="o")
        plt.plot(self.dataset[-84:].rolling(window=7).mean(), color='red', label='Rolling mean')
        plt.plot(self.dataset[-84:].rolling(window=7).std(), color='green', label='Rolling std')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Net Cashflow', fontsize=12)
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')

        plt.savefig(f'{self.output_path}/mean_std_plot.png')

        return f'{self.output_path}/mean_std_plot.png'

    def DF_test(self):
        dftest = adfuller(self.dataset,autolag="AIC")
        dfoutput = pd.Series(
            dftest[0:4],
            index=[
                "Test Statistic",
                "p-value",
                "Lags Used",
                "Number of Samples",
            ],
        )
        for k, val in dftest[4].items():
            dfoutput["Critical Value (%s)" % k] = val
        if dfoutput["p-value"] <= 0.05:
            df_result = "Data is stationary"
        else:
            df_result = "Data is not stationary"

        return "Results of Dickey-Fuller Test:\n"+str(dfoutput)+"\n"+df_result+"\n"

    def seasonal_decomp(self):
        decomp = sm.tsa.seasonal.seasonal_decompose(self.dataset[-68:], model="additive")
        seasonal = decomp.seasonal
        trend = decomp.trend
        residual = decomp.resid
        decomp_f = decomp.plot()
        decomp_f.set_size_inches(15, 7)
        plt.savefig(f'{self.output_path}/seasonal_decomp.png')

        return f'{self.output_path}/seasonal_decomp.png'


    def ordered_pacf_acf(self):
        ###AutoCorrelation and Partial AutoCorrelation
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        plot_acf(self.dataset, alpha=.05, title="ACF with No Diff.")
        plot_pacf(self.dataset, alpha=.05, title="PACF with No Diff.")
        plt.savefig(f'{self.output_path}/pacf_acf_order_0.png')
        ### First order diff
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        plot_acf(self.dataset.diff().dropna(), alpha=.05, title="ACF with First Order Diff.")
        plot_pacf(self.dataset.diff().dropna(), alpha=.05, title="PACF with First Order Diff.")
        plt.savefig(f'{self.output_path}/pacf_acf_order_1.png')
        ### Second order diff
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        plot_acf(self.dataset.diff().diff().dropna(), alpha=.05, title="ACF with Second Order Diff.")
        plot_pacf(self.dataset.diff().diff().dropna(), alpha=.05, title="PACF with Second Order Diff.")
        plt.savefig(f'{self.output_path}/pacf_acf_order_2.png')

        return f'{self.output_path}/pacf_acf_order_0.png', f'{self.output_path}/pacf_acf_order_1.png', f'{self.output_path}/pacf_acf_order_2.png'

