import io
import itertools
import warnings
import streamlit as st
import holidays
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf
from stqdm import stqdm
from termcolor import colored

warnings.simplefilter('ignore', category=UserWarning)


class Shell_SARIMAX:
    def __init__(self, filepath, train_set_name, column_name, start_date, end_date, prediction_start_date,
                 prediction_end_date):
        self.filepath = filepath
        self.train_set_name = train_set_name
        self.column_name = column_name
        self.start_date = start_date
        self.end_date = end_date
        self.prediction_start_date = prediction_start_date
        self.prediction_end_date = prediction_end_date

        self.train_set = None
        self.train_endog = None
        self.train_exog = None
        self.test_exog = None
        self.best_params = None
        self.best_SARIMAX = None
        self.seasonal_periods = None

    def transform_day_of_week(self, df):
        df.index = pd.to_datetime(df.index)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        df['is_monday'] = (df.index.dayofweek == 1).astype(int)
        df['is_friday'] = (df.index.dayofweek == 4).astype(int)

        return df

    def create_train_set(self):
        file = self.filepath.get(self.train_set_name)
        file_content = file.read()
        file_obj = io.BytesIO(file_content)
        train_set = pd.read_csv(file_obj, index_col="Date")
        self.train_set = pd.DataFrame(train_set[self.column_name], columns=[self.column_name])
        self.train_endog = pd.DataFrame(self.train_set[f"{self.column_name}"], columns=["Net Cashflow from Operations"],
                                        index=self.train_set.index)
        self.train_endog.index = pd.to_datetime(self.train_endog.index)
        self.train_endog = self.train_endog.asfreq('D').fillna(0)

    def pacf_acf_plots(self):
        # Partial Autocorrelation Function and Autocorrelation Function
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        plot_acf(self.train_endog["Net Cashflow from Operations"], ax=ax1, lags=50)
        plot_pacf(self.train_endog["Net Cashflow from Operations"], ax=ax2, lags=50)
        plt.show()

    def seasonal_decomposition(self):
        # Seasonal Decomposition
        res = seasonal_decompose(self.train_endog[self.train_endog.index >= self.start_date], model='additive')
        # Extract the seasonality component
        seasonality = res.seasonal
        # Find the most dominant period of seasonality
        self.seasonal_periods = []
        for i in range(2, len(seasonality) // 2):
            autocorr = np.abs(acf(seasonality, nlags=i, fft=True))
            if autocorr[-1] > 2 / np.sqrt(len(seasonality)):
                self.seasonal_periods.append(i)
        # Print the dominant seasonal periods
        # print("Dominant seasonal periods found:", self.seasonal_periods)

    def create_train_test_exog_endog(self):
        # feature engineering
        self.train_exog = self.train_endog.copy()
        self.test_set = pd.DataFrame(index=pd.date_range(start=self.prediction_start_date,
                                                         end=self.prediction_end_date,
                                                         freq='D'), columns=["Net Cashflow from Operations"])
        self.train_exog = pd.concat([self.train_exog, self.test_set], axis=0)
        self.train_exog.index = pd.to_datetime(self.train_exog.index)
        self.train_exog = self.train_exog.asfreq('D').fillna(0)
        self.train_exog = self.transform_day_of_week(self.train_exog)
        turkish_holidays = holidays.Turkey(years=self.train_exog.index.year.unique())
        # create a holiday feature
        self.train_exog['Date'] = self.train_exog.index
        self.train_exog['is_holiday'] = self.train_exog['Date'].apply(lambda x: x in turkish_holidays).astype(int)
        self.train_exog.drop(["Date"], axis=1, inplace=True)
        self.train_exog["Net Cashflow from Operations"] = self.train_exog["Net Cashflow from Operations"].astype(
            "float64")
        self.train_exog = self.train_exog.drop("Net Cashflow from Operations", axis=1)
        self.train_exog.index = pd.to_datetime(self.train_exog.index)
        self.test_exog = self.train_exog[self.train_exog.index >= self.prediction_start_date]
        self.train_exog = self.train_exog[self.train_exog.index < self.prediction_start_date]

        self.train_endog = self.train_endog.dropna()
        # self.train_exog.to_csv(f'{self.filepath}/train_exog.csv')
        # self.train_endog.to_csv(f'{self.filepath}/train_endog.csv')

    def SARIMAX_gridsearch(self):
        p_values = range(0, 3)  # Replace with the desired range of p values
        q_values = range(0, 3)  # Replace with the desired range of q values
        d_values = range(0, 3)  # Replace with the desired range of d values
        P_values = range(0, 3)  # Replace with the desired range of P values
        D_values = range(0, 3)  # Replace with the desired range of D values
        Q_values = range(0, 3)  # Replace with the desired range of Q values

        parameters = list(
            itertools.product(p_values, d_values, q_values, P_values, D_values, Q_values, self.seasonal_periods[:5]))
        chunk_size = 300  # Number of parameter combinations to process in each chunk

        best_aic = float("inf")
        self.best_params = None

        num_chunks = (len(parameters) + chunk_size - 1) // chunk_size
        for chunk_idx in stqdm(range(num_chunks)):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(parameters))
            chunk = parameters[start_idx:end_idx]
            st.cache_data.clear()
            for param in stqdm(chunk):
                try:
                    # Fit the SARIMAX model with the current combination of parameters
                    model = SARIMAX(
                        self.train_endog[self.train_endog.index >= self.start_date],
                        exog=self.train_exog[self.train_exog.index >= self.start_date],
                        order=(param[0], param[1], param[2]),
                        seasonal_order=(param[3], param[4], param[5], param[6])
                    )
                    results = model.fit(disp=False)

                    # Calculate the AIC score for the model
                    aic = results.aic

                    # Update the best AIC and parameters if the current model has a lower AIC
                    if aic < best_aic:
                        best_aic = aic
                        self.best_params = param

                except Exception as e:
                    continue

        return self.best_params

    def SARIMAX_train_test(self):
        self.model = SARIMAX(self.train_endog[self.train_endog.index >= self.start_date],
                             exog=self.train_exog[self.train_exog.index >= self.start_date],
                             order=self.best_params[:3], seasonal_order=self.best_params[3:], verbose=1)
        self.best_SARIMAX = self.model.fit()
        print(colored("SARIMAX is succesfully trained", "green"))

    def Sarimax_Forecast(self):
        # Forecast using the trained model
        forecast = self.best_SARIMAX.get_forecast(steps=len(self.test_exog), exog=self.test_exog)
        forecast_values = forecast.predicted_mean
        forecast_values = forecast_values[~forecast_values.index.weekday.isin([5, 6])]
        print(colored("Succesfully forecasted using SARIMAX", "green"))
        return forecast_values

# depo_pump_imm = pd.read_csv(f'{filepath}/depo_pump_imm.csv')
# depo_pump_imm = depo_pump_imm.rename(columns={"Yıl": "Year", "Ay": "Month"})
# depo_pump_imm['Date'] = pd.to_datetime(depo_pump_imm[['Year', 'Month']].assign(day=1))
# depo_pump_imm.set_index('Date', inplace=True)
# depo_pump_imm = depo_pump_imm.resample('D').asfreq()
# depo_pump_imm = depo_pump_imm.reset_index().ffill()
# depo_pump_imm = depo_pump_imm.drop(["Year", "Month"], axis=1)
# depo_pump_imm.set_index('Date', inplace=True)
# depo_pump_imm_shifted = depo_pump_imm
# depo_pump_imm_shifted.index = depo_pump_imm_shifted.index + pd.DateOffset(years=1)
#
# for column in depo_pump_imm_shifted.columns:
#     depo_pump_imm_shifted = depo_pump_imm_shifted.rename(columns={column: f"{column}_shifted"})
#     depo_pump_imm = pd.concat([depo_pump_imm, depo_pump_imm_shifted], axis=1)
#
# non_unique_indexes = depo_pump_imm.index[depo_pump_imm.index.duplicated()]
# # depo_pump_imm = depo_pump_imm[~depo_pump_imm.index.duplicated(keep='first')].replace(',', '.', regex=True).astype(
# #     'float64')
# otv = pd.read_csv(f'{filepath}/otv.csv', index_col="Tarih").drop(["ÖTV Oranı", "Adet"], axis=1)
# otv = otv.rename(index={"Tarih": "Date"})
# otv.index = pd.to_datetime(otv.index)
#
# platts = pd.read_csv(f'{filepath}/platts.csv', index_col="Tarih").drop(["Avrupa Birliği Birimi"], axis=1)
# platts = platts.rename(index={"Tarih": "Date"})
# platts.index = pd.to_datetime(platts.index)
# platts_1 = platts.iloc[:1825, :]
# platts_1 = platts_1.drop("Ürün", axis=1)
# for column in platts_1.columns:
#     platts_1 = platts_1.rename(columns={column: f"{column}_10 ppm ULSD CIF Med(Genova/Lavera)"})
#
# platts_2 = platts.iloc[1825:, :]
# platts_2 = platts_2.drop("Ürün", axis=1)
# for columns in platts_2.columns:
#     platts_2 = platts_2.rename(columns={columns: f"{columns}_Prem Unl 10 ppm ULSD CIF Med(Genova/Lavera)"})
#
# usd = pd.read_csv(f'{filepath}/usd.csv', index_col="Tarih").drop(["Yıl"], axis=1)
# usd = usd.rename(index={"Tarih": "Date"})
# usd.index = pd.to_datetime(usd.index)
#
# volume = pd.read_csv(f'{filepath}/volume.csv', index_col="Posting date")
# volume = volume.rename(index={"Posting date": "Date"})
# volume.index = pd.to_datetime(volume.index)
#
# groups = otv.groupby(otv["ÖTV uygulanan Ürün Adı"])
#
# # Create a dictionary to store the unique indexed DataFrames
# unique_dfs = {}
#
# # Iterate over the groups and create unique indexed DataFrames
# for group_name, group_df in groups:
#     unique_dfs[group_name] = group_df
#
# # Access the individual unique indexed DataFrames
# for key, value in unique_dfs.items():
#     key = key.replace(" ", "_")
#     locals()[f'df_otv_{key}'] = value
#     non_unique_indexes = locals()[f'df_otv_{key}'].index[locals()[f'df_otv_{key}'].index.duplicated()]
#     locals()[f'df_otv_{key}'] = locals()[f'df_otv_{key}'][~locals()[f'df_otv_{key}'].index.duplicated(keep='first')]
#     locals()[f'df_otv_{key}'] = locals()[f'df_otv_{key}'].rename(
#         columns={locals()[f'df_otv_{key}'].columns[0]: f"OTV_{key}"})
#     locals()[f'df_otv_{key}'] = locals()[f'df_otv_{key}'].drop(["ÖTV uygulanan Ürün Adı"], axis=1)
#     locals()[f'df_otv_{key}'] = locals()[f'df_otv_{key}'].resample('D').asfreq().ffill()
#     # train_exog = pd.concat([train_exog, locals()[f'df_otv_{key}']], axis=1)
#
# # train_exog = pd.concat([train_exog, depo_pump_imm, usd, platts_1, platts_2], axis=1)
# # train_exog = train_exog[train_exog.index <= "2023-02-01 00:00:00"]
