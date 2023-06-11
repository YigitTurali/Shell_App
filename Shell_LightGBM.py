import io
import warnings

import holidays
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from termcolor import colored

warnings.filterwarnings("ignore", category=UserWarning)


class Shell_LGBM:

    def __init__(self, filepath, train_set_name, column_name, start_date, end_date, prediction_start_date,
                 prediction_end_date, param_set):
        self.filepath = filepath
        self.train_set_name = train_set_name
        self.column_name = column_name
        self.start_date = start_date
        self.end_date = end_date
        self.prediction_start_date = prediction_start_date
        self.prediction_end_date = prediction_end_date
        self.param_set = param_set
        self.best_params = None
        self.best_model = None

    def transform_day_of_week(self, df):
        df.index = pd.to_datetime(df.index)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        df['day_of_month_sin'] = np.sin(2 * np.pi * df.index.day / 30)
        df['day_of_month_cos'] = np.cos(2 * np.pi * df.index.day / 30)
        return df

    def create_train_set(self):
        file = self.filepath.get(self.train_set_name)
        file_content = file.read()
        file_obj = io.BytesIO(file_content)
        self.train_set = pd.read_csv(file_obj, index_col="Date")
        self.train_endog = pd.DataFrame(self.train_set[f"{self.column_name}"],
                                        columns=["Net Cashflow from Operations"], index=self.train_set.index)
        self.train_endog.index = pd.to_datetime(self.train_endog.index)
        self.train_endog = self.train_endog.asfreq('D').fillna(0)

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

    def LGBM_GridSearch(self):
        boosting_type = self.param_set["boosting_type"]
        num_leaves = self.param_set["num_leaves"]
        max_depth = self.param_set["max_depth"]
        learning_rate = self.param_set["learning_rate"]
        subsample = self.param_set["subsample"]
        colsample_bytree = self.param_set["colsample_bytree"]
        reg_alpha = self.param_set["reg_alpha"]
        reg_lambda = self.param_set["reg_lambda"]
        n_estimators = self.param_set["n_estimators"]
        random_state = self.param_set["random_state"]

        params = {'objective': ['regression'],
                  'boosting_type': boosting_type,
                  'num_leaves': num_leaves,
                  'max_depth': max_depth,
                  'learning_rate': learning_rate,
                  'subsample': subsample,
                  'colsample_bytree': colsample_bytree,
                  'reg_alpha': reg_alpha,
                  'reg_lambda': reg_lambda,
                  'n_estimators': n_estimators,
                  'random_state': random_state,
                  'num_iterations': [100, 200, 500]}

        model = lgb.LGBMRegressor()
        random_search = RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=250, cv=5,
                                           random_state=42, verbose=4, scoring="neg_mean_absolute_error", n_jobs=-1)
        dataset = lgb.Dataset(self.train_exog[self.train_exog.index >= self.start_date],
                              label=self.train_endog[self.train_endog.index >= self.start_date])
        random_search.fit(dataset.data, dataset.label)

        self.best_params = random_search.best_params_
        self.best_model = random_search.best_estimator_

        print(colored(f"Best parameter set for LightGBM: {self.best_params}", "green"))
        print(colored("Succesfully trained with LightGBM", "green"))

    def Train_LightGBM(self):
        dataset = lgb.Dataset(self.train_exog[self.train_exog.index >= self.start_date],
                              label=self.train_endog[self.train_endog.index >= self.start_date])
        model = lgb.LGBMRegressor(**self.best_params, verbose=1)
        model.fit(dataset.data, dataset.label, verbose=1)
        self.best_model = model
        print(colored("Succesfully trained with LightGBM", "green"))

    def Forecast_LightGBM(self):
        forecast_values = pd.Series(
            self.best_model.predict(self.test_exog, steps=len(self.test_exog)), index=self.test_exog.index)
        forecast_values = forecast_values[~forecast_values.index.weekday.isin([5, 6])]
        print(colored("Succesfully forecasted with LightGBM", "green"))
        return forecast_values
