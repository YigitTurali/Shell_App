import pandas as pd
import sys
import json
import io
from Shell_EDA import Shell_EDA
from Shell_Ensemble import Shell_Ensemble
from Shell_LightGBM import Shell_LGBM
from Shell_SARIMAX import Shell_SARIMAX
import argparse
parser = argparse.ArgumentParser(description='Main Script')

if __name__ == '__main__':
    parser.add_argument('--EDA', type=str, help='EDA Flag',default=False)
    parser.add_argument('--filepath', type=str, help='File Path', default="new-shell-cashflow-datathon-2023")
    parser.add_argument('--train_set_name', type=str, help='train_set_name', default="cash_flow_train.csv")
    parser.add_argument('--column_name', type=str, help='column_name', default="Net Cashflow from Operations")
    parser.add_argument('--start_date', type=str, help='start_date', default="2021-01-01")
    args = parser.parse_args()
    EDA = args.EDA
    filepath = args.filepath
    train_set_name = args.train_set_name
    column_name = args.column_name
    start_date = args.start_date

    if EDA:
        shell_data_eda = Shell_EDA(filepath=filepath,
                                   train_set_name=train_set_name,
                                   column_name=column_name,
                                   start_date=start_date,
                                   end_date=None)
        shell_data_eda.read_data()
        acf_pacf_path = shell_data_eda.acf_pacf()
        seasonal_decomp_path = shell_data_eda.decompose()
        info_path_1, info_path_2 = shell_data_eda.information()
        mean_std_plot_path = shell_data_eda.mean_std_plot()
        df_test_path = shell_data_eda.DF_test()
        seasonal_decomp_path_2 = shell_data_eda.seasonal_decomp()
        order_1_path,order_2_path,order_3_path = shell_data_eda.ordered_pacf_acf()

        response = {
            'acfPacfPlotPath': acf_pacf_path,
            'seasonalDecompPlotPath': seasonal_decomp_path,
            'pandasInfoDescribe': [info_path_1, info_path_2],
            'meanStdPlotPath': mean_std_plot_path,
            'dickeyFullerTest': df_test_path,
            'seasonalDecompPlotPath2': seasonal_decomp_path_2,
            'orderedPacfAcfPlotsPath': [order_1_path,order_2_path,order_3_path]
        }

        data = json.load(sys.stdin)
        json.dump(response, sys.stdout)
    else:
        shell_sarimax_class = Shell_SARIMAX(filepath="new-shell-cashflow-datathon-2023",
                                            train_set_name="cash_flow_train.csv",
                                            column_name="Net Cashflow from Operations",
                                            start_date="2021-01-01",
                                            end_date=None, prediction_start_date="2023-02-02",
                                            prediction_end_date="2023-05-12")

        shell_sarimax_class.create_train_set()
        shell_sarimax_class.pacf_acf_plots()
        shell_sarimax_class.seasonal_decomposition()
        shell_sarimax_class.create_train_test_exog_endog()

        # shell_sarimax_best_params = shell_sarimax_class.SARIMAX_gridsearch()
        shell_sarimax_class.best_params = (1, 0, 1, 2, 2, 2, 4)
        shell_sarimax_class.SARIMAX_train_test()
        shell_sarimax_class_forecast_output = shell_sarimax_class.Sarimax_Forecast()

        shell_lgbm_class = Shell_LGBM(filepath="new-shell-cashflow-datathon-2023", train_set_name="cash_flow_train.csv",
                                      column_name="Net Cashflow from Operations",
                                      start_date="2022-01-01",
                                      end_date=None,
                                      prediction_start_date="2023-02-02",
                                      prediction_end_date="2023-05-12",
                                      param_set={'objective': ['regression'],
                                                 'boosting_type': ["goss"],
                                                 'num_leaves': [15, 31, 63, 127, 255],
                                                 'max_depth': [3, 5, 7, 15, 31],
                                                 'learning_rate': [0.1, 0.01, 0.001],
                                                 'subsample': [0.8, 0.6, 1.0],
                                                 'colsample_bytree': [0.8, 0.6, 1.0],
                                                 'reg_alpha': [0.0, 0.1, 0.5],
                                                 'reg_lambda': [0.0, 0.1, 0.5],
                                                 'n_estimators': [100, 200, 500],
                                                 'random_state': [42],
                                                 'num_iterations': [100, 200, 500]})

        shell_lgbm_class.create_train_set()
        shell_lgbm_class.create_train_test_exog_endog()
        shell_lgbm_class.LGBM_GridSearch()
        shell_lgbm_class_forecast_output = shell_lgbm_class.Forecast_LightGBM()
        sarimax_output = pd.DataFrame(columns = ["Date","Net Cashflow from Operations"])
        lgbm_output = pd.DataFrame(columns=["Date", "Net Cashflow from Operations"])
        sarimax_output["Date"] = shell_sarimax_class_forecast_output.index
        sarimax_output["Net Cashflow from Operations"] = shell_sarimax_class_forecast_output.values

        lgbm_output["Date"] = shell_lgbm_class_forecast_output.index
        lgbm_output["Net Cashflow from Operations"] = shell_lgbm_class_forecast_output.values

        shell_sarimax_class_forecast_output.to_csv(f"{shell_sarimax_class.filepath}/submission_sarimax.csv", index=False)
        shell_lgbm_class_forecast_output.to_csv(f"{shell_lgbm_class.filepath}/submission_lgbm.csv", index=False)

        shell_ensemble_class = Shell_Ensemble(submission_SARIMAX=shell_sarimax_class_forecast_output,
                                              submission_LGBM=shell_lgbm_class_forecast_output,
                                              file_path=shell_sarimax_class.filepath,
                                              file_name="submission_ensemble.csv")
        shell_ensemble_class.Ensemble()