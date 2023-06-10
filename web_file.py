import io
import json
import os
import sys

import pandas as pd
from flask import Flask, request, render_template, redirect

from Shell_EDA import Shell_EDA
from Shell_Ensemble import Shell_Ensemble
from Shell_LightGBM import Shell_LGBM
from Shell_SARIMAX import Shell_SARIMAX

app = Flask(__name__)

console_output = []
progress = 0
is_training_complete = False
# Get the absolute path of the script
script_dir = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")

# Print the script directory path
print("Script directory path:", script_dir)


@app.route("/", methods=['GET'])
def hello():
    return render_template('Shell_Cashflow.html')


@app.route("/forecast", methods=['POST'])
def forecast():
    # Handle the form submission and generate the forecast

    # Retrieve the form data
    csv_file = request.files['csv-file']
    csv_file = pd.read_csv(csv_file)
    csv_file.to_csv('static/train_dataset.csv')
    grid_search_lightgbm = request.form.get('grid-search-lightgbm', False)
    grid_search_arima = request.form.get('grid-search-arima', False)
    model_selection = request.form.get('model-selection')
    try:
        sarimax_parameters = {
            'p-value': int(request.form.get('p-value-1')),
            'q-value': int(request.form.get('q-value-1')),
            'd-value': int(request.form.get('d-value-1')),
            'P-value': int(request.form.get('P-value-2')),
            'D-value': int(request.form.get('D-value-2')),
            'Q-value': int(request.form.get('Q-value-2')),
            's-value': int(request.form.get('s-value-2')),
        }

    except:
        sarimax_parameters = {
            'p-value': request.form.get('p-value-1'),
            'q-value': request.form.get('q-value-1'),
            'd-value': request.form.get('d-value-1'),
            'P-value': request.form.get('P-value-2'),
            'D-value': request.form.get('D-value-2'),
            'Q-value': request.form.get('Q-value-2'),
            's-value': request.form.get('s-value-2'),
        }
    try:
        lightgbm_parameters = {'num_leaves': int(request.form.get('num_leaves')),
                               'max_depth': int(request.form.get('max_depth')),
                               'learning_rate': float(request.form.get('learning_rate')),
                               'colsample_bytree': float(request.form.get('colsample_bytree')),
                               'subsample': float(request.form.get('subsample')),
                               'reg_alpha': float(request.form.get('reg_alpha')),
                               'reg_lambda': float(request.form.get('reg_lambda')),
                               'n_estimators': int(request.form.get('n_estimators')),
                               'random_state': int(request.form.get('random_state')),
                               'num_iterations': int(request.form.get('num_iterations'))}

    except:
        lightgbm_parameters = {'num_leaves': request.form.get('num_leaves'),
                               'max_depth': request.form.get('max_depth'),
                               'learning_rate': request.form.get('learning_rate'),
                               'colsample_bytree': request.form.get('colsample_bytree'),
                               'subsample': request.form.get('subsample'),
                               'reg_alpha': request.form.get('reg_alpha'),
                               'reg_lambda': request.form.get('reg_lambda'),
                               'n_estimators': request.form.get('n_estimators'),
                               'random_state': request.form.get('random_state'),
                               'num_iterations': request.form.get('num_iterations')}

    training_start_date_sarimax = request.form.get('training-start-date-sarimax')
    training_end_date_sarimax = request.form.get('training-end-date-sarimax')
    training_start_date_lightgbm = request.form.get('training-start-date-lightgbm')
    training_end_date_lightgbm = request.form.get('training-end-date-lightgbm')
    test_start_date = request.form.get('test-start-date')
    test_end_date = request.form.get('test-end-date')
    eda_flag = request.form.get('analysis-flag')

    print(eda_flag)

    full_train = {"filepath": "static",
                  "csv_file_name": "train_dataset.csv",
                  "grid_search_lightgbm": grid_search_lightgbm,
                  "grid_search_arima": grid_search_arima,
                  "model_selection": model_selection,
                  "sarimax_parameters": sarimax_parameters,
                  "lightgbm_parameters": lightgbm_parameters,
                  "training_start_date_sarimax": training_start_date_sarimax,
                  "training_end_date_sarimax": training_end_date_sarimax,
                  "training_start_date_lightgbm": training_start_date_lightgbm,
                  "training_end_date_lightgbm": training_end_date_lightgbm,
                  "test_start_date": test_start_date,
                  "test_end_date": test_end_date,
                  "eda_flag": eda_flag}

    with open('static/full_train.json', 'w') as fp:
        json.dump(full_train, fp)
    # Process the form data and generate the forecast
    if eda_flag == "True":
        return redirect('/eda')

    else:
        return redirect('/forecast_start')


@app.route("/eda", methods=['GET'])
def eda_start():
    response_1 = {}
    return render_template('Shell_Eda.html', response=response_1)


@app.route("/eda", methods=['POST'])
def eda():
    # Handle the form submission in Shell_Eda.html
    # Retrieve the form data and perform the desired operations
    csv_file = request.files['csv-file']
    eda_dataset = pd.read_csv(csv_file)
    eda_dataset.to_csv('static/eda_dataset.csv')

    # Process the data and perform the necessary operations
    shell_data_eda = Shell_EDA(filepath="static",
                               train_set_name="eda_dataset.csv",
                               column_name="Net Cashflow from Operations",
                               start_date="2021-01-01",
                               end_date=None,
                               output_path=f"static", )
    shell_data_eda.read_data()
    acf_pacf_path = shell_data_eda.acf_pacf()
    seasonal_decomp_path = shell_data_eda.decompose()
    info_path_1 = shell_data_eda.information()
    mean_std_plot_path = shell_data_eda.mean_std_plot()
    df_test_path = shell_data_eda.DF_test()
    seasonal_decomp_path_2 = shell_data_eda.seasonal_decomp()
    order_1_path, order_2_path, order_3_path = shell_data_eda.ordered_pacf_acf()

    response = {
        'acfpacfPlotPath': acf_pacf_path,
        'seasonalDecompPlotPath': seasonal_decomp_path,
        'pandasInfoDescribe': info_path_1,
        'meanStdPlotPath': mean_std_plot_path,
        'dickeyFullerTest': df_test_path,
        'seasonalDecompPlotPath2': seasonal_decomp_path_2,
        'orderedPacfAcfPlotsPath': [order_1_path, order_2_path, order_3_path]
    }
    with open('static/response.json', 'w') as fp:
        json.dump(response, fp)
    return render_template('Shell_Eda.html', response=response)


@app.route('/forecast_start', methods=['GET'])
def index():
    response_output = {}
    return render_template('Shell_Forecast_Finish.html', response=response_output)


@app.route("/forecast_start", methods=['POST'])
def forecast_start():
    # render_template('Shell_Forecast_Finish.html')
    global console_output, progress, is_training_complete
    # Reset training progress
    console_output = []
    progress = 0
    is_training_complete = 0
    output_buffer = io.StringIO()
    sys.stdout = output_buffer
    with open('static/full_train.json') as f:
        full_train = json.load(f)

    if full_train["model_selection"] == "ensemble":
        shell_sarimax_class = Shell_SARIMAX(filepath=full_train["filepath"],
                                            train_set_name=full_train["csv_file_name"],
                                            column_name="Net Cashflow from Operations",
                                            start_date=full_train["training_start_date_sarimax"],
                                            end_date=None, prediction_start_date=full_train["test_start_date"],
                                            prediction_end_date=full_train["test_end_date"])

        shell_sarimax_class.create_train_set()
        shell_sarimax_class.seasonal_decomposition()
        shell_sarimax_class.create_train_test_exog_endog()
        if full_train["grid_search_arima"] == "on":
            shell_sarimax_class.SARIMAX_gridsearch()
        else:
            shell_sarimax_class.best_params = list(full_train["sarimax_parameters"].values())

        shell_sarimax_class.SARIMAX_train_test()
        shell_sarimax_class_forecast_output = shell_sarimax_class.Sarimax_Forecast()

        shell_lgbm_class = Shell_LGBM(filepath=full_train["filepath"], train_set_name=full_train["csv_file_name"],
                                      column_name="Net Cashflow from Operations",
                                      start_date=full_train["training_start_date_lightgbm"],
                                      end_date=None,
                                      prediction_start_date=full_train["test_start_date"],
                                      prediction_end_date=full_train["test_end_date"],
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
        if full_train["grid_search_lightgbm"] == "on":
            shell_lgbm_class.LGBM_GridSearch()
        else:
            param_set = {'objective': 'regression',
                         'boosting_type': "goss",
                         'num_leaves': full_train["lightgbm_parameters"]["num_leaves"],
                         'max_depth': full_train["lightgbm_parameters"]["max_depth"],
                         'learning_rate': full_train["lightgbm_parameters"]["learning_rate"],
                         'subsample': full_train["lightgbm_parameters"]["subsample"],
                         'colsample_bytree': full_train["lightgbm_parameters"]["colsample_bytree"],
                         'reg_alpha': full_train["lightgbm_parameters"]["reg_alpha"],
                         'reg_lambda': full_train["lightgbm_parameters"]["reg_lambda"],
                         'n_estimators': full_train["lightgbm_parameters"]["n_estimators"],
                         'random_state': full_train["lightgbm_parameters"]["random_state"],
                         'num_iterations': full_train["lightgbm_parameters"]["num_iterations"]}

            shell_lgbm_class.best_params = param_set
            shell_lgbm_class.Train_LightGBM()

        shell_lgbm_class_forecast_output = shell_lgbm_class.Forecast_LightGBM()
        sarimax_output = pd.DataFrame(columns=["Date", "Net Cashflow from Operations"])
        lgbm_output = pd.DataFrame(columns=["Date", "Net Cashflow from Operations"])

        sarimax_output["Date"] = shell_sarimax_class_forecast_output.index
        sarimax_output["Net Cashflow from Operations"] = shell_sarimax_class_forecast_output.values

        lgbm_output["Date"] = shell_lgbm_class_forecast_output.index
        lgbm_output["Net Cashflow from Operations"] = shell_lgbm_class_forecast_output.values

        shell_sarimax_class_forecast_output.to_csv(f"{shell_sarimax_class.filepath}/submission_sarimax.csv",
                                                   index=False)
        shell_lgbm_class_forecast_output.to_csv(f"{shell_lgbm_class.filepath}/submission_lgbm.csv", index=False)

        shell_ensemble_class = Shell_Ensemble(submission_SARIMAX=shell_sarimax_class_forecast_output,
                                              submission_LGBM=shell_lgbm_class_forecast_output,
                                              file_path=shell_sarimax_class.filepath,
                                              file_name="submission_ensemble.csv")
        shell_ensemble_class.Ensemble()

    elif full_train["model_selection"] == "sarimax":
        shell_sarimax_class = Shell_SARIMAX(filepath=full_train["filepath"],
                                            train_set_name=full_train["csv_file_name"],
                                            column_name="Net Cashflow from Operations",
                                            start_date=full_train["training_start_date_sarimax"],
                                            end_date=None, prediction_start_date=full_train["test_start_date"],
                                            prediction_end_date=full_train["test_end_date"])

        shell_sarimax_class.create_train_set()
        shell_sarimax_class.seasonal_decomposition()
        shell_sarimax_class.create_train_test_exog_endog()
        if full_train["grid_search_arima"] == "on":
            shell_sarimax_class.SARIMAX_gridsearch()
        else:
            shell_sarimax_class.best_params = list(full_train["sarimax_parameters"].values())

        shell_sarimax_class.SARIMAX_train_test()
        shell_sarimax_class_forecast_output = shell_sarimax_class.Sarimax_Forecast()
        sarimax_output = pd.DataFrame(columns=["Date", "Net Cashflow from Operations"])
        sarimax_output["Date"] = shell_sarimax_class_forecast_output.index
        sarimax_output["Net Cashflow from Operations"] = shell_sarimax_class_forecast_output.values
        sarimax_output.to_csv(f"{shell_sarimax_class.filepath}/submission_sarimax.csv",
                              index=False)

    else:
        shell_lgbm_class = Shell_LGBM(filepath=full_train["filepath"], train_set_name=full_train["csv_file_name"],
                                      column_name="Net Cashflow from Operations",
                                      start_date=full_train["training_start_date_lightgbm"],
                                      end_date=None,
                                      prediction_start_date=full_train["test_start_date"],
                                      prediction_end_date=full_train["test_end_date"],
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
        if full_train["grid_search_lightgbm"] == "on":
            shell_lgbm_class.LGBM_GridSearch()
        else:
            param_set = {'objective': 'regression',
                         'boosting_type': "goss",
                         'num_leaves': full_train["lightgbm_parameters"]["num_leaves"],
                         'max_depth': full_train["lightgbm_parameters"]["max_depth"],
                         'learning_rate': full_train["lightgbm_parameters"]["learning_rate"],
                         'subsample': full_train["lightgbm_parameters"]["subsample"],
                         'colsample_bytree': full_train["lightgbm_parameters"]["colsample_bytree"],
                         'reg_alpha': full_train["lightgbm_parameters"]["reg_alpha"],
                         'reg_lambda': full_train["lightgbm_parameters"]["reg_lambda"],
                         'n_estimators': full_train["lightgbm_parameters"]["n_estimators"],
                         'random_state': full_train["lightgbm_parameters"]["random_state"],
                         'num_iterations': full_train["lightgbm_parameters"]["num_iterations"]}

            shell_lgbm_class.best_params = param_set

        shell_lgbm_class_forecast_output = shell_lgbm_class.Forecast_LightGBM()
        lgbm_output = pd.DataFrame(columns=["Date", "Net Cashflow from Operations"])

        lgbm_output["Date"] = shell_lgbm_class_forecast_output.index
        lgbm_output["Net Cashflow from Operations"] = shell_lgbm_class_forecast_output.values
        lgbm_output.to_csv(f"{shell_lgbm_class.filepath}/submission_lgbm.csv", index=False)

    output = output_buffer.getvalue()
    console_output = output.splitlines()
    is_training_complete = 1
    with open(f"static/console_output.txt", "w") as f:
        for item in console_output:
            # write each item on a new line
            f.write("%s\n" % item)
    sys.stdout = sys.__stdout__

    if full_train["model_selection"] == "ensemble":
        file_output = "static/submission_ensemble.csv"

    elif full_train["model_selection"] == "sarimax":
        file_output = "static/submission_sarimax.csv"
    else:
        file_output = "static/submission_lgbm.csv"

    response = {"console_output": "static/console_output.txt",
                "is_training_complete": is_training_complete,
                "csv_url": file_output}
    with open('static/console_response.json', 'w') as fp:
        json.dump(response, fp)
    return render_template('Shell_Forecast_Finish.html', response=response)


if __name__ == '__main__':
    app.run()
