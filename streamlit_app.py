import os
import time

import pandas as pd
import streamlit as st

from Shell_EDA import Shell_EDA
from Shell_Ensemble import Shell_Ensemble
from Shell_LightGBM import Shell_LGBM
from Shell_SARIMAX import Shell_SARIMAX


def main():
    main_c1, main_c2, main_c3 = st.columns([1, 5, 1])
    with main_c2:
        image_c1, image_c2, image_c3 = st.columns([1, 3, 1])
        title_c1, title_c2, title_c3 = st.columns([1, 5, 1])
        sb_c1, sb_c2, sb_c3 = st.columns([1, 3, 1])
        with image_c2:
            st.image(
                "https://github.com/YigitTurali/Shell_App/blob/2448bc79211378ab8717abcf56b78ca8ba03a77f/static/shell_logo.png?raw=true",
                width=200, use_column_width="auto")
        with title_c2:
            st.title("Royal Dutch Shell")
        with sb_c2:
            st.subheader("Cash Flow Forecasting")

        st.write("Please provide the following information to generate a cash flow forecast:")
        uploaded_file = st.file_uploader("Choose a CSV file")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            df.to_csv('static/eda_dataset.csv')

        with st.container():
            ms = ""
            file_output = pd.DataFrame()
            ms_c1, ms_c2, ms_c3 = st.columns([1, 2, 1])
            with ms_c2:
                model_selection = st.selectbox("Model Selection:", ["", "SARIMAX", "LightGBM", "Ensemble"])

            if model_selection == "Ensemble":
                col1_param, col2_param = st.columns(2)
                if st.checkbox("LightGBM Grid Search"):
                    lgbm_grid_search = True
                else:
                    lgbm_grid_search = False
                    with col1_param:
                        st.write("LightGBM Parameters:")
                        num_leaves = st.text_input("num_leaves:")
                        max_depth = st.text_input("max_depth:")
                        learning_rate = st.text_input("learning_rate:")
                        subsample = st.text_input("subsample:")
                        colsample_bytree = st.text_input("colsample_bytree:")
                        reg_alpha = st.text_input("reg_alpha:")
                        reg_lambda = st.text_input("reg_lambda:")
                        n_estimators = st.text_input("n_estimators:")
                        random_state = st.text_input("random_state:")
                        num_iterations = st.text_input("num_iterations:")
                        try:
                            lightgbm_parameters = {"num_leaves": int(num_leaves),
                                                   "max_depth": int(max_depth),
                                                   "learning_rate": float(learning_rate),
                                                   "subsample": float(subsample),
                                                   "colsample_bytree": float(colsample_bytree),
                                                   "reg_alpha": float(reg_alpha),
                                                   "reg_lambda": float(reg_lambda),
                                                   "n_estimators": int(n_estimators),
                                                   "random_state": int(random_state),
                                                   "num_iterations": int(num_iterations)}

                        except:
                            pass

                if st.checkbox("SARIMAX Grid Search"):
                    sarimax_grid_search = True
                else:
                    sarimax_grid_search = False
                    with col2_param:
                        st.write("SARIMAX Parameters:")
                        st.write("Order")
                        p = st.text_input("p:")
                        q = st.text_input("q:")
                        d = st.text_input("d:")
                        st.write("Seasonal Order")
                        P = st.text_input("P:")
                        D = st.text_input("D:")
                        Q = st.text_input("Q:")
                        s = st.text_input("s:")
                        try:
                            sarimax_params = [int(p), int(q), int(d), int(P), int(D), int(Q), int(s)]

                        except:
                            pass

            elif model_selection == "LightGBM":
                if st.checkbox("LightGBM Grid Search"):
                    lgbm_grid_search = True
                else:
                    lgbm_grid_search = False
                    st.write("LightGBM Parameters:")
                    num_leaves = st.text_input("num_leaves:")
                    max_depth = st.text_input("max_depth:")
                    learning_rate = st.text_input("learning_rate:")
                    subsample = st.text_input("subsample:")
                    colsample_bytree = st.text_input("colsample_bytree:")
                    reg_alpha = st.text_input("reg_alpha:")
                    reg_lambda = st.text_input("reg_lambda:")
                    n_estimators = st.text_input("n_estimators:")
                    random_state = st.text_input("random_state:")
                    num_iterations = st.text_input("num_iterations:")
                    try:
                        lightgbm_parameters = {"num_leaves": int(num_leaves),
                                               "max_depth": int(max_depth),
                                               "learning_rate": float(learning_rate),
                                               "subsample": float(subsample),
                                               "colsample_bytree": float(colsample_bytree),
                                               "reg_alpha": float(reg_alpha),
                                               "reg_lambda": float(reg_lambda),
                                               "n_estimators": int(n_estimators),
                                               "random_state": int(random_state),
                                               "num_iterations": int(num_iterations)}
                    except:
                        pass
            elif model_selection == "SARIMAX":
                if st.checkbox("SARIMAX Grid Search"):
                    sarimax_grid_search = True
                else:
                    sarimax_grid_search = False
                    st.write("SARIMAX Parameters:")
                    st.write("Order")
                    p = st.text_input("p:")
                    d = st.text_input("d:")
                    q = st.text_input("q:")
                    st.write("Seasonal Order")
                    P = st.text_input("P:")
                    D = st.text_input("D:")
                    Q = st.text_input("Q:")
                    s = st.text_input("s:")
                    try:
                        sarimax_params = [int(p), int(q), int(d), int(P), int(D), int(Q), int(s)]
                    except:
                        pass
            st.write("Training Start/End Dates:")
            col1_date, col2_date = st.columns(2)
            with col1_date:
                training_start_date_sarimax = pd.to_datetime(st.date_input("Training Start Date for SARIMAX:"))
                training_end_date_sarimax = pd.to_datetime(st.date_input("Training End Date for SARIMAX:"))
            with col2_date:
                training_start_date_lightgbm = pd.to_datetime(st.date_input("Training Start Date for LightGBM:"))
                training_end_date_lightgbm = pd.to_datetime(st.date_input("Training End Date for LightGBM:"))

            st.write("Test Start/End Dates:")
            col1_date_test, col2_date_test, col3_date_test = st.columns([1, 3, 1])
            with col2_date_test:
                st.write("Test Start/End Dates:")
                test_start_date = pd.to_datetime(st.date_input("Test Start Date:"))
                test_end_date = pd.to_datetime(st.date_input("Test End Date:"))

            analysis_flag = st.empty()
            b1_test, b2_test, b3_test = st.columns([2, 3, 2])
            with b2_test:
                if st.button("Explanatory Data Analysis"):
                    analysis_flag.value = "EDA"

                if st.button("Forecast the Future Cashflow!"):
                    analysis_flag.value = "Forecast"

        if analysis_flag.value == "EDA":
            # Perform EDA and display results
            shell_data_eda = Shell_EDA(filepath="static",
                                       train_set_name="eda_dataset.csv",
                                       column_name="Net Cashflow from Operations",
                                       start_date="2021-01-01",
                                       end_date=None,
                                       output_path=f"static", )
            shell_data_eda.read_data()
            acf_pacf_path = shell_data_eda.acf_pacf()
            seasonal_decomp_text = shell_data_eda.decompose()
            info_text = shell_data_eda.information()
            mean_std_plot_path = shell_data_eda.mean_std_plot()
            df_test_text = shell_data_eda.DF_test()
            seasonal_decomp_path_2 = shell_data_eda.seasonal_decomp()
            order_1_path, order_2_path, order_3_path = shell_data_eda.ordered_pacf_acf()
            st.title("Explanatory Data Analysis")

            with st.container():
                st.header("ACF/PACF Plot")
                st.image(acf_pacf_path)

            with st.container():
                st.header("Seasonal Decomposition Plot")
                st.text_area("Seasonal Decomposition Plot", seasonal_decomp_text, height=330, label_visibility="hidden")

            with st.container():
                st.header("Information")
                st.text_area("Information", info_text, height=400, label_visibility="hidden")

            with st.container():
                st.header("Mean/Standard Deviation Plot")
                st.image(mean_std_plot_path)

            with st.container():
                st.header("Seasonal Decomposition Plot")
                st.image(seasonal_decomp_path_2)

            with st.container():
                st.header("DF Test")
                st.text_area("DF Test", df_test_text, height=250, label_visibility="hidden")

            with st.container():
                st.header("ACF/PACF Plot with Order 0")
                st.image(order_1_path)
                st.header("ACF/PACF Plot with Order 1")
                st.image(order_2_path)
                st.header("ACF/PACF Plot with Order 2")
                st.image(order_3_path)

        elif analysis_flag.value == "Forecast":

            if model_selection == "SARIMAX":
                ms = "SARIMAX"
            elif model_selection == "LightGBM":
                ms = "LightGBM"
            elif model_selection == "Ensemble":
                ms = "Ensemble"

            if ms == "Ensemble":
                shell_sarimax_class = Shell_SARIMAX(filepath="static",
                                                    train_set_name="train_dataset.csv",
                                                    column_name="Net Cashflow from Operations",
                                                    start_date=training_start_date_sarimax,
                                                    end_date=None, prediction_start_date=test_start_date,
                                                    prediction_end_date=test_end_date)

                shell_sarimax_class.create_train_set()
                shell_sarimax_class.seasonal_decomposition()
                shell_sarimax_class.create_train_test_exog_endog()
                if sarimax_grid_search:
                    st.info('SARIMAX Grid Search Started', icon="ℹ️")
                    time.sleep(0.5)
                    shell_sarimax_class.SARIMAX_gridsearch()
                    st.success('SARIMAX Grid Search Finished', icon="✅")
                    time.sleep(0.5)
                else:
                    st.info('Loading SARIMAX Best Parameters', icon="ℹ️")
                    time.sleep(0.5)
                    shell_sarimax_class.best_params = sarimax_params
                    st.success('Loaded SARIMAX Best Parameters', icon="✅")
                    time.sleep(0.5)

                st.info('SARIMAX Training is Starting', icon="ℹ️")
                time.sleep(0.5)
                shell_sarimax_class.SARIMAX_train_test()
                st.success('SARIMAX is Trained Succesfully!', icon="✅")
                time.sleep(0.5)
                st.info('SARIMAX Forecast Started', icon="ℹ️")
                time.sleep(0.5)
                shell_sarimax_class_forecast_output = shell_sarimax_class.Sarimax_Forecast()
                st.success('SARIMAX Forecast is Succesfull!', icon="✅")
                time.sleep(0.5)

                shell_lgbm_class = Shell_LGBM(filepath="static", train_set_name="train_dataset.csv",
                                              column_name="Net Cashflow from Operations",
                                              start_date=training_start_date_lightgbm,
                                              end_date=None,
                                              prediction_start_date=test_start_date,
                                              prediction_end_date=test_end_date,
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
                if lgbm_grid_search:
                    st.info('LightGBM Grid Search Started', icon="ℹ️")
                    time.sleep(0.5)
                    shell_lgbm_class.LGBM_GridSearch()
                    st.success('LightGBM Grid Search Finished', icon="✅")
                    time.sleep(0.5)
                else:
                    st.info('Loading LightGBM Best Parameters', icon="ℹ️")
                    time.sleep(0.5)

                    param_set = {'objective': 'regression',
                                 'boosting_type': "goss",
                                 'num_leaves': lightgbm_parameters["num_leaves"],
                                 'max_depth': lightgbm_parameters["max_depth"],
                                 'learning_rate': lightgbm_parameters["learning_rate"],
                                 'subsample': lightgbm_parameters["subsample"],
                                 'colsample_bytree': lightgbm_parameters["colsample_bytree"],
                                 'reg_alpha': lightgbm_parameters["reg_alpha"],
                                 'reg_lambda': lightgbm_parameters["reg_lambda"],
                                 'n_estimators': lightgbm_parameters["n_estimators"],
                                 'random_state': lightgbm_parameters["random_state"],
                                 'num_iterations': lightgbm_parameters["num_iterations"]}
                    print(param_set)
                    st.success('Loaded LightGBM Best Parameters', icon="✅")
                    time.sleep(0.5)
                    shell_lgbm_class.best_params = param_set
                    st.info('LightGBM Training is Starting', icon="ℹ️")
                    time.sleep(0.5)
                    shell_lgbm_class.Train_LightGBM()
                    st.success('LightGBM is Trained Succesfully!', icon="✅")
                    time.sleep(0.5)

                st.info('LightGBM Forecast Started', icon="ℹ️")
                time.sleep(0.5)
                shell_lgbm_class_forecast_output = shell_lgbm_class.Forecast_LightGBM()
                st.success('LightGBM Forecast is Succesfull!', icon="✅")
                time.sleep(0.5)

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
                ensemble_output = shell_ensemble_class.Ensemble()

            elif ms == "SARIMAX":
                shell_sarimax_class = Shell_SARIMAX(filepath="static",
                                                    train_set_name="train_dataset.csv",
                                                    column_name="Net Cashflow from Operations",
                                                    start_date=training_start_date_sarimax,
                                                    end_date=None, prediction_start_date=test_start_date,
                                                    prediction_end_date=test_end_date)

                shell_sarimax_class.create_train_set()
                shell_sarimax_class.seasonal_decomposition()
                shell_sarimax_class.create_train_test_exog_endog()
                if sarimax_grid_search:
                    st.info('SARIMAX Grid Search Started', icon="ℹ️")
                    time.sleep(0.5)
                    shell_sarimax_class.SARIMAX_gridsearch()
                    st.success('SARIMAX Grid Search Finished', icon="✅")
                    time.sleep(0.5)
                else:
                    st.info('Loading SARIMAX Best Parameters', icon="ℹ️")
                    time.sleep(0.5)
                    shell_sarimax_class.best_params = sarimax_params
                    st.success('Loaded SARIMAX Best Parameters', icon="✅")
                    time.sleep(0.5)

                st.info('SARIMAX Training is Starting', icon="ℹ️")
                time.sleep(0.5)
                shell_sarimax_class.SARIMAX_train_test()
                st.success('SARIMAX is Trained Succesfully!', icon="✅")
                time.sleep(0.5)
                st.info('SARIMAX Forecast Started', icon="ℹ️")
                time.sleep(0.5)
                shell_sarimax_class_forecast_output = shell_sarimax_class.Sarimax_Forecast()
                st.success('SARIMAX Forecast is Succesfull!', icon="✅")
                time.sleep(0.5)
                sarimax_output = pd.DataFrame(columns=["Date", "Net Cashflow from Operations"])
                sarimax_output["Date"] = shell_sarimax_class_forecast_output.index
                sarimax_output["Net Cashflow from Operations"] = shell_sarimax_class_forecast_output.values
                sarimax_output.to_csv(f"{shell_sarimax_class.filepath}/submission_sarimax.csv",
                                      index=False)


            elif ms == "LightGBM":
                shell_lgbm_class = Shell_LGBM(filepath="static", train_set_name="train_dataset.csv",
                                              column_name="Net Cashflow from Operations",
                                              start_date=training_start_date_lightgbm,
                                              end_date=None,
                                              prediction_start_date=test_start_date,
                                              prediction_end_date=test_end_date,
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
                if lgbm_grid_search:
                    st.info('LightGBM Grid Search Started', icon="ℹ️")
                    time.sleep(0.5)
                    shell_lgbm_class.LGBM_GridSearch()
                    st.success('LightGBM Grid Search Finished', icon="✅")
                    time.sleep(0.5)
                else:
                    st.info('Loading LightGBM Best Parameters', icon="ℹ️")
                    time.sleep(0.5)
                    param_set = {'objective': 'regression',
                                 'boosting_type': "goss",
                                 'num_leaves': lightgbm_parameters["num_leaves"],
                                 'max_depth': lightgbm_parameters["max_depth"],
                                 'learning_rate': lightgbm_parameters["learning_rate"],
                                 'subsample': lightgbm_parameters["subsample"],
                                 'colsample_bytree': lightgbm_parameters["colsample_bytree"],
                                 'reg_alpha': lightgbm_parameters["reg_alpha"],
                                 'reg_lambda': lightgbm_parameters["reg_lambda"],
                                 'n_estimators': lightgbm_parameters["n_estimators"],
                                 'random_state': lightgbm_parameters["random_state"],
                                 'num_iterations': lightgbm_parameters["num_iterations"]}
                    st.success('Loaded LightGBM Best Parameters', icon="✅")
                    time.sleep(0.5)

                    shell_lgbm_class.best_params = param_set
                    st.info('LightGBM Training is Starting', icon="ℹ️")
                    time.sleep(0.5)
                    shell_lgbm_class.Train_LightGBM()
                    st.success('LightGBM is Trained Succesfully!', icon="✅")
                    time.sleep(0.5)

                st.info('LightGBM Forecast Started', icon="ℹ️")
                time.sleep(0.5)
                shell_lgbm_class_forecast_output = shell_lgbm_class.Forecast_LightGBM()
                st.success('LightGBM Forecast is Succesfull!', icon="✅")
                time.sleep(0.5)
                lgbm_output = pd.DataFrame(columns=["Date", "Net Cashflow from Operations"])

                lgbm_output["Date"] = shell_lgbm_class_forecast_output.index
                lgbm_output["Net Cashflow from Operations"] = shell_lgbm_class_forecast_output.values
                lgbm_output.to_csv(f"{shell_lgbm_class.filepath}/submission_lgbm.csv", index=False)

            if ms == "Ensemble":
                file_output = ensemble_output
            elif ms == "SARIMAX":
                file_output = sarimax_output
            elif ms == "LightGBM":
                file_output = lgbm_output

            @st.cache_data
            def convert_df(df):
                # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv(index=False).encode('utf-8')

            def download_button_clicked():
                st.success('Downloaded Succesfully!', icon="✅")
                time.sleep(0.5)
                st.balloons()
                files_to_delete = ["static/eda_dataset.csv", "static/train_dataset.csv",
                                   "static/submission_sarimax.csv", "static/submission_lgbm.csv",
                                   "static/submission_ensemble.csv", "static/acf_pacf_plot.png",
                                   "static/mean_std_plot.png",
                                   "static/pacf_acf_order_0.png", "static/pacf_acf_order_1.png",
                                   "static/pacf_acf_order_2.png",
                                   "static/seasonal_decomp.png", "static/train_endog.csv", "static/train_exog.csv"]
                for file_path in files_to_delete:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        print(f"Deleted file: {file_path}")
                    else:
                        print(f"File not found: {file_path}")
                forecast_csv = pd.DataFrame(columns=["Date", "Net Cashflow from Operations"])

            forecast_csv = convert_df(file_output)
            with b2_test:
                st.download_button(
                    label="Download the Forecast as CSV",
                    data=forecast_csv,
                    file_name='forecast.csv',
                    mime='csv',
                    on_click=download_button_clicked,
                )


if __name__ == "__main__":
    main()
