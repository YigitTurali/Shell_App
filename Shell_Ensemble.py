import pandas as pd
class Shell_Ensemble:
    def __init__(self, submission_SARIMAX, submission_LGBM,file_path,file_name):
        self.submission_SARIMAX = submission_SARIMAX
        self.submission_LGBM = submission_LGBM
        self.filepath = file_path
        self.file_name = file_name

    def Ensemble(self):
        sub_ensemble = (self.submission_SARIMAX+self.submission_LGBM)/2
        ensemble_output = pd.DataFrame(columns=["Date", "Net Cashflow from Operations"])
        ensemble_output["Date"] = sub_ensemble.index
        ensemble_output["Net Cashflow from Operations"] = sub_ensemble.values

        ensemble_output.to_csv(f"{self.filepath}/{self.file_name}",index=False)
        return ensemble_output
