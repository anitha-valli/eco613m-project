# Install openpyxl to read .xlsx, required python libraries and R packages

# Please ensure x64, x86 Visual C++ are present and updated

# pip install openpyxl statsmodels rpy2 scikit-learn seaborn
# run the above line in the terminal

# install.packages("forecast")
# install.packages("quantreg")
# run the above two lines in the terminal under the folder where R is present

# -------------- Import the relevant libraries ---------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss

# =========***========= Modify the following as needed (R-4.5.x) - =========***=========
r_home = r"C:\Program Files\R\R-4.5.3" 
os.environ['R_HOME'] = r_home
os.environ['PATH'] = (r_home + r"\bin\x64;" + r_home + r"\bin;" + os.environ['PATH'])
# --------------------------------------------------------------------------------------


import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri

forecast_pkg = importr('forecast')
base = importr('base')
stats_r = importr("stats")

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def mean_fractional_bias(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = (y_pred + y_true) / 2
    mask = denom != 0
    return np.mean((y_pred[mask] - y_true[mask]) / denom[mask])

def mean_fractional_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = (y_pred + y_true) / 2
    mask = denom != 0
    return np.mean(np.abs(y_pred[mask] - y_true[mask]) / denom[mask])



# --------------------- DATA IMPORT/PROCESSING ----------------------


files = ["2020.xlsx", "2021.xlsx", "2022.xlsx", "2023.xlsx", "2024.xlsx"]
cols = ["Date", "44T"]

dfs = []

for f in files:
    df = pd.read_excel(f, parse_dates = ["Date"])
    df = df.rename(columns = str.strip)
    df = df[cols]

    df["44T"] = pd.to_numeric(df["44T"], errors = "coerce")
    df = df.dropna(subset = ["Date", "44T"])

    dfs.append(df)

merged_df = pd.concat(dfs, ignore_index = True)

merged_df["Date"] = pd.to_datetime(merged_df["Date"], errors = "coerce")
merged_df["Date"] = merged_df["Date"].dt.date
merged_df = merged_df.sort_values("Date")

#print(merged_df)

merged_df.to_excel("Full_Data.xlsx", index = False)

ts_data = merged_df.set_index("Date")["44T"]

train = ts_data[:1447]
test = ts_data[1447:]

# For verification
#print("Train size:", len(train))
#print("Test size:", len(test))

plt.figure(figsize=(16, 4))
plt.plot(merged_df["Date"], merged_df["44T"], marker = 'o', linestyle = '-', markersize = 3, linewidth = 1)
plt.xlabel("Date")
plt.ylabel("PM2.5 Concentration")
plt.grid(True)
plt.gcf().autofmt_xdate()
plt.show()

# --------------------------- Supplementary Figures ---------------------------------

# By Year

statistics = []

for i, df in enumerate(dfs):
    stats = {
        "File": files[i].replace(".xlsx", ""),
        "Mean": df["44T"].mean(),
        "SD": df["44T"].std(),
        "Min": df["44T"].min(),
        "Max": df["44T"].max(),
    }
    statistics.append(stats)

stats_df = pd.DataFrame(statistics)
stats_df = stats_df.set_index("File").T

print(stats_df)

stats_df.T.plot(kind = "bar")

plt.xlabel("Year")
plt.ylabel("PM2.5 Concentration")
plt.legend()
plt.xticks(rotation = 0)
plt.tight_layout()

plt.show()

# By Month

merged_df["Month"] = pd.to_datetime(merged_df["Date"]).dt.month

month_stats = merged_df.groupby("Month")["44T"].agg(Mean = "mean", SD = "std", Min = "min", Max="max").reset_index()

month_stats = month_stats.set_index("Month")

print(month_stats)

month_stats.plot(kind = "bar")

plt.xlabel("Month")
plt.ylabel("PM2.5 Concentration")
plt.xticks(rotation = 0)
plt.tight_layout()
plt.show()


# ------------------ Distribution Plot --------------------

import seaborn as sb

plt.hist(merged_df["44T"], bins = 25, density = True, alpha = 0.5)

sb.kdeplot(merged_df["44T"], color = "red", linewidth = 2)
plt.axvline(merged_df["44T"].median(), color = "red", linestyle = "--", linewidth = 2)
plt.xlabel("PM2.5 Concentration")
plt.ylabel("Density")
plt.grid(True)
plt.show()


# -------------------- Stationarity tests -----------------------

result_adf = adfuller(train)
print("\nADF Test:", result_adf)

result_kpss = kpss(train, regression='c', nlags="auto")
print("\nKPSS Test:", result_kpss)



# ======================================== PART I - ARIMA ======================================

# ---------------------- R ARIMA MODEL ----------------------

r_train = ro.FloatVector(train.values)
ts = ro.r.ts(r_train, frequency = 365)
ts1 = ro.r.ts(r_train)
model = forecast_pkg.auto_arima(ts)
print("\n", ro.r.summary(model))
order = ro.r('arimaorder')(model)
print("\nSelected ARIMA order using auto-arima:", list(order))

# ARIMA(1,1,2)
manual_model = forecast_pkg.Arima(ts, order=ro.IntVector([1,1,2]), include_drift = True)
manual_model1 = forecast_pkg.Arima(ts1, order=ro.IntVector([1,1,2]))
print("\n============ R ARIMA Model Summary ============")
print("\n", ro.r.summary(manual_model))
print("\n===============================================\n")

# ----------------------- R FORECAST ------------------------

stlf = ro.r['stlf']
fc_stl = stlf(ts, h = len(test))

# Convert forecast back to Python
with localconverter(ro.default_converter + pandas2ri.converter):
    forecast_r = ro.conversion.rpy2py(fc_stl.rx2('mean'))

y_true = test.values
y_pred = np.array(forecast_r)


# --------------------- R PLOT FORECAST ----------------------

plt.figure(figsize=(16, 4))
plt.plot(test.index, y_true, label = "Actual")
plt.plot(test.index, y_pred, label = "Forecast")
plt.xlabel("Date")
plt.ylabel("PM2.5")
plt.title("R ARIMA vs Actual")
plt.legend()
plt.grid(True)
plt.show(block = True)


# -------------------------- Metrics --------------------------

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
mfb = mean_fractional_bias(y_true, y_pred)
mfe = mean_fractional_error(y_true, y_pred)

print("\n====== R ARIMA Performance ======")
print("RMSE:", rmse)
print("MAE :", mae)
print("MAPE:", mape)
print("R²  :", r2)
print("MFB :", mfb)
print("MFE :", mfe)
print("=================================\n")

# --------------------- RESIDUAL DIAGNOSTICS -------------------

print("===== Residual Diagnostics =====")
ro.r('checkresiduals')(manual_model1)
print("================================\n")

# ------------------------ STAM ARIMA MODEL --------------------------

# ARIMA Model Selection
p, q = 3, 3
arima_results = []
for p1 in range(p):
    for q1 in range(q):
        try:
            model = ARIMA(train, order = (p1, 1, q1), trend = 'n')
            model_fit = model.fit()
            model_d = ARIMA(train, order = (p1, 1, q1), trend = 't')
            model_fit_d = model_d.fit()
            forecast = model_fit.forecast(steps = len(test))
            y_true = test.values
            forecast_d = model_fit_d.forecast(steps = len(test))

            # Calculate RMSE and MAPE
            rmse = np.sqrt(np.mean((y_true - forecast)**2))
            mape = mean_absolute_percentage_error(y_true, forecast)
            rmse_d = np.sqrt(np.mean((y_true - forecast_d)**2))
            mape_d = mean_absolute_percentage_error(y_true, forecast_d)

            print(p1, "1", q1, "AIC =", model_fit.aic, "BIC =", model_fit.bic, "HQIC =", model_fit.hqic, "RMSE =", rmse, "MAPE =", mape)
            print(p1, "1", q1, " (with drift) ", "AIC =", model_fit_d.aic, "BIC =", model_fit_d.bic, "HQIC =", model_fit_d.hqic, "RMSE =", rmse_d, "MAPE =", mape_d)
            
            arima_results.append({"model": f"({p1},1,{q1})", "AIC": model_fit.aic, "BIC": model_fit.bic,      "type": ""})

            arima_results.append({"model": f"({p1},1,{q1})", "AIC": model_fit_d.aic, "BIC": model_fit_d.bic,              "type": "with drift"})

        except:
            continue

# ---------------------------- PLOT AIC/BIC --------------------------------

arima_df = pd.DataFrame(arima_results)

arima_df["label"] = arima_df["model"] + " " + arima_df["type"]

plot_df = arima_df
plt.figure()
y = np.arange(len(plot_df))
plt.barh(y - 0.2, plot_df["AIC"], height = 0.4, label = "AIC", color = "firebrick")
plt.barh(y + 0.2, plot_df["BIC"], height = 0.4, label = "BIC", color = "steelblue")
plt.yticks(y, plot_df["label"])
plt.xlim(7400, 7750)
plt.legend()
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Best Fit ARIMA Model 
model = ARIMA(train, order = (1, 1, 2), trend = 't')
model_fit = model.fit()
print(model_fit.summary())

residuals = model_fit.resid

# ----------------------- STAM FORECAST -----------------------

from statsmodels.tsa.forecasting.stl import STLForecast

stl_model = STLForecast(train, ARIMA, model_kwargs = {"order": (1,1,2)}, period = 365)
stl_fit = stl_model.fit()

n = len(test)
forecast_stl = stl_fit.forecast(n)

y_pred_s = forecast_stl

# ----------------- Metrics --------------------

rmse_s = np.sqrt(mean_squared_error(y_true, y_pred_s))
mae_s = mean_absolute_error(y_true, y_pred_s)
mape_s = mean_absolute_percentage_error(y_true, y_pred_s)
r2_s = r2_score(y_true, y_pred_s)
mfb_s = mean_fractional_bias(y_true, y_pred_s)
mfe_s = mean_fractional_error(y_true, y_pred_s)

print("\n====== STAM ARIMA Performance ======")
print("RMSE:", rmse_s)
print("MAE :", mae_s)
print("MAPE:", mape_s)
print("R²  :", r2_s)
print("MFB :", mfb_s)
print("MFE :", mfe_s)
print("====================================\n")


# --------------------- STAM PLOT FORECAST ---------------------

plt.figure(figsize=(16, 4))
plt.plot(test.index, y_true, label = "Actual")
plt.plot(test.index, y_pred_s, label = "Forecast")
plt.xlabel("Date")
plt.ylabel("PM2.5")
plt.title("STAM ARIMA vs Actual")
plt.legend()
plt.grid(True)
plt.show()


# ======================================== PART II - ANN ======================================

stlm = ro.r['stlm']
forecast_ann = ro.r['forecast']
nnetar = ro.r['nnetar']

# New nnetar required to specify structure

ro.r("""
nnetar1 <- function(x) {
    forecast::nnetar(x, p = 1, size = 1, repeats = 20)
}
""")

nnetar1 = ro.r['nnetar1']

model_ann = stlm(ts, modelfunction = nnetar1)
nnar_model = model_ann.rx2("model")
print(nnar_model)

fc_stl_ann = forecast_ann(model_ann, h = len(test))

with localconverter(ro.default_converter + pandas2ri.converter):
    forecast_values_ann = ro.conversion.rpy2py(fc_stl_ann.rx2('mean'))


y_pred_ann = np.array(forecast_values_ann)

# ---------------------- ANN PLOT FORECAST ---------------------

plt.figure(figsize=(16, 4))
plt.plot(test.index, y_true, label = "Actual")
plt.plot(test.index, y_pred_ann, label = "Forecast")
plt.xlabel("Date")
plt.ylabel("PM2.5")
plt.title("ANN Forecast vs Actual")
plt.legend()
plt.grid(True)
plt.show()

# ----------------- Metrics ----------------------

rmse_ann = np.sqrt(mean_squared_error(y_true, y_pred_ann))
mae_ann = mean_absolute_error(y_true, y_pred_ann)
mape_ann = mean_absolute_percentage_error(y_true, y_pred_ann)
r2_ann = r2_score(y_true, y_pred_ann)
mfb_ann = mean_fractional_bias(y_true, y_pred_ann)
mfe_ann = mean_fractional_error(y_true, y_pred_ann)

print("\n======= ANN Performance =======")
print("RMSE:", rmse_ann)
print("MAE :", mae_ann)
print("MAPE:", mape_ann)
print("R²  :", r2_ann)
print("MFB :", mfb_ann)
print("MFE :", mfe_ann)
print("===============================\n")


# ================================= PART III - ARIMA+ANN HYBRID =================================

arima_fits = ro.r('fitted')(manual_model)

with localconverter(ro.default_converter + pandas2ri.converter):
    arima_fitted = np.array(ro.conversion.rpy2py(arima_fits))

arima_fitted = pd.Series(arima_fitted, index=train.index[-len(arima_fitted):])

residuals_h = train.loc[arima_fitted.index] - arima_fitted
residuals_h = residuals_h.dropna()

residuals_r = ro.FloatVector(residuals_h.values)
residuals_ts = ro.r.ts(residuals_r)

model_resid_ann = nnetar1(residuals_ts)

fc_resid_ann = forecast_ann(model_resid_ann, h = len(test))

with localconverter(ro.default_converter + pandas2ri.converter):
    forecast_residuals_ann = ro.conversion.rpy2py(fc_resid_ann.rx2('mean'))


residuals_fc = np.array(forecast_residuals_ann)

y_pred_hr = y_pred + residuals_fc
y_pred_hstam = y_pred_s + residuals_fc

# ---------------------- HYBRID PLOT FORECAST ---------------------

plt.figure(figsize=(16, 4))
plt.plot(test.index, y_true, label = "Actual")
plt.plot(test.index, y_pred_hr, label = "R Forecast")
plt.plot(test.index, y_pred_hstam, label = "STAM Forecast")
plt.xlabel("Date")
plt.ylabel("PM2.5")
plt.title("ARIMA-ANN vs Actual")
plt.legend()
plt.grid(True)
plt.show()


# ----------------- Metrics ----------------------

rmse_h = np.sqrt(mean_squared_error(y_true, y_pred_hr))
mae_h = mean_absolute_error(y_true, y_pred_hr)
mape_h = mean_absolute_percentage_error(y_true, y_pred_hr)
r2_h = r2_score(y_true, y_pred_hr)
mfb_h = mean_fractional_bias(y_true, y_pred_hr)
mfe_h = mean_fractional_error(y_true, y_pred_hr)

print("\n===== ARIMA-ANN Performance ======")
print("RMSE:", rmse_h)
print("MAE :", mae_h)
print("MAPE:", mape_h)
print("R²  :", r2_h)
print("MFB :", mfb_h)
print("MFE :", mfe_h)
print("=====================================\n")

sigma2_h = np.var(residuals_h)

print(f"Hybrid residual variance is {sigma2_h}")



# ================================= PART IV - HYBRID+REG =================================

from rpy2.robjects import globalenv

arima_fits = ro.r('fitted')(manual_model)

with localconverter(ro.default_converter + pandas2ri.converter):
    arima_fit = np.array(ro.conversion.rpy2py(arima_fits))

model_ann = nnetar1(residuals_ts)

ann_fits = ro.r('fitted')(model_ann)

with localconverter(ro.default_converter + pandas2ri.converter):
    ann_fit = np.array(ro.conversion.rpy2py(ann_fits))

y_train = np.array(train.values)

arima_series = pd.Series(arima_fit)
ann_series = pd.Series(ann_fit)

min_len = min(len(train), len(arima_series), len(ann_series))

y_aligned = train.iloc[-min_len:]
arima_aligned = arima_series.iloc[-min_len:]
ann_aligned = ann_series.iloc[-min_len:]

arima_aligned.index = y_aligned.index
ann_aligned.index = y_aligned.index

df_reg = pd.DataFrame({"y": y_aligned, "arima": arima_aligned, "ann": ann_aligned}).dropna()

with localconverter(ro.default_converter + pandas2ri.converter):
    globalenv["df_reg"] = df_reg

ro.r("""
model_reg <- lm(y ~ arima + ann - 1, data = df_reg)
summary(model_reg)
""")

print("\n===== ARIMA–ANN-REG Results =====")
reg_summary = ro.r("summary(model_reg)")
print(reg_summary)
print("==================================\n")

ro.r("""
df_reg$ols_pred <- predict(model_reg, newdata = df_reg)
""")

reg_coef = ro.r("coef(model_reg)")

with localconverter(ro.default_converter + pandas2ri.converter):
    reg_coef = np.array(reg_coef)

beta_arima_reg = reg_coef[0]
beta_annresid_reg = reg_coef[1]

y_pred_reg = beta_arima_reg * y_pred + beta_annresid_reg * residuals_fc


plt.figure(figsize=(16, 4))
plt.plot(test.index, y_true, label = "Actual")
plt.plot(test.index, y_pred_reg, label = "REG Forecast")
plt.xlabel("Date")
plt.ylabel("PM2.5")
plt.title("ARIMA-ANN-REG vs Actual")
plt.legend()
plt.grid(True)
plt.show()

# ----------------- Metrics ----------------------

rmse_reg = np.sqrt(mean_squared_error(y_true, y_pred_reg))
mae_reg = mean_absolute_error(y_true, y_pred_reg)
mape_reg = mean_absolute_percentage_error(y_true, y_pred_reg)
r2_reg = r2_score(y_true, y_pred_reg)
mfb_reg = mean_fractional_bias(y_true, y_pred_reg)
mfe_reg = mean_fractional_error(y_true, y_pred_reg)


print("\n===== ARIMA-ANN-REG Performance =====")
print("RMSE:", rmse_reg)
print("MAE :", mae_reg)
print("MAPE:", mape_reg)
print("R²  :", r2_reg)
print("MFB :", mfb_reg)
print("MFE :", mfe_reg)
print("=====================================\n")

# ================================= PART V - HYBRID+QREG =================================

rq = importr("quantreg")

ro.r("""
model_qreg <- rq(y ~ arima + ann - 1, tau = 0.5, data = df_reg)
summary(model_qreg)
""")

qr_summary = ro.r("summary(model_qreg, se = 'boot')")

print("\n===== ARIMA–ANN-QREG Results =====")
print("\nQuantile Regression Summary:\n", qr_summary)
print("==================================\n")

qreg_coef = ro.r("coef(model_qreg)")

with localconverter(ro.default_converter + pandas2ri.converter):
    qreg_coef = np.array(qreg_coef)

beta_arima_qreg = qreg_coef[0]
beta_annresid_qreg = qreg_coef[1]

y_pred_qreg = beta_arima_qreg * y_pred + beta_annresid_qreg * residuals_fc


plt.figure(figsize=(16, 4))
plt.plot(test.index, y_true, label = "Actual")
plt.plot(test.index, y_pred_qreg, label = "QREG Forecast")
plt.xlabel("Date")
plt.ylabel("PM2.5")
plt.title("ARIMA-ANN-QREG vs Actual")
plt.legend()
plt.grid(True)
plt.show()

# ----------------- Metrics ----------------------

rmse_qreg = np.sqrt(mean_squared_error(y_true, y_pred_qreg))
mae_qreg = mean_absolute_error(y_true, y_pred_qreg)
mape_qreg = mean_absolute_percentage_error(y_true, y_pred_qreg)
r2_qreg = r2_score(y_true, y_pred_qreg)
mfb_qreg = mean_fractional_bias(y_true, y_pred_qreg)
mfe_qreg = mean_fractional_error(y_true, y_pred_qreg)


print("\n===== ARIMA-ANN-QREG Performance =====")
print("RMSE:", rmse_qreg)
print("MAE :", mae_qreg)
print("MAPE:", mape_qreg)
print("R²  :", r2_qreg)
print("MFB :", mfb_qreg)
print("MFE :", mfe_qreg)
print("====================================\n")

print(df_reg.corr())

# ----------------------- Comparing other Quantiles --------------------------

ro.r("""
qr_25 <- rq(y ~ arima + ann - 1, tau = 0.25, data = df_reg)
qr_50 <- rq(y ~ arima + ann - 1, tau = 0.5, data = df_reg)
qr_75 <- rq(y ~ arima + ann - 1, tau = 0.75, data = df_reg)

summary(qr_25, se="boot")
summary(qr_50, se="boot")
summary(qr_75, se="boot")
""")

qr_summary25 = ro.r("summary(qr_25, se = 'boot')")
qr_summary50 = ro.r("summary(qr_50, se = 'boot')")
qr_summary75 = ro.r("summary(qr_75, se = 'boot')")

print("\n===== ARIMA–ANN-QREG Results =====")
print("\nQuantile Regression Summary:\n", qr_summary25)
print("\nQuantile Regression Summary:\n", qr_summary50)
print("\nQuantile Regression Summary:\n", qr_summary75)
print("==================================\n")


# ================================= PART VI - COMPARISON OF ALL MODELS =================================

# ------------------- TABLE 5 -------------------

table_5 = pd.DataFrame({"Model": ["ARIMA", "ANN", "ARIMA-ANN", "ARIMA-ANN-REG", "ARIMA-ANN-QREG"],
    "RMSE": [rmse, rmse_ann, rmse_h, rmse_reg, rmse_qreg],
    "MAE":  [mae, mae_ann, mae_h, mae_reg, mae_qreg],
    "MAPE": [mape, mape_ann, mape_h, mape_reg, mape_qreg],
    "R2":   [r2, r2_ann, r2_h, r2_reg, r2_qreg],
    "MFB":  [mfb, mfb_ann, mfb_h, mfb_reg, mfb_qreg],
    "MFE":  [mfe, mfe_ann, mfe_h, mfe_reg, mfe_qreg]
})

table_5 = table_5.set_index("Model")

print("\n============== SUMMARY OF MODEL ACCURACY ==============\n")
print(table_5)
print("\n=======================================================\n")

# ------------------- FIG 11 --------------------

def pc(comp, base):
    return ((comp - base) / base) * 100

rmse_vals = table_5["RMSE"]
mae_vals  = table_5["MAE"]
mape_vals = table_5["MAPE"]

comparison = {
    "ARIMA-ANN vs ARIMA(1,1,2)": [
        pc(rmse_vals["ARIMA-ANN"], rmse_vals["ARIMA"]),
        pc(mae_vals["ARIMA-ANN"], mae_vals["ARIMA"]),
        pc(mape_vals["ARIMA-ANN"], mape_vals["ARIMA"])
    ],
    
    "ARIMA-ANN vs ANN": [
        pc(rmse_vals["ARIMA-ANN"], rmse_vals["ANN"]),
        pc(mae_vals["ARIMA-ANN"], mae_vals["ANN"]),
        pc(mape_vals["ARIMA-ANN"], mape_vals["ANN"])
    ],
    
    "ARIMA-ANN-REG vs ARIMA-ANN": [
        pc(rmse_vals["ARIMA-ANN-REG"], rmse_vals["ARIMA-ANN"]),
        pc(mae_vals["ARIMA-ANN-REG"], mae_vals["ARIMA-ANN"]),
        pc(mape_vals["ARIMA-ANN-REG"], mape_vals["ARIMA-ANN"])
    ],
    
    "ARIMA-ANN-QREG vs ARIMA-ANN": [
        pc(rmse_vals["ARIMA-ANN-QREG"], rmse_vals["ARIMA-ANN"]),
        pc(mae_vals["ARIMA-ANN-QREG"], mae_vals["ARIMA-ANN"]),
        pc(mape_vals["ARIMA-ANN-QREG"], mape_vals["ARIMA-ANN"])
    ]
}

labels = list(comparison.keys())
data = np.array(list(comparison.values()))

x = np.arange(len(labels))
width = 0.12

plt.figure(figsize = (18, 8))

plt.bar(x - width, data[:,0], width, label="RMSE")
plt.bar(x,         data[:,1], width, label="MAE")
plt.bar(x + width, data[:,2], width, label="MAPE (%)")

plt.xticks(x, labels)
plt.ylabel("Percentage change (%)")
plt.axhline(0)

plt.legend()
plt.tight_layout()
plt.subplots_adjust(left=0.1)
plt.show()
