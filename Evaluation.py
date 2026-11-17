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



# ================================================================================================================
# ============================================= EVALUATION PART I ================================================
# =========================================== DIFFERENT TIME PERIOD ==============================================



# --------------------- DATA IMPORT/PROCESSING ----------------------


files = ["2016.xlsx", "2017.xlsx", "2018.xlsx", "2019.xlsx", "2020.xlsx", "2021.xlsx", "2022.xlsx"]
cols = ["Date", "44T"]

dfs = []

for f in files:
    df = pd.read_excel(f, parse_dates = ["Date"])
    df = df.rename(columns = str.strip)
    df = df[cols]

    df["44T"] = pd.to_numeric(df["44T"], errors = "coerce")
    df = df.dropna(subset = ["Date", "44T"])

    if "2022" in f:
        df = df[df["Date"] <= "2022-06-30"]

    dfs.append(df)

merged_df = pd.concat(dfs, ignore_index = True)

merged_df["Date"] = pd.to_datetime(merged_df["Date"], errors = "coerce")
merged_df["Date"] = merged_df["Date"].dt.date
merged_df = merged_df.sort_values("Date")

print(merged_df)

merged_df.to_excel("Evaluation_File.xlsx", index = False)

ts_data = merged_df.set_index("Date")["44T"]

split_index = int(len(ts_data) * 0.8)

train = ts_data[:split_index]
test = ts_data[split_index:]

# For verification
print("Train size:", len(train))
print("Test size:", len(test))


plt.figure(figsize=(16, 4))
plt.plot(range(len(merged_df)), merged_df["44T"], marker = 'o', linestyle = '-', markersize = 3, linewidth = 1)
plt.xlabel("Date")
plt.ylabel("PM2.5 Concentration")
plt.grid(True)
plt.gcf().autofmt_xdate()
plt.show()

# ======================================== PART I - ARIMA ======================================

# ---------------------- R ARIMA MODEL ----------------------

r_train = ro.FloatVector(train.values)
ts = ro.r.ts(r_train, frequency = 365)

# ARIMA(1,1,2)
manual_model = forecast_pkg.Arima(ts, order=ro.IntVector([1,1,2]), include_drift = True)

# ----------------------- R FORECAST ------------------------

stlf = ro.r['stlf']
fc_stl = stlf(ts, h = len(test))

# Convert forecast back to Python
with localconverter(ro.default_converter + pandas2ri.converter):
    forecast_r = ro.conversion.rpy2py(fc_stl.rx2('mean'))

y_true = test.values
y_pred = np.array(forecast_r)

# -------------------------- Metrics --------------------------

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
mfb = mean_fractional_bias(y_true, y_pred)
mfe = mean_fractional_error(y_true, y_pred)


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


# ----------------- Metrics ----------------------

rmse_ann = np.sqrt(mean_squared_error(y_true, y_pred_ann))
mae_ann = mean_absolute_error(y_true, y_pred_ann)
mape_ann = mean_absolute_percentage_error(y_true, y_pred_ann)
r2_ann = r2_score(y_true, y_pred_ann)
mfb_ann = mean_fractional_bias(y_true, y_pred_ann)
mfe_ann = mean_fractional_error(y_true, y_pred_ann)


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

# ----------------- Metrics ----------------------

rmse_h = np.sqrt(mean_squared_error(y_true, y_pred_hr))
mae_h = mean_absolute_error(y_true, y_pred_hr)
mape_h = mean_absolute_percentage_error(y_true, y_pred_hr)
r2_h = r2_score(y_true, y_pred_hr)
mfb_h = mean_fractional_bias(y_true, y_pred_hr)
mfe_h = mean_fractional_error(y_true, y_pred_hr)


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

# ----------------- Metrics ----------------------

rmse_reg = np.sqrt(mean_squared_error(y_true, y_pred_reg))
mae_reg = mean_absolute_error(y_true, y_pred_reg)
mape_reg = mean_absolute_percentage_error(y_true, y_pred_reg)
r2_reg = r2_score(y_true, y_pred_reg)
mfb_reg = mean_fractional_bias(y_true, y_pred_reg)
mfe_reg = mean_fractional_error(y_true, y_pred_reg)


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


# ----------------- Metrics ----------------------

rmse_qreg = np.sqrt(mean_squared_error(y_true, y_pred_qreg))
mae_qreg = mean_absolute_error(y_true, y_pred_qreg)
mape_qreg = mean_absolute_percentage_error(y_true, y_pred_qreg)
r2_qreg = r2_score(y_true, y_pred_qreg)
mfb_qreg = mean_fractional_bias(y_true, y_pred_qreg)
mfe_qreg = mean_fractional_error(y_true, y_pred_qreg)



# ================================= PART VI - COMPARISON OF ALL MODELS =================================

import seaborn as sb

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

print("\n========== SUMMARY OF MODEL ACCURACY ==========\n")
print(table_5)
print("\n========================================================\n")

# ------------------- BAR PLOT FOR METRICS -------------------

import matplotlib.patches as matpat

metrics = ["RMSE", "MAE", "MAPE", "R2", "MFB", "MFE"]
models = table_5.index.tolist()

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_map = dict(zip(models, default_colors))

fig, axes = plt.subplots(2, 3, figsize = (18, 8))
axes = axes.flatten()

for i, metric in enumerate(metrics):
    ax = axes[i]

    sb.barplot(x=models, y=table_5[metric].values, ax=ax, palette=[color_map[m] for m in models])
    ax.set_xticks([]) 
    ax.set_xlabel("")
    ax.set_ylabel(metric)
    ax.grid(axis = 'y', linestyle='--', alpha = 0.5)

plt.subplots_adjust(top = 0.83, bottom = 0.15, wspace=0.3)

legendm = [matpat.Patch(color=color_map[m], label=m) for m in models]
fig.legend(handles = legendm, loc = "lower center", ncol = len(models), bbox_to_anchor = (0.5, 0.02))

plt.suptitle("Comparison of Model Performance", y = 0.90)
plt.show()



# ================================================================================================================
# ============================================= EVALUATION PART II ===============================================
# ============================================= DIFFERENT STATIONS ===============================================



# --------------------------------- DATA IMPORT/PROCESSING ---------------------------------


files = ["2020.xlsx", "2021.xlsx", "2022.xlsx", "2023.xlsx", "2024.xlsx"]
cols = ["Date", "35T", "05T", "63T"]

# -------------------------------------- STATION 35T ---------------------------------------

T35 = []

for f in files:
    df = pd.read_excel(f, parse_dates = ["Date"])
    df = df.rename(columns = str.strip)
    df = df[cols]

    df["35T"] = pd.to_numeric(df["35T"], errors = "coerce")
    df = df.dropna(subset = ["Date", "35T"])

    T35.append(df)

merged_35 = pd.concat(T35, ignore_index=True)

merged_35["Date"] = pd.to_datetime(merged_35["Date"], errors = "coerce")
merged_35["Date"] = merged_35["Date"].dt.date
merged_35 = merged_35.sort_values("Date")

#print(merged_35)

ts_35 = merged_35.set_index("Date")["35T"]

split_index = int(len(ts_35) * 0.8)

train35 = ts_35[:split_index]
test35 = ts_35[split_index:]

# For verification
print("\nTrain size:", len(train35))
print("Test size:", len(test35))


plt.figure(figsize=(16, 4))
plt.plot(merged_35["Date"], merged_35["35T"], marker = 'o', linestyle = '-', markersize = 3, linewidth = 1)
plt.xlabel("Date")
plt.ylabel("PM2.5 Concentration")
plt.title("Station 35T")
plt.grid(True)
plt.gcf().autofmt_xdate()
plt.show()

# -------------------------------------- STATION 05T ---------------------------------------

T05 = []

for f in files:
    df = pd.read_excel(f, parse_dates = ["Date"])
    df = df.rename(columns = str.strip)
    df = df[cols]

    df["05T"] = pd.to_numeric(df["05T"], errors = "coerce")
    df = df.dropna(subset = ["Date", "05T"])

    T05.append(df)

merged_05 = pd.concat(T05, ignore_index = True)

merged_05["Date"] = pd.to_datetime(merged_05["Date"], errors = "coerce")
merged_05["Date"] = merged_05["Date"].dt.date
merged_05 = merged_05.sort_values("Date")

#print(merged_05)

ts_05 = merged_05.set_index("Date")["05T"]

split_index = int(len(ts_05) * 0.8)

train05 = ts_05[:split_index]
test05 = ts_05[split_index:]

# For verification
print("\nTrain size:", len(train05))
print("Test size:", len(test05))


plt.figure(figsize=(16, 4))
plt.plot(merged_05["Date"], merged_05["05T"], marker = 'o', linestyle = '-', markersize = 3, linewidth = 1)
plt.xlabel("Date")
plt.ylabel("PM2.5 Concentration")
plt.title("Station 05T")
plt.grid(True)
plt.gcf().autofmt_xdate()
plt.show()

# -------------------------------------- STATION 63T ---------------------------------------

T63 = []

for f in files:
    df = pd.read_excel(f, parse_dates = ["Date"])
    df = df.rename(columns = str.strip)
    df = df[cols]

    df["63T"] = pd.to_numeric(df["63T"], errors = "coerce")
    df = df.dropna(subset = ["Date", "63T"])

    T63.append(df)

merged_63 = pd.concat(T63, ignore_index = True)

merged_63["Date"] = pd.to_datetime(merged_63["Date"], errors = "coerce")
merged_63["Date"] = merged_63["Date"].dt.date
merged_63 = merged_63.sort_values("Date")

#print(merged_63)

ts_63 = merged_63.set_index("Date")["63T"]

split_index = int(len(ts_63) * 0.8)

train63 = ts_63[:split_index]
test63 = ts_63[split_index:]

# For verification
print("\nTrain size:", len(train63))
print("Test size:", len(test63))


plt.figure(figsize=(16, 4))
plt.plot(merged_63["Date"], merged_63["63T"], marker = 'o', linestyle = '-', markersize = 3, linewidth = 1)
plt.xlabel("Date")
plt.ylabel("PM2.5 Concentration")
plt.title("Station 63T")
plt.grid(True)
plt.gcf().autofmt_xdate()
plt.show()

# ======================================== PART I - ARIMA ======================================

# ---------------------- R ARIMA MODEL ----------------------

r_train35 = ro.FloatVector(train35.values)
r_train05 = ro.FloatVector(train05.values)
r_train63 = ro.FloatVector(train63.values)
ts35 = ro.r.ts(r_train35, frequency = 365)
ts05 = ro.r.ts(r_train05, frequency = 365)
ts63 = ro.r.ts(r_train63, frequency = 365)

# ARIMA(1,1,2)
manual_model35 = forecast_pkg.Arima(ts35, order=ro.IntVector([1,1,2]), include_drift=True)
manual_model05 = forecast_pkg.Arima(ts05, order=ro.IntVector([1,1,2]), include_drift=True)
manual_model63 = forecast_pkg.Arima(ts63, order=ro.IntVector([1,1,2]), include_drift=True)

# ----------------------- R FORECAST ------------------------

stlf = ro.r['stlf']

fc_stl35 = stlf(ts35, h = len(test35))
fc_stl05 = stlf(ts05, h = len(test05))
fc_stl63 = stlf(ts63, h = len(test63))

# Convert forecast back to Python
with localconverter(ro.default_converter + pandas2ri.converter):
    forecast_r35 = ro.conversion.rpy2py(fc_stl35.rx2('mean'))
with localconverter(ro.default_converter + pandas2ri.converter):
    forecast_r05 = ro.conversion.rpy2py(fc_stl05.rx2('mean'))
with localconverter(ro.default_converter + pandas2ri.converter):
    forecast_r63 = ro.conversion.rpy2py(fc_stl63.rx2('mean'))

y_true35 = test35.values
y_pred35 = np.array(forecast_r35)
y_true05 = test05.values
y_pred05 = np.array(forecast_r05)
y_true63 = test63.values
y_pred63 = np.array(forecast_r63)

# ======================================== PART II - ANN_RESID =================================

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

arima_fits35 = ro.r('fitted')(manual_model35)
arima_fits05 = ro.r('fitted')(manual_model05)
arima_fits63 = ro.r('fitted')(manual_model63)

with localconverter(ro.default_converter + pandas2ri.converter):
    arima_fitted35 = np.array(ro.conversion.rpy2py(arima_fits35))
with localconverter(ro.default_converter + pandas2ri.converter):
    arima_fitted05 = np.array(ro.conversion.rpy2py(arima_fits05))
with localconverter(ro.default_converter + pandas2ri.converter):
    arima_fitted63 = np.array(ro.conversion.rpy2py(arima_fits63))

arima_fitted35 = pd.Series(arima_fitted35, index=train35.index[-len(arima_fitted35):])
arima_fitted05 = pd.Series(arima_fitted05, index=train05.index[-len(arima_fitted05):])
arima_fitted63 = pd.Series(arima_fitted63, index=train63.index[-len(arima_fitted63):])

residuals_h35 = train35.loc[arima_fitted35.index] - arima_fitted35
residuals_h35 = residuals_h35.dropna()
residuals_h05 = train05.loc[arima_fitted05.index] - arima_fitted05
residuals_h05 = residuals_h05.dropna()
residuals_h63 = train63.loc[arima_fitted63.index] - arima_fitted63
residuals_h63 = residuals_h63.dropna()

residuals_r35 = ro.FloatVector(residuals_h35.values)
residuals_ts35 = ro.r.ts(residuals_r35)
residuals_r05 = ro.FloatVector(residuals_h05.values)
residuals_ts05 = ro.r.ts(residuals_r05)
residuals_r63 = ro.FloatVector(residuals_h63.values)
residuals_ts63 = ro.r.ts(residuals_r63)


# ===================================== PART III - HYBRID+QREG =================================

from rpy2.robjects import globalenv

model_ann35 = nnetar1(residuals_ts35)
model_ann05 = nnetar1(residuals_ts05)
model_ann63 = nnetar1(residuals_ts63)

ann_fits35 = ro.r('fitted')(model_ann35)
ann_fits05 = ro.r('fitted')(model_ann05)
ann_fits63 = ro.r('fitted')(model_ann63)

with localconverter(ro.default_converter + pandas2ri.converter):
    ann_fit35 = np.array(ro.conversion.rpy2py(ann_fits35))
with localconverter(ro.default_converter + pandas2ri.converter):
    ann_fit05 = np.array(ro.conversion.rpy2py(ann_fits05))
with localconverter(ro.default_converter + pandas2ri.converter):
    ann_fit63 = np.array(ro.conversion.rpy2py(ann_fits63))

fc_resid_ann35 = forecast_ann(model_ann35, h = len(test35))

with localconverter(ro.default_converter + pandas2ri.converter):
    forecast_residuals_ann35 = ro.conversion.rpy2py(fc_resid_ann35.rx2('mean'))

residuals_fc35 = np.array(forecast_residuals_ann35)

fc_resid_ann05 = forecast_ann(model_ann05, h = len(test05))

with localconverter(ro.default_converter + pandas2ri.converter):
    forecast_residuals_ann05 = ro.conversion.rpy2py(fc_resid_ann05.rx2('mean'))

residuals_fc05 = np.array(forecast_residuals_ann05)

fc_resid_ann63 = forecast_ann(model_ann63, h = len(test63))

with localconverter(ro.default_converter + pandas2ri.converter):
    forecast_residuals_ann63 = ro.conversion.rpy2py(fc_resid_ann63.rx2('mean'))

residuals_fc63 = np.array(forecast_residuals_ann63)

y_train35 = np.array(train35.values)
y_train05 = np.array(train05.values)
y_train63 = np.array(train63.values)

arima_series35 = pd.Series(arima_fitted35)
ann_series35 = pd.Series(ann_fit35)
arima_series05 = pd.Series(arima_fitted05)
ann_series05 = pd.Series(ann_fit05)
arima_series63 = pd.Series(arima_fitted63)
ann_series63 = pd.Series(ann_fit63)

min_len35 = min(len(train35), len(arima_series35), len(ann_series35))
min_len05 = min(len(train05), len(arima_series05), len(ann_series05))
min_len63 = min(len(train63), len(arima_series63), len(ann_series63))

y_aligned35 = train35.iloc[-min_len35:]
arima_aligned35 = arima_series35.iloc[-min_len35:]
ann_aligned35 = ann_series35.iloc[-min_len35:]
y_aligned05 = train05.iloc[-min_len05:]
arima_aligned05 = arima_series05.iloc[-min_len05:]
ann_aligned05 = ann_series05.iloc[-min_len05:]
y_aligned63 = train63.iloc[-min_len63:]
arima_aligned63 = arima_series63.iloc[-min_len63:]
ann_aligned63 = ann_series63.iloc[-min_len63:]

arima_aligned35.index = y_aligned35.index
ann_aligned35.index = y_aligned35.index
arima_aligned05.index = y_aligned05.index
ann_aligned05.index = y_aligned05.index
arima_aligned63.index = y_aligned63.index
ann_aligned63.index = y_aligned63.index

df_reg35 = pd.DataFrame({"y": y_aligned35, "arima": arima_aligned35, "ann": ann_aligned35}).dropna()

with localconverter(ro.default_converter + pandas2ri.converter):
    globalenv["df_reg35"] = df_reg35

df_reg05 = pd.DataFrame({"y": y_aligned05, "arima": arima_aligned05, "ann": ann_aligned05}).dropna()

with localconverter(ro.default_converter + pandas2ri.converter):
    globalenv["df_reg05"] = df_reg05

df_reg63 = pd.DataFrame({"y": y_aligned63, "arima": arima_aligned63, "ann": ann_aligned63}).dropna()

with localconverter(ro.default_converter + pandas2ri.converter):
    globalenv["df_reg63"] = df_reg63


rq = importr("quantreg")

ro.r("""
model_qreg35 <- rq(y ~ arima + ann - 1, tau = 0.5, data = df_reg35)
summary(model_qreg35)
""")

qr_summary35 = ro.r("summary(model_qreg35, se = 'boot')")

ro.r("""
model_qreg05 <- rq(y ~ arima + ann - 1, tau = 0.5, data = df_reg05)
summary(model_qreg05)
""")

qr_summary05 = ro.r("summary(model_qreg05, se = 'boot')")

ro.r("""
model_qreg63 <- rq(y ~ arima + ann - 1, tau = 0.5, data = df_reg63)
summary(model_qreg63)
""")

qr_summary63 = ro.r("summary(model_qreg63, se = 'boot')")

print("\n===== ARIMA–ANN-QREG Results =====")
print("\nQuantile Regression Summary:\n", qr_summary35)
print("\nQuantile Regression Summary:\n", qr_summary05)
print("\nQuantile Regression Summary:\n", qr_summary63)
print("==================================\n")

qreg_coef35 = ro.r("coef(model_qreg35)")
qreg_coef05 = ro.r("coef(model_qreg05)")
qreg_coef63 = ro.r("coef(model_qreg63)")

with localconverter(ro.default_converter + pandas2ri.converter):
    qreg_coef35 = np.array(qreg_coef35)
with localconverter(ro.default_converter + pandas2ri.converter):
    qreg_coef05 = np.array(qreg_coef05)
with localconverter(ro.default_converter + pandas2ri.converter):
    qreg_coef63 = np.array(qreg_coef63)

beta_arima_qreg35 = qreg_coef35[0]
beta_annresid_qreg35 = qreg_coef35[1]

y_pred_qreg35 = beta_arima_qreg35 * y_pred35 + beta_annresid_qreg35 * residuals_fc35

beta_arima_qreg05 = qreg_coef05[0]
beta_annresid_qreg05 = qreg_coef05[1]

y_pred_qreg05 = beta_arima_qreg05 * y_pred05 + beta_annresid_qreg05 * residuals_fc05

beta_arima_qreg63 = qreg_coef63[0]
beta_annresid_qreg63 = qreg_coef63[1]

y_pred_qreg63 = beta_arima_qreg63 * y_pred63 + beta_annresid_qreg63 * residuals_fc63


# ----------------- Metrics ----------------------

rmse_qreg35 = np.sqrt(mean_squared_error(y_true35, y_pred_qreg35))
mae_qreg35 = mean_absolute_error(y_true35, y_pred_qreg35)
mape_qreg35 = mean_absolute_percentage_error(y_true35, y_pred_qreg35)
r2_qreg35 = r2_score(y_true35, y_pred_qreg35)
mfb_qreg35 = mean_fractional_bias(y_true35, y_pred_qreg35)
mfe_qreg35 = mean_fractional_error(y_true35, y_pred_qreg35)

rmse_qreg05 = np.sqrt(mean_squared_error(y_true05, y_pred_qreg05))
mae_qreg05 = mean_absolute_error(y_true05, y_pred_qreg05)
mape_qreg05 = mean_absolute_percentage_error(y_true05, y_pred_qreg05)
r2_qreg05 = r2_score(y_true05, y_pred_qreg05)
mfb_qreg05 = mean_fractional_bias(y_true05, y_pred_qreg05)
mfe_qreg05 = mean_fractional_error(y_true05, y_pred_qreg05)

rmse_qreg63 = np.sqrt(mean_squared_error(y_true63, y_pred_qreg63))
mae_qreg63 = mean_absolute_error(y_true63, y_pred_qreg63)
mape_qreg63 = mean_absolute_percentage_error(y_true63, y_pred_qreg63)
r2_qreg63 = r2_score(y_true63, y_pred_qreg63)
mfb_qreg63 = mean_fractional_bias(y_true63, y_pred_qreg63)
mfe_qreg63 = mean_fractional_error(y_true63, y_pred_qreg63)


# ================================= PART VI - COMPARISON OF ALL MODELS =================================

import seaborn as sb

# ------------------- TABLE 7 -------------------

table_7 = pd.DataFrame({"Station": ["35T", "05T", "63T"],
    "n": [len(ts_35), len(ts_05), len(ts_63)],
    "RMSE": [rmse_qreg35, rmse_qreg05, rmse_qreg63],
    "MAE":  [mae_qreg35, mae_qreg05, mae_qreg63],
    "MAPE": [mape_qreg35, mape_qreg05, mape_qreg63],
    "R2":   [r2_qreg35, r2_qreg05, r2_qreg63],
    "MFB":  [mfb_qreg35, mfb_qreg05, mfb_qreg63],
    "MFE":  [mfe_qreg35, mfe_qreg05, mfe_qreg63]
})

table_7 = table_7.set_index("Station")

print("\n========== SUMMARY OF MODEL ACCURACY ==========\n")
print(table_7)
print("\n========================================================\n")

