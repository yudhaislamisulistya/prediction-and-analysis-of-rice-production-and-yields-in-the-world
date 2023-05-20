# 1. Import Library

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf, pacf

st.set_option('deprecation.showPyplotGlobalUse', False)
st.sidebar.title("[Table Of Content](#prediction-and-analysis-of-rice-production-and-yields-in-the-world-using-ensemble-learning-techniques)", anchor='top')
st.sidebar.markdown("1. [Dataset Information](#1-dataset-information)")
st.sidebar.markdown("2. [Data Preprocessing Awal](#2-data-preprocessing-awal)")
st.sidebar.markdown("3. [Exploratory Data Analysis](#3-exploratory-data-analysis)")
st.sidebar.markdown("4. [Data Preprocessing Lanjutan](#4-data-preprocessing-lanjutan)")
st.sidebar.markdown("5. [Modelling](#5-modelling)")
st.sidebar.markdown("6. [Evaluasi Model](#6-evaluasi-model)")
st.sidebar.markdown("7. [Kesimpulan](#7-kesimpulan)")
st.sidebar.markdown("8. [Software Development Goals](#8-software-development-goals)")
st.sidebar.markdown("9. [Referensi/Formula](#9-reference-formula)")

# FUNGSI

st.title("Prediction and Analysis of Rice Production and Yields in the World Using Ensemble Learning Techniques")

# 2. Membaca dan Informasi Dataset

st.header("1. Dataset Information")
dataset_original = pd.read_csv('dataset.csv')
st.write(dataset_original)

feature_list = dataset_original.columns.tolist()
feature_list = ', '.join([str(elem) for elem in feature_list])
st.write("1. Dataset Berisi {} Baris dan {} Kolom".format(dataset_original.shape[0], dataset_original.shape[1]))
st.write("2. Kolom yang ada pada dataset ini adalah {}".format(feature_list))
st.write("3. Tipe data dari masing-masing kolom adalah {}".format(dataset_original.dtypes))
st.write("4. Terdapat {} data yang hilang pada dataset ini".format(dataset_original.isnull().sum().sum()))
st.write("5. Terdapat {} data duplikat pada dataset ini".format(dataset_original.duplicated().sum()))


# 3. Data Preprocessing
st.markdown("# 2. Data Preprocessing Awal")
# remove feature
st.markdown("## 1. Menghapus Kolom yang Tidak Digunakan")
dataset_after_remove = dataset_original.drop(['Domain Code', 'Domain','Area Code (M49)','Element Code', 'Item Code (CPC)', 'Item', 'Year Code', 'Flag', 'Flag Description'], axis=1)
st.dataframe(dataset_after_remove, width=2000)
st.markdown("## 2. Split Dataset Berdasarkan Element")
dataset_area_harvested = dataset_after_remove[dataset_after_remove['Element'] == 'Area harvested']
dataset_yield = dataset_after_remove[dataset_after_remove['Element'] == 'Yield']
dataset_production = dataset_after_remove[dataset_after_remove['Element'] == 'Production']
st.markdown("### 2.1. Dataset Area Harvested")
st.dataframe(dataset_area_harvested, width=2000)
st.write("Dataset Area Harvested memiliki {} baris dan {} kolom".format(dataset_area_harvested.shape[0], dataset_area_harvested.shape[1]))
st.write("Dataset Area Harvested memiliki {} data duplikat".format(dataset_area_harvested.duplicated().sum()))
st.write("Dataset Area Harvested memiliki {} data yang hilang".format(dataset_area_harvested.isnull().sum().sum()))
st.write("Area harvested  Ini merujuk pada total luas tanah yang benar-benar dipanen untuk suatu tanaman tertentu, dalam hal ini, padi. Area ini tidak termasuk area yang ditanam tetapi kemudian gagal dipanen karena alasan seperti kekeringan, penyakit, atau banjir. Oleh karena itu, 'area harvested' memberikan ukuran lebih akurat tentang sejauh mana tanaman tersebut berhasil tumbuh dan dipanen.")
st.markdown("### 2.2. Dataset Yield")
st.dataframe(dataset_yield, width=2000)
st.write("Dataset Yield memiliki {} baris dan {} kolom".format(dataset_yield.shape[0], dataset_yield.shape[1]))
st.write("Dataset Yield memiliki {} data duplikat".format(dataset_yield.duplicated().sum()))
st.write("Dataset Yield memiliki {} data yang hilang".format(dataset_yield.isnull().sum().sum()))
st.write("Yield Ini adalah rasio dari total produksi padi terhadap luas tanah yang dipanen. Ini adalah ukuran efisiensi dalam produksi padi dan biasanya diukur dalam ton per hektare atau kilogram per hektare. Tingkat hasil yang tinggi biasanya mengindikasikan teknologi dan praktik pertanian yang efisien dan efektif.")
st.markdown("### 2.3. Dataset Production")
st.dataframe(dataset_production, width=2000)
st.write("Dataset Production memiliki {} baris dan {} kolom".format(dataset_production.shape[0], dataset_production.shape[1]))
st.write("Dataset Production memiliki {} data duplikat".format(dataset_production.duplicated().sum()))
st.write("Dataset Production memiliki {} data yang hilang".format(dataset_production.isnull().sum().sum()))
st.write("Production Ini merujuk pada total volume atau berat dari padi yang dihasilkan dan dipanen. Produksi bisa dipengaruhi oleh banyak faktor, seperti cuaca, jenis tanah, penggunaan pupuk dan pestisida, dan teknologi pertanian. Biasanya diukur dalam ton atau kilogram..")

# 3. EDA (Exploratory Data Analysis)
st.markdown("# 3. Exploratory Data Analysis")
st.markdown("## 1. Statistik Deskriptif")
describe_value_dataset_area_harvested = dataset_area_harvested.rename(columns={'Value': 'Value Area Harvested (ha)'})['Value Area Harvested (ha)'].describe()
describe_value_dataset_yield = dataset_yield.rename(columns={'Value': 'Value Yield (hg/ha)'})['Value Yield (hg/ha)'].describe()
describe_value_dataset_production = dataset_production.rename(columns={'Value': 'Value Production (ton)'})['Value Production (ton)'].describe()
marge_describe_value = pd.concat([describe_value_dataset_area_harvested, describe_value_dataset_yield, describe_value_dataset_production], axis=1)
st.dataframe(marge_describe_value, width=2000)


st.markdown("## 2. Pengecakan Nilai Unik pada Setiap Kolom")
for i in dataset_after_remove.columns:
    list_unique = dataset_after_remove[i].unique()
    st.write("Kolom {} memiliki {} nilai unik, yaitu {}".format(i, len(list_unique), list_unique))
    
st.markdown("## 3. Visualisasi Data")
st.markdown("### 3.1. Area Harvested")
st.markdown("#### 3.1.1. Visualisasi Data Area Harvested Berdasarkan Tahu")
fig = px.line(dataset_area_harvested, x="Year", y="Value", color='Area', title='Area Harvested (Line Chart)')
st.plotly_chart(fig)
st.markdown("#### 3.1.2. Visualisasi Data Map Area Harvested")
total_value_area_harvested = dataset_area_harvested.groupby('Area')['Value'].sum().reset_index()
fig = px.choropleth(total_value_area_harvested, locations="Area", locationmode='country names', color="Value", hover_name="Area", range_color=[0, 10000000], color_continuous_scale="plasma", title='Area Harvested Map')
st.plotly_chart(fig)
st.markdown("### 3.2. Yield")
st.markdown("#### 3.2.1. Visualisasi Data Yield Berdasarkan Tahun")
fig = px.line(dataset_yield, x="Year", y="Value", color='Area', title='Yield (Line Chart)')
st.plotly_chart(fig)
st.markdown("#### 3.2.2. Visualisasi Data Map Yield")
total_value_yield = dataset_yield.groupby('Area')['Value'].sum().reset_index()
fig = px.choropleth(total_value_yield, locations="Area", locationmode='country names', color="Value", hover_name="Area", range_color=[0, 1000000], color_continuous_scale="plasma", title='Yield Map')
st.plotly_chart(fig)
st.markdown("### 3.3. Production")
st.markdown("#### 3.3.1. Visualisasi Data Production Berdasarkan Tahun")
fig = px.line(dataset_production, x="Year", y="Value", color='Area', title='Production (Line Chart)')
st.plotly_chart(fig)
st.markdown("#### 3.3.2. Visualisasi Data Map Production")
total_value_production = dataset_production.groupby('Area')['Value'].sum().reset_index()
fig = px.choropleth(total_value_production, locations="Area", locationmode='country names', color="Value", hover_name="Area", range_color=[0, 10000000], color_continuous_scale="plasma", title='Production Map')
st.plotly_chart(fig)
st.markdown("### 3.4. Data Area Harvested, Yield, dan Production (Gabungan)")
combined_dataset = pd.merge(dataset_area_harvested, dataset_yield, on=['Area', 'Year'])
combined_dataset = pd.merge(combined_dataset, dataset_production, on=['Area', 'Year'])
combined_dataset = combined_dataset.rename(columns={'Value_x': 'Value Area Harvested (ha)', 'Value_y': 'Value Yield (hg/ha)', 'Value': 'Value Production (ton)'})
# input area default value indonesia with st.selectbox

input_area = st.selectbox('Pilih Area', combined_dataset['Area'].unique(), index=55)
fig_area_indonesia_only = px.line(combined_dataset[combined_dataset['Area'] == input_area], x="Year", y=["Value Area Harvested (ha)", "Value Yield (hg/ha)", "Value Production (ton)"], title='Area Harvested, Yield, dan Production di {}'.format(input_area), labels={"variable": "Tipe", "value": "Value"}, color_discrete_map={'Value Area Harvested (ha)': 'blue', 'Value Yield (hg/ha)': 'red', 'Value Production (ton)': 'green'})
st.plotly_chart(fig_area_indonesia_only)
st.write("Misalkan Negeara yang dipilih adalah {}, Sebagai contoh pada tahun 2015, Area Harvested di {} adalah {} ha, Yield di {} adalah {} hg/ha, dan Production di {} adalah {} tonnes".format(input_area, input_area, combined_dataset[(combined_dataset['Area'] == input_area) & (combined_dataset['Year'] == 2015)]['Value Area Harvested (ha)'].values[0], input_area, combined_dataset[(combined_dataset['Area'] == input_area) & (combined_dataset['Year'] == 2015)]['Value Yield (hg/ha)'].values[0], input_area, combined_dataset[(combined_dataset['Area'] == input_area) & (combined_dataset['Year'] == 2015)]['Value Production (ton)'].values[0]))

st.markdown("## 4. Korelasi")
st.markdown("### 4.1. Korelasi Area Harvested, Yield, dan Production")
fig = px.scatter(combined_dataset, x="Value Area Harvested (ha)", y="Value Yield (hg/ha)", color="Value Production (ton)", title='Korelasi Area Harvested, Yield, dan Production')
st.plotly_chart(fig)
encoder = LabelEncoder()
combined_dataset['Area'] = encoder.fit_transform(combined_dataset['Area'])

st.write(combined_dataset)
correlation_area_harvested_yield = combined_dataset['Value Area Harvested (ha)'].corr(combined_dataset['Value Yield (hg/ha)'])
correlation_area_harvested_production = combined_dataset['Value Area Harvested (ha)'].corr(combined_dataset['Value Production (ton)'])
corrlation_yield_production = combined_dataset['Value Yield (hg/ha)'].corr(combined_dataset['Value Production (ton)'])
corrlation_area_harvested_yield_production = combined_dataset[['Area','Year','Value Area Harvested (ha)', 'Value Yield (hg/ha)', 'Value Production (ton)']].corr()
st.write(corrlation_area_harvested_yield_production)

st.markdown("### 4.2. Korelasi Area Harvested dan Yield")
fig = px.scatter(combined_dataset, x="Value Area Harvested (ha)", y="Value Yield (hg/ha)", title='Korelasi Area Harvested dan Yield')
st.plotly_chart(fig)
st.write("Korelasi Area Harvested dan Yield adalah {}".format(correlation_area_harvested_yield))

st.markdown("### 4.3. Korelasi Area Harvested dan Production")
fig = px.scatter(combined_dataset, x="Value Area Harvested (ha)", y="Value Production (ton)", title='Korelasi Area Harvested dan Production')
st.plotly_chart(fig)
st.write("Korelasi Area Harvested dan Production adalah {}".format(correlation_area_harvested_production))

st.markdown("### 4.4. Korelasi Yield dan Production")
fig = px.scatter(combined_dataset, x="Value Yield (hg/ha)", y="Value Production (ton)", title='Korelasi Yield dan Production')
st.plotly_chart(fig)
st.write("Korelasi Yield dan Production adalah {}".format(corrlation_yield_production))

st.markdown("## 5. Kesimpulan EDA")
st.write('Berdasarkan analisis eksplorasi data (EDA) yang telah dilakukan, berikut adalah kesimpulan yang dapat diambil:')
st.write('1. Korelasi antar Variabel: Korelasi antara variabel Year dan variabel lainnya (Area Harvested, Yield, dan Production) relatif lemah, dengan koefisien korelasi kurang dari 0.4. Sebaliknya, korelasi antara Area Harvested dan Production cukup kuat (0.9064), menunjukkan bahwa peningkatan area panen biasanya diikuti oleh peningkatan produksi.')
st.write('2. Variabilitas Data: Dataset mencakup 149 negara unik dan mencakup 3 elemen yang berbeda, yaitu Area Harvested, Yield, dan Production. Data ini dicatat dalam 3 unit yang berbeda (ha, hg/ha, tonnes) selama periode 61 tahun.')
st.write('3. Kekhasan Data: Area atau negara yang diamati sangat beragam, mulai dari Afghanistan hingga Zimbabwe. Dalam hal ini, nilai Area Harvested, Yield, dan Production tentu saja akan sangat bervariasi antar negara karena berbagai faktor seperti teknologi, iklim, jenis tanah, dll.')
st.write('4. Jangka Waktu: Data mencakup 61 tahun, mulai dari 1961 hingga 2021. Ini menunjukkan bahwa kita memiliki data historis yang cukup lama untuk melakukan analisis tren dan peramalan masa depan. Namun, perlu diingat bahwa hubungan lemah antara tahun dan variabel lainnya berarti bahwa peramalan berdasarkan tahun mungkin akan kurang akurat.')
st.write('5. Total data yang telah mencakup elemen Area Harvested, Yield dan Production yakni 6970.')
st.write("Jadi dalam kesimpulan ini, kami ingin melakukan beberapa pendekatan untuk memprediksi nilai Area Harvested, Yield, dan Production di masa depan.")
st.write("1. Kami akan menggunakan model regresi untuk memprediksi nilai-nilai (Area Harvested, Yield dan Production) ini berdasarkan tahun atau negara. serta menggunakan Area Harvested untuk memprediksi Production")
st.write("2. Kami juga akan menggunakan model time series untuk memprediksi nilai-nilai ini berdasarkan tahun saja.")
st.write("3. Kami akan membandingkan kedua model ini dan melihat mana yang lebih akurat.")
st.write("4. Kemudian kami akan mencoba menggunakan ensemble learning untuk memprediksi nilai-nilai berdasarkan tahun dengan model time series")

st.markdown("# 4. Data Preprocessing Lanjutan")
dataset = combined_dataset.copy()
dataset = dataset.drop(['Element_x', 'Unit_x', 'Element_y', 'Unit_y', 'Element', 'Unit', ], axis=1)
# Reverse Label Encoder
dataset['Area'] = encoder.inverse_transform(dataset['Area'])

st.markdown("## 4.1. Label Encoding")
dataset['Area'] = encoder.fit_transform(dataset['Area'])
st.dataframe(dataset, width=2000)

st.markdown("## 4.2. Normalisasi Data")
scaler_value_area = MinMaxScaler()
scaler_value_yield = MinMaxScaler()
scaler_value_production = MinMaxScaler()
scaler_year = MinMaxScaler()

dataset['Value Area Harvested (ha)'] = scaler_value_area.fit_transform(dataset[['Value Area Harvested (ha)']])
dataset['Value Yield (hg/ha)'] = scaler_value_yield.fit_transform(dataset[['Value Yield (hg/ha)']])
dataset['Value Production (ton)'] = scaler_value_production.fit_transform(dataset[['Value Production (ton)']])
dataset['Year'] = scaler_year.fit_transform(dataset[['Year']])
st.dataframe(dataset, width=2000)

st.markdown("## 4.3. Sum Berdasarkan Tahun (1961-2021)")
st.markdown("### 4.3.1. Original Dataset")
dataset_year = combined_dataset.copy()
dataset_year = dataset_year.groupby(['Year']).sum()
dataset_year = dataset_year.reset_index()
dataset_year = dataset_year.drop(['Area'], axis=1)
st.dataframe(dataset_year, width=2000)

st.markdown("### 4.3.2. Setelah Normalisasi")
scaler_value_area_year = MinMaxScaler()
scaler_value_yield_year = MinMaxScaler()
scaler_value_production_year = MinMaxScaler()
scaler_year_year = MinMaxScaler()

dataset_year['Value Area Harvested (ha)'] = scaler_value_area_year.fit_transform(dataset_year[['Value Area Harvested (ha)']])
dataset_year['Value Yield (hg/ha)'] = scaler_value_yield_year.fit_transform(dataset_year[['Value Yield (hg/ha)']])
dataset_year['Value Production (ton)'] = scaler_value_production_year.fit_transform(dataset_year[['Value Production (ton)']])
dataset_year['Year'] = scaler_year_year.fit_transform(dataset_year[['Year']])
st.dataframe(dataset_year, width=2000)

st.markdown("# 5. Modelling")

st.markdown("## 5.1. Linear Regression")
st.markdown("### 5.1.1. Model 1: Menggunakan Area Harvested untuk memprediksi Production")
X1 = dataset[['Value Area Harvested (ha)']]
y1 = dataset[['Value Production (ton)']]

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=0)


st.write("Data Training: {} baris".format(X1_train.shape[0]))
st.write("Data Testing: {} baris".format(X1_test.shape[0]))

model1 = LinearRegression()
model1.fit(X1_train, y1_train)
y1_pred = model1.predict(X1_test)

mse_area_harvested_production = mean_squared_error(y1_test, y1_pred)
rmse_area_harvested_production = np.sqrt(mse_area_harvested_production)
r2_area_harvested_production = r2_score(y1_test, y1_pred)

dataframe_mse_rmse_r2_area_harvested_production = pd.DataFrame({'MSE': [mse_area_harvested_production], 'RMSE': [rmse_area_harvested_production], 'R2': [r2_area_harvested_production]}, index=['Model 1 (LR): Area Harvested Production'])
st.dataframe(dataframe_mse_rmse_r2_area_harvested_production, width=2000)

X1_test = scaler_value_area.inverse_transform(X1_test)
X1_test = pd.DataFrame(X1_test, columns=['Value Area Harvested (ha)'])
X1_train = scaler_value_area.inverse_transform(X1_train)
X1_train = pd.DataFrame(X1_train, columns=['Value Area Harvested (ha)'])
y1_train = scaler_value_production.inverse_transform(y1_train)
y1_train = pd.DataFrame(y1_train, columns=['Value Production (ton)'])
y1_test = scaler_value_production.inverse_transform(y1_test)
y1_test = pd.DataFrame(y1_test, columns=['Value Production (ton)'])
y1_pred = scaler_value_production.inverse_transform(y1_pred)

plt.scatter(X1_test, y1_test, color = 'blue', label = 'Actual')

plt.scatter(X1_test, y1_pred, color = 'red', label = 'Predicted')

plt.plot(X1_test, y1_pred, color = 'black', label = 'Regression Line')

scatter_actual = go.Scatter(x=X1_test['Value Area Harvested (ha)'], y=y1_test['Value Production (ton)'], mode='markers', name='Actual', marker=dict(color='LightSkyBlue'))
scatter_predicted = go.Scatter(x=X1_test['Value Area Harvested (ha)'], y=y1_pred.flatten(), mode='markers', name='Predicted', marker=dict(color='red'))
scatter_regression = go.Scatter(x=X1_test['Value Area Harvested (ha)'], y=y1_pred.flatten(), mode='lines', name='Regression Line', line=dict(color='white', width=2))

data = [scatter_actual, scatter_predicted, scatter_regression]

layout = go.Layout(title='Model 1 (Area Harvested)', xaxis=dict(title='Value Area Harvested (ha)'), yaxis=dict(title='Value Production (ton)'))

fig = go.Figure(data=data, layout=layout)

st.plotly_chart(fig)

prediksi = st.number_input('Masukkan nilai Area Harvested (ha)', min_value=0.0, value=0.0, step=0.01)
prediksi = np.array(prediksi).reshape(-1, 1)
prediksi = scaler_value_area.transform(prediksi)
prediksi = model1.predict(prediksi)
prediksi = scaler_value_production.inverse_transform(prediksi)
prediksi = prediksi.flatten()
if st.button('Prediksi'):
    prediksi = prediksi[0]
    prediksi = round(prediksi, 2)
    prediksi = int(prediksi)
    prediksi = "{:,}".format(prediksi)
    st.write("Prediksi Production (tonnes) : {}".format(prediksi))

st.markdown("### 5.1.2. Model 2: Menggunakan Year untuk memprediksi Production")

X2 = dataset_year[['Year']]
y2 = dataset_year[['Value Production (ton)']]

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=0)


st.write("Data Training: {} baris".format(X2_train.shape[0]))
st.write("Data Testing: {} baris".format(X2_test.shape[0]))

model2 = LinearRegression()
model2.fit(X2_train, y2_train)
y2_pred = model2.predict(X2_test)

mse_year_production_lr = mean_squared_error(y2_test, y2_pred)
rmse_year_production_lr = np.sqrt(mse_year_production_lr)
r2_year_production_lr = r2_score(y2_test, y2_pred)

dataframe_mse_rmse_r2_year_production_lr = pd.DataFrame({'MSE': [mse_year_production_lr], 'RMSE': [rmse_year_production_lr], 'R2': [r2_year_production_lr]}, index=['Model 2 (LR): Year Production'])
st.dataframe(dataframe_mse_rmse_r2_year_production_lr, width=2000)

y2_test_year_production_lr_scaled =  y2_test
y2_pred_year_production_lr_scaled = y2_pred

X2_test = scaler_year_year.inverse_transform(X2_test)
X2_test = pd.DataFrame(X2_test, columns=['Year'])
X2_train = scaler_year_year.inverse_transform(X2_train)
X2_train = pd.DataFrame(X2_train, columns=['Year'])
y2_train = scaler_value_production_year.inverse_transform(y2_train)
y2_train = pd.DataFrame(y2_train, columns=['Value Production (ton)'])
y2_test = scaler_value_production_year.inverse_transform(y2_test)
y2_test = pd.DataFrame(y2_test, columns=['Value Production (ton)'])
y2_pred = scaler_value_production_year.inverse_transform(y2_pred)

plt.scatter(X2_test, y2_test, color = 'blue', label = 'Actual')

plt.scatter(X2_test, y2_pred, color = 'red', label = 'Predicted')

plt.plot(X2_test, y2_pred, color = 'black', label = 'Regression Line')

scatter_actual = go.Scatter(x=X2_test['Year'], y=y2_test['Value Production (ton)'], mode='markers', name='Actual', marker=dict(color='LightSkyBlue'))
scatter_predicted = go.Scatter(x=X2_test['Year'], y=y2_pred.flatten(), mode='markers', name='Predicted', marker=dict(color='red'))
scatter_regression = go.Scatter(x=X2_test['Year'], y=y2_pred.flatten(), mode='lines', name='Regression Line', line=dict(color='white', width=2))

data = [scatter_actual, scatter_predicted, scatter_regression]

layout = go.Layout(title='Model 2 (Year)', xaxis=dict(title='Year'), yaxis=dict(title='Value Production (ton)'))

fig = go.Figure(data=data, layout=layout)

st.plotly_chart(fig)

prediksi2 = st.number_input('Masukkan nilai Year', min_value=0.0, value=0.0, step=1.0)
prediksi2 = np.array(prediksi2).reshape(-1, 1)
prediksi2 = scaler_year_year.transform(prediksi2)
prediksi2 = model2.predict(prediksi2)
prediksi2 = scaler_value_production_year.inverse_transform(prediksi2)
prediksi2 = prediksi2.flatten()
if st.button('Prediksi Model 2'):
    # ubah exponen menjadi float
    prediksi2 = float(prediksi2)
    # format dengan titik sebagai pemisah ribuan
    prediksi2 = "{:,.0f}".format(prediksi2)
    st.write("Prediksi Model 2 Production (tonnes) : {}".format(prediksi2))


st.markdown("### 5.1.3. Model 3: Menggunakan Year Dengan Filter Negara untuk memprediksi Production")

unique_area = dataset['Area'].unique()
unique_area_inverse = encoder.inverse_transform(unique_area.reshape(-1, 1))
unique_area_inverse = unique_area_inverse.flatten()
unique_area_inverse = np.sort(unique_area_inverse)
unique_area_inverse = unique_area_inverse.tolist()

input_area_model3 = st.selectbox('Pilih Area', unique_area_inverse, index=55)

index_input_area_model3 = unique_area_inverse.index(input_area_model3)

dataset_area = dataset[dataset['Area'] == index_input_area_model3]


X3 = dataset_area[['Year']]
y3 = dataset_area[['Value Production (ton)']]

X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=0)

st.write("Data Training: {} baris".format(X3_train.shape[0]))
st.write("Data Testing: {} baris".format(X3_test.shape[0]))

model3 = LinearRegression()
model3.fit(X3_train, y3_train)
y3_pred = model3.predict(X3_test)

mse_year_filter_production = mean_squared_error(y3_test, y3_pred)
rmse_year_filter_production = np.sqrt(mse_year_filter_production)
r2_year_filter_production = r2_score(y3_test, y3_pred)

dataframe_mse_rmse_r2_year_filter_production = pd.DataFrame({'MSE': [mse_year_filter_production], 'RMSE': [rmse_year_filter_production], 'R2': [r2_year_filter_production]}, index=['Model 3 (LR): Year Filter Production'])
st.dataframe(dataframe_mse_rmse_r2_year_filter_production, width=2000)

X3_test = scaler_year.inverse_transform(X3_test)
X3_test = pd.DataFrame(X3_test, columns=['Year'])
X3_train = scaler_year.inverse_transform(X3_train)
X3_train = pd.DataFrame(X3_train, columns=['Year'])
y3_train = scaler_value_production.inverse_transform(y3_train)
y3_train = pd.DataFrame(y3_train, columns=['Value Production (ton)'])
y3_test = scaler_value_production.inverse_transform(y3_test)
y3_test = pd.DataFrame(y3_test, columns=['Value Production (ton)'])
y3_pred = scaler_value_production.inverse_transform(y3_pred)

plt.scatter(X3_test, y3_test, color = 'blue', label = 'Actual')

plt.scatter(X3_test, y3_pred, color = 'red', label = 'Predicted')

plt.plot(X3_test, y3_pred, color = 'black', label = 'Regression Line')

scatter_actual = go.Scatter(x=X3_test['Year'], y=y3_test['Value Production (ton)'], mode='markers', name='Actual', marker=dict(color='LightSkyBlue'))
scatter_predicted = go.Scatter(x=X3_test['Year'], y=y3_pred.flatten(), mode='markers', name='Predicted', marker=dict(color='red'))
scatter_regression = go.Scatter(x=X3_test['Year'], y=y3_pred.flatten(), mode='lines', name='Regression Line', line=dict(color='white', width=2))

data = [scatter_actual, scatter_predicted, scatter_regression]

layout = go.Layout(title='Model 3 (Year)', xaxis=dict(title='Year'), yaxis=dict(title='Value Production (ton)'))

fig = go.Figure(data=data, layout=layout)

st.plotly_chart(fig)

st.markdown("### 5.1.4. Model 4: Menggunakan Year untuk memprediksi Yield")

X4 = dataset_year[['Year']]
y4 = dataset_year[['Value Yield (hg/ha)']]

X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.2, random_state=0)

st.write("Data Training: {} baris".format(X4_train.shape[0]))
st.write("Data Testing: {} baris".format(X4_test.shape[0]))

model4 = LinearRegression()
model4.fit(X4_train, y4_train)
y4_pred = model4.predict(X4_test)

mse_year_yield_lr = mean_squared_error(y4_test, y4_pred)
rmse_year_yield_lr = np.sqrt(mse_year_yield_lr)
r2_year_yield_lr = r2_score(y4_test, y4_pred)

dataframe_mse_rmse_r2_year_yield_lr = pd.DataFrame({'MSE': [mse_year_yield_lr], 'RMSE': [rmse_year_yield_lr], 'R2': [r2_year_yield_lr]}, index=['Model 2 (LR): Year Production'])
st.dataframe(dataframe_mse_rmse_r2_year_yield_lr, width=2000)

y4_test_year_yield_lr_scaled = y4_test
y4_pred_year_yield_lr_scaled = y4_pred

X4_test = scaler_year_year.inverse_transform(X4_test)
X4_test = pd.DataFrame(X4_test, columns=['Year'])
X4_train = scaler_year_year.inverse_transform(X4_train)
X4_train = pd.DataFrame(X4_train, columns=['Year'])
y4_train = scaler_value_yield_year.inverse_transform(y4_train)
y4_train = pd.DataFrame(y4_train, columns=['Value Yield (hg/ha)'])
y4_test = scaler_value_yield_year.inverse_transform(y4_test)
y4_test = pd.DataFrame(y4_test, columns=['Value Yield (hg/ha)'])
y4_pred = scaler_value_yield_year.inverse_transform(y4_pred)

plt.scatter(X4_test, y4_test, color = 'blue', label = 'Actual')

plt.scatter(X4_test, y4_pred, color = 'red', label = 'Predicted')

plt.plot(X4_test, y4_pred, color = 'black', label = 'Regression Line')

scatter_actual = go.Scatter(x=X4_test['Year'], y=y4_test['Value Yield (hg/ha)'], mode='markers', name='Actual', marker=dict(color='LightSkyBlue'))
scatter_predicted = go.Scatter(x=X4_test['Year'], y=y4_pred.flatten(), mode='markers', name='Predicted', marker=dict(color='red'))
scatter_regression = go.Scatter(x=X4_test['Year'], y=y4_pred.flatten(), mode='lines', name='Regression Line', line=dict(color='white', width=2))

data = [scatter_actual, scatter_predicted, scatter_regression]

layout = go.Layout(title='Model 4 (Year)', xaxis=dict(title='Year'), yaxis=dict(title='Value Yield (hg/ha)'))

fig = go.Figure(data=data, layout=layout)

st.plotly_chart(fig)

st.markdown("### 5.1.5. Model 5: Menggunakan Year Dengan Filter Negara untuk memprediksi Yield")

input_area_model5 = st.selectbox('Pilih Area', unique_area_inverse, key='input_area_model5', index=55)

index_input_area_model5 = unique_area_inverse.index(input_area_model5)

dataset_area = dataset[dataset['Area'] == index_input_area_model5]

X5 = dataset_area[['Year']]
y5 = dataset_area[['Value Yield (hg/ha)']]

X5_train, X5_test, y5_train, y5_test = train_test_split(X5, y5, test_size=0.2, random_state=0)

st.write("Data Training: {} baris".format(X5_train.shape[0]))
st.write("Data Testing: {} baris".format(X5_test.shape[0]))

model5 = LinearRegression()
model5.fit(X5_train, y5_train)
y5_pred = model5.predict(X5_test)

mse_year_filter_yield = mean_squared_error(y5_test, y5_pred)
rmse_year_filter_yield = np.sqrt(mse_year_filter_yield)
r2_year_filter_yield = r2_score(y5_test, y5_pred)

dataframe_mse_rmse_r2_year_filter_yield = pd.DataFrame({'MSE': [mse_year_filter_yield], 'RMSE': [rmse_year_filter_yield], 'R2': [r2_year_filter_yield]}, index=['Model 5 (LR): Year Filter Yield'])
st.dataframe(dataframe_mse_rmse_r2_year_filter_yield, width=2000)

X5_test = scaler_year.inverse_transform(X5_test)
X5_test = pd.DataFrame(X5_test, columns=['Year'])
X5_train = scaler_year.inverse_transform(X5_train)
X5_train = pd.DataFrame(X5_train, columns=['Year'])
y5_train = scaler_value_yield.inverse_transform(y5_train)
y5_train = pd.DataFrame(y5_train, columns=['Value Yield (hg/ha)'])
y5_test = scaler_value_yield.inverse_transform(y5_test)
y5_test = pd.DataFrame(y5_test, columns=['Value Yield (hg/ha)'])
y5_pred = scaler_value_yield.inverse_transform(y5_pred)

plt.scatter(X5_test, y5_test, color = 'blue', label = 'Actual')

plt.scatter(X5_test, y5_pred, color = 'red', label = 'Predicted')

plt.plot(X5_test, y5_pred, color = 'black', label = 'Regression Line')

scatter_actual = go.Scatter(x=X5_test['Year'], y=y5_test['Value Yield (hg/ha)'], mode='markers', name='Actual', marker=dict(color='LightSkyBlue'))
scatter_predicted = go.Scatter(x=X5_test['Year'], y=y5_pred.flatten(), mode='markers', name='Predicted', marker=dict(color='red'))
scatter_regression = go.Scatter(x=X5_test['Year'], y=y5_pred.flatten(), mode='lines', name='Regression Line', line=dict(color='white', width=2))

data = [scatter_actual, scatter_predicted, scatter_regression]

layout = go.Layout(title='Model 5 (Year)', xaxis=dict(title='Year'), yaxis=dict(title='Value Yield (hg/ha)'))

fig = go.Figure(data=data, layout=layout)

st.plotly_chart(fig)

st.markdown("### 5.1.6. Model 6: Menggunakan Year untuk memprediksi Area Harvested")

X6 = dataset_year[['Year']]
y6 = dataset_year[['Value Area Harvested (ha)']]


X6_train, X6_test, y6_train, y6_test = train_test_split(X6, y6, test_size=0.2, random_state=0)

st.write("Data Training: {} baris".format(X6_train.shape[0]))
st.write("Data Testing: {} baris".format(X6_test.shape[0]))

model6 = LinearRegression()
model6.fit(X6_train, y6_train)
y6_pred = model6.predict(X6_test)

mse_year_area_harvested_lr = mean_squared_error(y6_test, y6_pred)
rmse_year_area_harvested_lr = np.sqrt(mse_year_area_harvested_lr)
r2_year_area_harvested_lr = r2_score(y6_test, y6_pred)

dataframe_mse_rmse_r2_year_area_harvested_lr = pd.DataFrame({'MSE': [mse_year_area_harvested_lr], 'RMSE': [rmse_year_area_harvested_lr], 'R2': [r2_year_area_harvested_lr]}, index=['Model 6 (LR): Year Area Harvested'])
st.dataframe(dataframe_mse_rmse_r2_year_area_harvested_lr, width=2000)

y6_test_year_area_harvested_lr_scaled = y6_test
y6_pred_year_area_harvested_lr_scaled = y6_pred

X6_test = scaler_year_year.inverse_transform(X6_test)
X6_test = pd.DataFrame(X6_test, columns=['Year'])
X6_train = scaler_year_year.inverse_transform(X6_train)
X6_train = pd.DataFrame(X6_train, columns=['Year'])
y6_train = scaler_value_area_year.inverse_transform(y6_train)
y6_train = pd.DataFrame(y6_train, columns=['Value Area Harvested (ha)'])
y6_test = scaler_value_area_year.inverse_transform(y6_test)
y6_test = pd.DataFrame(y6_test, columns=['Value Area Harvested (ha)'])
y6_pred = scaler_value_area_year.inverse_transform(y6_pred)

plt.scatter(X6_test, y6_test, color = 'blue', label = 'Actual')

plt.scatter(X6_test, y6_pred, color = 'red', label = 'Predicted')

plt.plot(X6_test, y6_pred, color = 'black', label = 'Regression Line')

scatter_actual = go.Scatter(x=X6_test['Year'], y=y6_test['Value Area Harvested (ha)'], mode='markers', name='Actual', marker=dict(color='LightSkyBlue'))
scatter_predicted = go.Scatter(x=X6_test['Year'], y=y6_pred.flatten(), mode='markers', name='Predicted', marker=dict(color='red'))
scatter_regression = go.Scatter(x=X6_test['Year'], y=y6_pred.flatten(), mode='lines', name='Regression Line', line=dict(color='white', width=2))

data = [scatter_actual, scatter_predicted, scatter_regression]

layout = go.Layout(title='Model 6 (Year)', xaxis=dict(title='Year'), yaxis=dict(title='Value Area Harvested (ha)'))

fig = go.Figure(data=data, layout=layout)

st.plotly_chart(fig)

st.markdown("### 5.1.7. Model 7: Menggunakan Year Dengan Filter Negara untuk memprediksi Area Harvested")

input_area_model7 = st.selectbox('Pilih Area', unique_area_inverse, key='input_area_model7', index=55)

index_input_area_model7 = unique_area_inverse.index(input_area_model7)

dataset_area = dataset[dataset['Area'] == index_input_area_model7]

X7 = dataset_area[['Year']]
y7 = dataset_area[['Value Area Harvested (ha)']]

X7_train, X7_test, y7_train, y7_test = train_test_split(X7, y7, test_size=0.2, random_state=0)

st.write("Data Training: {} baris".format(X7_train.shape[0]))
st.write("Data Testing: {} baris".format(X7_test.shape[0]))

model7 = LinearRegression()
model7.fit(X7_train, y7_train)
y7_pred = model7.predict(X7_test)

mse_year_filter_area_harvested = mean_squared_error(y7_test, y7_pred)
rmse_year_filter_area_harvested = np.sqrt(mse_year_filter_area_harvested)
r2_year_filter_area_harvested = r2_score(y7_test, y7_pred)

dataframe_mse_rmse_r2_year_filter_area_harvested = pd.DataFrame({'MSE': [mse_year_filter_area_harvested], 'RMSE': [rmse_year_filter_area_harvested], 'R2': [r2_year_filter_area_harvested]}, index=['Model 7 (LR): Year Filter Area Harvested'])
st.dataframe(dataframe_mse_rmse_r2_year_filter_area_harvested, width=2000)

X7_test = scaler_year.inverse_transform(X7_test)
X7_test = pd.DataFrame(X7_test, columns=['Year'])
X7_train = scaler_year.inverse_transform(X7_train)
X7_train = pd.DataFrame(X7_train, columns=['Year'])
y7_train = scaler_value_area.inverse_transform(y7_train)
y7_train = pd.DataFrame(y7_train, columns=['Value Area Harvested (ha)'])
y7_test = scaler_value_area.inverse_transform(y7_test)
y7_test = pd.DataFrame(y7_test, columns=['Value Area Harvested (ha)'])
y7_pred = scaler_value_area.inverse_transform(y7_pred)

plt.scatter(X7_test, y7_test, color = 'blue', label = 'Actual')

plt.scatter(X7_test, y7_pred, color = 'red', label = 'Predicted')

plt.plot(X7_test, y7_pred, color = 'black', label = 'Regression Line')

scatter_actual = go.Scatter(x=X7_test['Year'], y=y7_test['Value Area Harvested (ha)'], mode='markers', name='Actual', marker=dict(color='LightSkyBlue'))
scatter_predicted = go.Scatter(x=X7_test['Year'], y=y7_pred.flatten(), mode='markers', name='Predicted', marker=dict(color='red'))
scatter_regression = go.Scatter(x=X7_test['Year'], y=y7_pred.flatten(), mode='lines', name='Regression Line', line=dict(color='white', width=2))

data = [scatter_actual, scatter_predicted, scatter_regression]

layout = go.Layout(title='Model 7 (Year)', xaxis=dict(title='Year'), yaxis=dict(title='Value Area Harvested (ha)'))

fig = go.Figure(data=data, layout=layout)

st.plotly_chart(fig)

st.markdown("## 5.2 Forecasting dengan ARIMA")
st.markdown("### 5.2.1. Model 8: Menggunakan Year untuk memprediksi Value Area Harvested (ha)")

dataset_year = combined_dataset.copy()
dataset_year = dataset_year.groupby(['Year']).sum()
dataset_year = dataset_year.reset_index()
dataset_year = dataset_year.drop(['Area'], axis=1)

data_value_area = dataset_year['Value Area Harvested (ha)'].to_numpy().tolist()
data_value_yield = dataset_year['Value Yield (hg/ha)'].to_numpy().tolist()
data_value_production = dataset_year['Value Production (ton)'].to_numpy().tolist()
year = dataset_year['Year'].to_numpy().tolist()

dataset_forecast_value_area = pd.DataFrame(data=[data_value_area], index=['Value Area Harvested (ha)'], columns=year).T
scaler_value_area = MinMaxScaler(feature_range=(0, 1))
dataset_forecast_value_area['Value Area Harvested (ha)'] = scaler_value_area.fit_transform(dataset_forecast_value_area['Value Area Harvested (ha)'].values.reshape(-1, 1))

year_selected = X6_test['Year'].values.tolist()
year_selected = [round(i) for i in year_selected]
year_selected = [int(i) for i in year_selected]
year_selected = sorted(year_selected)

# Plot Auto Correlation untuk menentukan parameter p dan q pada ARIMA
autocorrelation = dataset_forecast_value_area['Value Area Harvested (ha)'].autocorr()
st.write("Autocorrelation : {}".format(autocorrelation))

acf_values = acf(dataset_forecast_value_area['Value Area Harvested (ha)'], nlags=60)
fig = px.line(x=[i for i in range(0, 61)], y=acf_values, title='ACF Plot')
fig.update_layout(xaxis_title='Lags', yaxis_title='ACF Values')
st.plotly_chart(fig)

pacf_values = pacf(dataset_forecast_value_area['Value Area Harvested (ha)'], nlags=25)

fig = px.line(x=[i for i in range(0, 26)], y=pacf_values, title='PACF Plot')
fig.update_layout(xaxis_title='Lags', yaxis_title='PACF Values')
st.plotly_chart(fig)

st.write("1. Nilai P Adalah 17, nilai P adalah orde AP nilai lag yang digunakan untuk membuat model ARIMA. Nilai P didapat dari nilai lag yang signifikan pada plot ACF atau yang pertama kali melewati batas signifikansi (warna biru) pada plot ACF, disini biasanya 0.2 atau 0.3")
st.write("2. Nilai Q Adalah 1, nilai Q adalah order MA yang merupakan jumlah lag yang signifikan turun drastic pada plot PACF. Disini nilai Q adalah 1 karena nilai lag yang signifikan turun drastic pada lag 1")
st.write("3. Nilai D adalah 1 yang mengindikasikan bahwa data yang digunakan adalah data belum stationary sehingga perlu dilakukan differencing sebanyak 1 kali untuk membuat data menjadi stationary")

model_arima_1 = ARIMA(dataset_forecast_value_area['Value Area Harvested (ha)'], order=(17, 1, 1))
model_arima_1_fit = model_arima_1.fit()
y_pred_arima_1 = model_arima_1_fit.predict()


# Plot Residuals Errors
residuals = pd.DataFrame(model_arima_1_fit.resid)
fig = px.line(x=year, y=residuals[0], title='Residuals Errors')
fig.update_layout(xaxis_title='Lags', yaxis_title='Residuals Errors')
st.plotly_chart(fig)

# Lakukan Forecasting
input_forecast_arima_1 = st.number_input('Input Forecasting', min_value=1, value=5, step=1, key='input_forecast_arima_1')
forecast = model_arima_1_fit.forecast(steps=input_forecast_arima_1)
forecast = scaler_value_area.inverse_transform(np.array(forecast).reshape(-1, 1))
forecast = forecast.flatten().tolist()
forecast = [round(i, 2) for i in forecast]
forecast = [0 if i < 0 else i for i in forecast]
forecast = [int(i) for i in forecast]

# Plot Forecasting
y_pred_arima_1_unscalled = scaler_value_area.inverse_transform(np.array(y_pred_arima_1).reshape(-1, 1))

data = pd.DataFrame(data=[data_value_area, y_pred_arima_1_unscalled.flatten().tolist(), forecast], index=['Actual', 'Predict', 'Forecast'], columns=year).T

fig = go.Figure()
# fig add trace untuk menambahkan data ke dalam plot yakni data evaluasi yang dipilih berdasarkan year selected
fig.add_trace(go.Scatter(x=year, y=data['Actual'], mode='lines+markers', name='Actual', line=dict(color='gainsboro', width=2), marker=dict(color='gainsboro', symbol='circle', size=5)))
fig.add_trace(go.Scatter(x=year, y=data['Predict'], mode='lines+markers', name='Predict', line=dict(color='yellow', width=2)))
fig.add_trace(go.Scatter(x=year_selected, y=data['Predict'][year_selected], mode='markers', name='Predict Evaluation', marker=dict(color='blue', symbol='circle', size=7)))
fig.add_trace(go.Scatter(x=[year[-1] + i for i in range(1, input_forecast_arima_1+1)], y=data['Forecast'], mode='lines+markers', name='Forecast', line=dict(color='red', width=2)))

fig.update_layout(xaxis_title='Year', yaxis_title='Value Area Harvested (ha)')
st.plotly_chart(fig)

# MSE, RMSE, R2 Score Berdasarkan Year Selected
df_year_selected_area_harvestd_arima_y_test_y_pred = pd.DataFrame(data=[dataset_forecast_value_area['Value Area Harvested (ha)'][year_selected], y_pred_arima_1[year_selected]], index=['Actual', 'Predict'], columns=year_selected).T
st.dataframe(df_year_selected_area_harvestd_arima_y_test_y_pred, width=2000)

mse_year_selected_area_harvested_arima = mean_squared_error(dataset_forecast_value_area['Value Area Harvested (ha)'][year_selected], y_pred_arima_1[year_selected])
rmse_year_selected_area_harvested_arima = np.sqrt(mse_year_selected_area_harvested_arima)
r2_year_selected_area_harvested_arima = r2_score(dataset_forecast_value_area['Value Area Harvested (ha)'][year_selected], y_pred_arima_1[year_selected])

dataframe_year_selected_area_harvested_arima = pd.DataFrame(data=[[mse_year_selected_area_harvested_arima, rmse_year_selected_area_harvested_arima, r2_year_selected_area_harvested_arima]], columns=['MSE', 'RMSE', 'R2 Score'], index=['Model ARIMA 8 (Year)'])
st.dataframe(dataframe_year_selected_area_harvested_arima, width=2000)

st.markdown("### 5.2.2. Model 9: Menggunakan Year untuk memprediksi Value Yield (hg/ha)")

dataset_forecast_value_yield = pd.DataFrame(data=[data_value_yield], index=['Value Yield (hg/ha)'], columns=year).T
scaler_value_area = MinMaxScaler(feature_range=(0, 1))
dataset_forecast_value_yield['Value Yield (hg/ha)'] = scaler_value_area.fit_transform(dataset_forecast_value_yield['Value Yield (hg/ha)'].values.reshape(-1, 1))

year_selected = X4_test['Year'].values.tolist()
year_selected = [round(i) for i in year_selected]
year_selected = [int(i) for i in year_selected]
year_selected = sorted(year_selected)

# Plot Auto Correlation untuk menentukan parameter p dan q pada ARIMA
autocorrelation = dataset_forecast_value_yield['Value Yield (hg/ha)'].autocorr()
st.write("Autocorrelation : {}".format(autocorrelation))

acf_values = acf(dataset_forecast_value_yield['Value Yield (hg/ha)'], nlags=60)
fig = px.line(x=[i for i in range(0, 61)], y=acf_values, title='ACF Plot')
fig.update_layout(xaxis_title='Lags', yaxis_title='ACF Values')
st.plotly_chart(fig)


pacf_values = pacf(dataset_forecast_value_yield['Value Yield (hg/ha)'], nlags=25)

fig = px.line(x=[i for i in range(0, 26)], y=pacf_values, title='PACF Plot')
fig.update_layout(xaxis_title='Lags', yaxis_title='PACF Values')
st.plotly_chart(fig)

st.write("1. Nilai P Adalah 17, nilai P adalah orde AP nilai lag yang digunakan untuk membuat model ARIMA. Nilai P didapat dari nilai lag yang signifikan pada plot ACF atau yang pertama kali melewati batas signifikansi (warna biru) pada plot ACF, disini biasanya 0.2 atau 0.3")
st.write("2. Nilai Q Adalah 1, nilai Q adalah order MA yang merupakan jumlah lag yang signifikan turun drastic pada plot PACF. Disini nilai Q adalah 1 karena nilai lag yang signifikan turun drastic pada lag 1")
st.write("3. Nilai D adalah 1 yang mengindikasikan bahwa data yang digunakan adalah data belum stationary sehingga perlu dilakukan differencing sebanyak 1 kali untuk membuat data menjadi stationary")

model_arima_2 = ARIMA(dataset_forecast_value_yield['Value Yield (hg/ha)'], order=(17, 1, 1))
model_arima_2_fit = model_arima_2.fit()
y_pred_arima_2 = model_arima_2_fit.predict()

# Plot Residuals Errors
residuals = pd.DataFrame(model_arima_2_fit.resid)
fig = px.line(x=year, y=residuals[0], title='Residuals Errors')
fig.update_layout(xaxis_title='Lags', yaxis_title='Residuals Errors')
st.plotly_chart(fig)

# Lakukan Forecasting
input_forecast_arima_2 = st.number_input('Input Forecasting', min_value=1, value=5, step=1, key='input_forecast_arima_2')
forecast = model_arima_2_fit.forecast(steps=input_forecast_arima_2)
forecast = scaler_value_area.inverse_transform(np.array(forecast).reshape(-1, 1))
forecast = forecast.flatten().tolist()
forecast = [round(i, 2) for i in forecast]
forecast = [0 if i < 0 else i for i in forecast]
forecast = [int(i) for i in forecast]


# Plot Forecasting
y_pred_arima_2_unscalled = scaler_value_area.inverse_transform(np.array(y_pred_arima_2).reshape(-1, 1))

data = pd.DataFrame(data=[data_value_yield, y_pred_arima_2_unscalled.flatten().tolist(), forecast], index=['Actual', 'Predict', 'Forecast'], columns=year).T

fig = go.Figure()
fig.add_trace(go.Scatter(x=year, y=data['Actual'], mode='lines+markers', name='Actual', line=dict(color='gainsboro', width=2), marker=dict(color='gainsboro', symbol='circle', size=5)))
fig.add_trace(go.Scatter(x=year, y=data['Predict'], mode='lines+markers', name='Predict', line=dict(color='yellow', width=2)))
fig.add_trace(go.Scatter(x=year_selected, y=data['Predict'][year_selected], mode='markers', name='Predict Evaluation', marker=dict(color='blue', symbol='circle', size=7)))
fig.add_trace(go.Scatter(x=[year[-1] + i for i in range(1, input_forecast_arima_2+1)], y=data['Forecast'], mode='lines+markers', name='Forecast', line=dict(color='red', width=2)))

fig.update_layout(xaxis_title='Year', yaxis_title='Value Yield (hg/ha)')
st.plotly_chart(fig)

# MSE, RMSE, R2 Score Berdasarkan Year Selected
df_year_selected_yield_arima_y_test_y_pred = pd.DataFrame(data=[dataset_forecast_value_yield['Value Yield (hg/ha)'][year_selected], y_pred_arima_2[year_selected]], index=['Actual', 'Predict'], columns=year_selected).T
st.dataframe(df_year_selected_yield_arima_y_test_y_pred, width=2000)

mse_year_selected_yield_arima = mean_squared_error(dataset_forecast_value_yield['Value Yield (hg/ha)'][year_selected], y_pred_arima_2[year_selected])
rmse_year_selected_yield_arima = np.sqrt(mse_year_selected_yield_arima)
r2_year_selected_yield_arima = r2_score(dataset_forecast_value_yield['Value Yield (hg/ha)'][year_selected], y_pred_arima_2[year_selected])

dataframe_year_selected_yield_arima = pd.DataFrame(data=[[mse_year_selected_yield_arima, rmse_year_selected_yield_arima, r2_year_selected_yield_arima]], columns=['MSE', 'RMSE', 'R2 Score'], index=['Model ARIMA 9 (Year)'])
st.dataframe(dataframe_year_selected_yield_arima, width=2000)

st.markdown("### 5.2.3. Model 10: Menggunakan Year untuk memprediksi Value Production (ton)")

dataset_forecast_value_production = pd.DataFrame(data=[data_value_production], index=['Value Production (ton)'], columns=year).T
scaler_value_production = MinMaxScaler(feature_range=(0, 1))
dataset_forecast_value_production['Value Production (ton)'] = scaler_value_production.fit_transform(dataset_forecast_value_production['Value Production (ton)'].values.reshape(-1, 1))

year_selected = X2_test['Year'].values.tolist()
year_selected = [round(i) for i in year_selected]
year_selected = [int(i) for i in year_selected]
year_selected = sorted(year_selected)

# Plot Auto Correlation untuk menentukan parameter p dan q pada ARIMA
autocorrelation = dataset_forecast_value_production['Value Production (ton)'].autocorr()
st.write("Autocorrelation : {}".format(autocorrelation))

acf_values = acf(dataset_forecast_value_production['Value Production (ton)'], nlags=60)
fig = px.line(x=[i for i in range(0, 61)], y=acf_values, title='ACF Plot')
fig.update_layout(xaxis_title='Lags', yaxis_title='ACF Values')
st.plotly_chart(fig)

pacf_values = pacf(dataset_forecast_value_production['Value Production (ton)'], nlags=25)

fig = px.line(x=[i for i in range(0, 26)], y=pacf_values, title='PACF Plot')
fig.update_layout(xaxis_title='Lags', yaxis_title='PACF Values')
st.plotly_chart(fig)

st.write("1. Nilai P Adalah 17, nilai P adalah orde AP nilai lag yang digunakan untuk membuat model ARIMA. Nilai P didapat dari nilai lag yang signifikan pada plot ACF atau yang pertama kali melewati batas signifikansi (warna biru) pada plot ACF, disini biasanya 0.2 atau 0.3")
st.write("2. Nilai Q Adalah 1, nilai Q adalah order MA yang merupakan jumlah lag yang signifikan turun drastic pada plot PACF. Disini nilai Q adalah 1 karena nilai lag yang signifikan turun drastic pada lag 1")
st.write("3. Nilai D adalah 1 yang mengindikasikan bahwa data yang digunakan adalah data belum stationary sehingga perlu dilakukan differencing sebanyak 1 kali untuk membuat data menjadi stationary")

model_arima_3 = ARIMA(dataset_forecast_value_production['Value Production (ton)'], order=(17, 1, 1))
model_arima_3_fit = model_arima_3.fit()
y_pred_arima_3 = model_arima_3_fit.predict()

# Plot Residuals Errors
residuals = pd.DataFrame(model_arima_3_fit.resid)
fig = px.line(x=year, y=residuals[0], title='Residuals Errors')
fig.update_layout(xaxis_title='Lags', yaxis_title='Residuals Errors')
st.plotly_chart(fig)

# Lakukan Forecasting
input_forecast_arima_3 = st.number_input('Input Forecasting', min_value=1, value=5, step=1, key='input_forecast_arima_3')
forecast = model_arima_3_fit.forecast(steps=input_forecast_arima_3)
forecast = scaler_value_production.inverse_transform(np.array(forecast).reshape(-1, 1))
forecast = forecast.flatten().tolist()
forecast = [round(i, 2) for i in forecast]
forecast = [0 if i < 0 else i for i in forecast]
forecast = [int(i) for i in forecast]

# Plot Forecasting
y_pred_arima_3_unscalled = scaler_value_production.inverse_transform(np.array(y_pred_arima_3).reshape(-1, 1))

data = pd.DataFrame(data=[data_value_production, y_pred_arima_3_unscalled.flatten().tolist(), forecast], index=['Actual', 'Predict', 'Forecast'], columns=year).T

fig = go.Figure()
fig.add_trace(go.Scatter(x=year, y=data['Actual'], mode='lines+markers', name='Actual', line=dict(color='gainsboro', width=2), marker=dict(color='gainsboro', symbol='circle', size=5)))
fig.add_trace(go.Scatter(x=year, y=data['Predict'], mode='lines+markers', name='Predict', line=dict(color='yellow', width=2)))
fig.add_trace(go.Scatter(x=year_selected, y=data['Predict'][year_selected], mode='markers', name='Predict Evaluation', marker=dict(color='blue', symbol='circle', size=7)))
fig.add_trace(go.Scatter(x=[year[-1] + i for i in range(1, input_forecast_arima_3+1)], y=data['Forecast'], mode='lines+markers', name='Forecast', line=dict(color='red', width=2)))

fig.update_layout(xaxis_title='Year', yaxis_title='Value Production (ton)')
st.plotly_chart(fig)

# MSE, RMSE, R2 Score Berdasarkan Year Selected
df_year_selected_production_arima_y_test_y_pred = pd.DataFrame(data=[dataset_forecast_value_production['Value Production (ton)'][year_selected], y_pred_arima_3[year_selected]], index=['Actual', 'Predict'], columns=year_selected).T
st.dataframe(df_year_selected_production_arima_y_test_y_pred, width=2000)

mse_year_selected_production_arima = mean_squared_error(dataset_forecast_value_production['Value Production (ton)'][year_selected], y_pred_arima_3[year_selected])
rmse_year_selected_production_arima = np.sqrt(mse_year_selected_production_arima)
r2_year_selected_production_arima = r2_score(dataset_forecast_value_production['Value Production (ton)'][year_selected], y_pred_arima_3[year_selected])

dataframe_year_selected_production_arima = pd.DataFrame(data=[[mse_year_selected_production_arima, rmse_year_selected_production_arima, r2_year_selected_production_arima]], columns=['MSE', 'RMSE', 'R2 Score'], index=['Model ARIMA 10 (Year)'])
st.dataframe(dataframe_year_selected_production_arima, width=2000)

st.markdown("### 5.3. Ensemble Model")
st.markdown("### 5.3.1. Model 11: Menggunakan Year untuk memprediksi Area Harvested (ha)")

year_selected = X6_test['Year'].values.tolist()
year_selected = [round(i) for i in year_selected]
year_selected = [int(i) for i in year_selected]

y_pred_arima_1_unscalled = pd.DataFrame(y_pred_arima_1_unscalled)
y_pred_arima_1_unscalled.index = year
y_pred_arima_1_unscalled = y_pred_arima_1_unscalled.loc[year_selected]

y6_pred = pd.DataFrame(y6_pred)
y6_pred.index = year_selected


dataset_ensemble_area_harvested = pd.DataFrame()
dataset_ensemble_area_harvested['Year'] = year_selected
dataset_ensemble_area_harvested['Actual'] = y6_test
dataset_ensemble_area_harvested['ARIMA'] = y_pred_arima_1_unscalled.values
dataset_ensemble_area_harvested['Linear Regression'] = y6_pred.values
dataset_ensemble_area_harvested['ARIMA DIFF'] = abs(dataset_ensemble_area_harvested['Actual'] - dataset_ensemble_area_harvested['ARIMA'])
dataset_ensemble_area_harvested['LR DIFF'] = abs(dataset_ensemble_area_harvested['Actual'] - dataset_ensemble_area_harvested['Linear Regression'])
dataset_ensemble_area_harvested["Ensemble"] = dataset_ensemble_area_harvested.loc[:, ["ARIMA DIFF", "LR DIFF"]].idxmin(axis=1).map(lambda x: 'ARIMA' if x == 'ARIMA DIFF' else 'Linear Regression')
dataset_ensemble_area_harvested["Ensemble Value"] = dataset_ensemble_area_harvested.apply(lambda row: row['ARIMA'] if row['Ensemble'] == 'ARIMA' else row['Linear Regression'], axis=1)

st.write(dataset_ensemble_area_harvested)

dataset_ensemble_area_harvested = dataset_ensemble_area_harvested.sort_values(by=['Year'])

# Buat Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=dataset_ensemble_area_harvested['Year'], y=dataset_ensemble_area_harvested['Actual'], mode='markers', name='Actual', line=dict(color='gainsboro', width=2), marker=dict(color='gainsboro', symbol='circle', size=5)))
fig.add_trace(go.Scatter(x=dataset_ensemble_area_harvested['Year'], y=dataset_ensemble_area_harvested['ARIMA'], mode='markers', name='ARIMA', line=dict(color='yellow', width=2)))
fig.add_trace(go.Scatter(x=dataset_ensemble_area_harvested['Year'], y=dataset_ensemble_area_harvested['Linear Regression'], mode='markers', name='Linear Regression', line=dict(color='red', width=2)))
fig.add_trace(go.Scatter(x=dataset_ensemble_area_harvested['Year'], y=dataset_ensemble_area_harvested['Ensemble Value'], mode='lines+markers', name='Ensemble', line=dict(color='blue', width=2)))

fig.update_layout(xaxis_title='Year', yaxis_title='Area Harvested (ha)')
st.plotly_chart(fig)

# Evaluasi Model dengan MSE, RMSE, dan R2 Untuk ARIMA
mse_area_harvested_arima = mean_squared_error(df_year_selected_area_harvestd_arima_y_test_y_pred['Actual'], df_year_selected_area_harvestd_arima_y_test_y_pred['Predict'])
rmse_area_harvested_arima = np.sqrt(mean_squared_error(df_year_selected_area_harvestd_arima_y_test_y_pred['Actual'], df_year_selected_area_harvestd_arima_y_test_y_pred['Predict']))
r2_area_harvested_arima = r2_score(df_year_selected_area_harvestd_arima_y_test_y_pred['Actual'], df_year_selected_area_harvestd_arima_y_test_y_pred['Predict'])

# Evaluasi Model dengan MSE, RMSE, dan R2 Untuk Linear Regression
mse_area_harvested_lr = mean_squared_error(y6_test_year_area_harvested_lr_scaled, y6_pred_year_area_harvested_lr_scaled)
rmse_area_harvested_lr = np.sqrt(mean_squared_error(y6_test_year_area_harvested_lr_scaled, y6_pred_year_area_harvested_lr_scaled))
r2_area_harvested_lr = r2_score(y6_test_year_area_harvested_lr_scaled, y6_pred_year_area_harvested_lr_scaled)

year_selected = X6_test['Year'].values.tolist()
year_selected = [round(i) for i in year_selected]
year_selected = [int(i) for i in year_selected]
dataframe_y6_area_harvested_lr = pd.DataFrame(data=[year_selected, y6_test_year_area_harvested_lr_scaled.values, np.array(y6_pred_year_area_harvested_lr_scaled)], index=['Year', 'Actual', 'Predict'], columns=year_selected).T
dataframe_y6_area_harvested_lr = dataframe_y6_area_harvested_lr.sort_values(by=['Year'])
y6_pred_year_area_harvested_lr_scaled = dataframe_y6_area_harvested_lr['Predict'].to_numpy().astype(float)

dataset_ensemble_area_harvested = df_year_selected_area_harvestd_arima_y_test_y_pred.copy()
dataset_ensemble_area_harvested.rename(columns={'Predict': 'ARIMA'}, inplace=True)
dataset_ensemble_area_harvested['ARIMA'] = df_year_selected_area_harvestd_arima_y_test_y_pred['Predict'].values
dataset_ensemble_area_harvested['Linear Regression'] = y6_pred_year_area_harvested_lr_scaled
dataset_ensemble_area_harvested['ARIMA DIFF'] = abs(dataset_ensemble_area_harvested['Actual'] - dataset_ensemble_area_harvested['ARIMA'])
dataset_ensemble_area_harvested['LR DIFF'] = abs(dataset_ensemble_area_harvested['Actual'] - dataset_ensemble_area_harvested['Linear Regression'])
dataset_ensemble_area_harvested["Ensemble"] = dataset_ensemble_area_harvested.loc[:, ["ARIMA DIFF", "LR DIFF"]].idxmin(axis=1).map(lambda x: 'ARIMA' if x == 'ARIMA DIFF' else 'Linear Regression')
dataset_ensemble_area_harvested["Ensemble Value"] = dataset_ensemble_area_harvested.apply(lambda row: row['ARIMA'] if row['Ensemble'] == 'ARIMA' else row['Linear Regression'], axis=1)
st.dataframe(dataset_ensemble_area_harvested, width=2000)

# Evaluasi Model dengan MSE, RMSE, dan R2 Ensemble
mse_area_harvested_ensemble = mean_squared_error(dataset_ensemble_area_harvested['Actual'], dataset_ensemble_area_harvested['Ensemble Value'])
rmse_area_harvested_ensemble = np.sqrt(mean_squared_error(dataset_ensemble_area_harvested['Actual'], dataset_ensemble_area_harvested['Ensemble Value']))
r2_area_harvested_ensemble = r2_score(dataset_ensemble_area_harvested['Actual'], dataset_ensemble_area_harvested['Ensemble Value'])

# Buatkan DataFrame untuk Evaluasi Model
data_eval_area_harvested = pd.DataFrame(data=[['ARIMA', mse_area_harvested_arima, rmse_area_harvested_arima, r2_area_harvested_arima], ['Linear Regression', mse_area_harvested_lr, rmse_area_harvested_lr, r2_area_harvested_lr], ['Ensemble', mse_area_harvested_ensemble, rmse_area_harvested_ensemble, r2_area_harvested_ensemble]], columns=['Model', 'MSE', 'RMSE', 'R2'])
data_eval_area_harvested = data_eval_area_harvested.set_index('Model')
st.dataframe(data_eval_area_harvested, width=2000)

st.markdown("### 5.3.2. Model 12: Menggunakan Year untuk memprediksi Yield (hg/ha)")

year_selected = X4_test['Year'].values.tolist()
year_selected = [round(i) for i in year_selected]
year_selected = [int(i) for i in year_selected]

y_pred_arima_2_unscalled = pd.DataFrame(y_pred_arima_2_unscalled)
y_pred_arima_2_unscalled.index = year
y_pred_arima_2_unscalled = y_pred_arima_2_unscalled.loc[year_selected]

y4_pred = pd.DataFrame(y4_pred)
y4_pred.index = year_selected

dataset_ensemble_yield = pd.DataFrame()
dataset_ensemble_yield['Year'] = year_selected
dataset_ensemble_yield['Actual'] = y4_test
dataset_ensemble_yield['ARIMA'] = y_pred_arima_2_unscalled.values
dataset_ensemble_yield['Linear Regression'] = y4_pred.values
dataset_ensemble_yield['ARIMA DIFF'] = abs(dataset_ensemble_yield['Actual'] - dataset_ensemble_yield['ARIMA'])
dataset_ensemble_yield['LR DIFF'] = abs(dataset_ensemble_yield['Actual'] - dataset_ensemble_yield['Linear Regression'])
dataset_ensemble_yield["Ensemble"] = dataset_ensemble_yield.loc[:, ["ARIMA DIFF", "LR DIFF"]].idxmin(axis=1).map(lambda x: 'ARIMA' if x == 'ARIMA DIFF' else 'Linear Regression')
dataset_ensemble_yield["Ensemble Value"] = dataset_ensemble_yield.apply(lambda row: row['ARIMA'] if row['Ensemble'] == 'ARIMA' else row['Linear Regression'], axis=1)

st.dataframe(dataset_ensemble_yield)


dataset_ensemble_yield = dataset_ensemble_yield.sort_values(by=['Year'])

# Buat Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=dataset_ensemble_yield['Year'], y=dataset_ensemble_yield['Actual'], mode='markers', name='Actual', line=dict(color='gainsboro', width=2), marker=dict(color='gainsboro', symbol='circle', size=5)))
fig.add_trace(go.Scatter(x=dataset_ensemble_yield['Year'], y=dataset_ensemble_yield['ARIMA'], mode='markers', name='ARIMA', line=dict(color='yellow', width=2)))
fig.add_trace(go.Scatter(x=dataset_ensemble_yield['Year'], y=dataset_ensemble_yield['Linear Regression'], mode='markers', name='Linear Regression', line=dict(color='red', width=2)))
fig.add_trace(go.Scatter(x=dataset_ensemble_yield['Year'], y=dataset_ensemble_yield['Ensemble Value'], mode='lines+markers', name='Ensemble', line=dict(color='blue', width=2)))

fig.update_layout(xaxis_title='Year', yaxis_title='Yield (hg/ha)')
st.plotly_chart(fig)


# Evaluasi Model dengan MSE, RMSE, dan R2 Untuk ARIMA
mse_yield_arima = mean_squared_error(df_year_selected_yield_arima_y_test_y_pred['Actual'], df_year_selected_yield_arima_y_test_y_pred['Predict'])
rmse_yield_arima = np.sqrt(mean_squared_error(df_year_selected_yield_arima_y_test_y_pred['Actual'], df_year_selected_yield_arima_y_test_y_pred['Predict']))
r2_yield_arima = r2_score(df_year_selected_yield_arima_y_test_y_pred['Actual'], df_year_selected_yield_arima_y_test_y_pred['Predict'])

# Evaluasi Model dengan MSE, RMSE, dan R2 Untuk Linear Regression
mse_yield_lr = mean_squared_error(y4_test_year_yield_lr_scaled, y4_pred_year_yield_lr_scaled)
rmse_yield_lr = np.sqrt(mean_squared_error(y4_test_year_yield_lr_scaled, y4_pred_year_yield_lr_scaled))
r2_yield_lr = r2_score(y4_test_year_yield_lr_scaled, y4_pred_year_yield_lr_scaled)

year_selected = X4_test['Year'].values.tolist()
year_selected = [round(i) for i in year_selected]
year_selected = [int(i) for i in year_selected]
dataframe_y4_yield_lr = pd.DataFrame(data=[year_selected, y4_test_year_yield_lr_scaled.values, np.array(y4_pred_year_yield_lr_scaled)], index=['Year', 'Actual', 'Predict'], columns=year_selected).T
dataframe_y4_yield_lr = dataframe_y4_yield_lr.sort_values(by=['Year'])
y4_pred_year_yield_lr_scaled = dataframe_y4_yield_lr['Predict'].to_numpy().astype(float)

dataset_ensemble_yield = df_year_selected_yield_arima_y_test_y_pred.copy()
dataset_ensemble_yield.rename(columns={'Predict': 'ARIMA'}, inplace=True)
dataset_ensemble_yield['ARIMA'] = df_year_selected_yield_arima_y_test_y_pred['Predict'].values
dataset_ensemble_yield['Linear Regression'] = y4_pred_year_yield_lr_scaled
dataset_ensemble_yield['ARIMA DIFF'] = abs(dataset_ensemble_yield['Actual'] - dataset_ensemble_yield['ARIMA'])
dataset_ensemble_yield['LR DIFF'] = abs(dataset_ensemble_yield['Actual'] - dataset_ensemble_yield['Linear Regression'])
dataset_ensemble_yield["Ensemble"] = dataset_ensemble_yield.loc[:, ["ARIMA DIFF", "LR DIFF"]].idxmin(axis=1).map(lambda x: 'ARIMA' if x == 'ARIMA DIFF' else 'Linear Regression')
dataset_ensemble_yield["Ensemble Value"] = dataset_ensemble_yield.apply(lambda row: row['ARIMA'] if row['Ensemble'] == 'ARIMA' else row['Linear Regression'], axis=1)
st.dataframe(dataset_ensemble_yield, width=2000)

# Evaluasi Model dengan MSE, RMSE, dan R2 Ensemble
mse_yield_ensemble = mean_squared_error(dataset_ensemble_yield['Actual'], dataset_ensemble_yield['Ensemble Value'])
rmse_yield_ensemble = np.sqrt(mean_squared_error(dataset_ensemble_yield['Actual'], dataset_ensemble_yield['Ensemble Value']))
r2_yield_ensemble = r2_score(dataset_ensemble_yield['Actual'], dataset_ensemble_yield['Ensemble Value'])

# Buatkan DataFrame untuk Evaluasi Model
data_eval_yield = pd.DataFrame(data=[['ARIMA', mse_yield_arima, rmse_yield_arima, r2_yield_arima], ['Linear Regression', mse_yield_lr, rmse_yield_lr, r2_yield_lr], ['Ensemble', mse_yield_ensemble, rmse_yield_ensemble, r2_yield_ensemble]], columns=['Model', 'MSE', 'RMSE', 'R2'])
data_eval_yield = data_eval_yield.set_index('Model')
st.dataframe(data_eval_yield, width=2000)

st.markdown("### 5.3.3. Model 13: Menggunakan Year untuk memprediksi Production (ton)")

year_selected = X2_test['Year'].values.tolist()
year_selected = [round(i) for i in year_selected]
year_selected = [int(i) for i in year_selected]

y_pred_arima_3_unscalled = pd.DataFrame(y_pred_arima_3_unscalled)
y_pred_arima_3_unscalled.index = year
y_pred_arima_3_unscalled = y_pred_arima_3_unscalled.loc[year_selected]

y2_pred = pd.DataFrame(y2_pred)
y2_pred.index = year_selected


dataset_ensemble_production = pd.DataFrame()
dataset_ensemble_production['Year'] = year_selected
dataset_ensemble_production['Actual'] = y2_test
dataset_ensemble_production['ARIMA'] = y_pred_arima_3_unscalled.values
dataset_ensemble_production['Linear Regression'] = y2_pred.values
dataset_ensemble_production['ARIMA DIFF'] = abs(dataset_ensemble_production['Actual'] - dataset_ensemble_production['ARIMA'])
dataset_ensemble_production['LR DIFF'] = abs(dataset_ensemble_production['Actual'] - dataset_ensemble_production['Linear Regression'])
dataset_ensemble_production["Ensemble"] = dataset_ensemble_production.loc[:, ["ARIMA DIFF", "LR DIFF"]].idxmin(axis=1).map(lambda x: 'ARIMA' if x == 'ARIMA DIFF' else 'Linear Regression')
dataset_ensemble_production["Ensemble Value"] = dataset_ensemble_production.apply(lambda row: row['ARIMA'] if row['Ensemble'] == 'ARIMA' else row['Linear Regression'], axis=1)

st.write(dataset_ensemble_production)

dataset_ensemble_production = dataset_ensemble_production.sort_values(by=['Year'])

# Buat Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=dataset_ensemble_production['Year'], y=dataset_ensemble_production['Actual'], mode='markers', name='Actual', line=dict(color='gainsboro', width=2), marker=dict(color='gainsboro', symbol='circle', size=5)))
fig.add_trace(go.Scatter(x=dataset_ensemble_production['Year'], y=dataset_ensemble_production['ARIMA'], mode='markers', name='ARIMA', line=dict(color='yellow', width=2)))
fig.add_trace(go.Scatter(x=dataset_ensemble_production['Year'], y=dataset_ensemble_production['Linear Regression'], mode='markers', name='Linear Regression', line=dict(color='red', width=2)))
fig.add_trace(go.Scatter(x=dataset_ensemble_production['Year'], y=dataset_ensemble_production['Ensemble Value'], mode='lines+markers', name='Ensemble', line=dict(color='blue', width=2)))

fig.update_layout(xaxis_title='Year', yaxis_title='Production (ton)')
st.plotly_chart(fig)

# Evaluasi Model dengan MSE, RMSE, dan R2 Untuk ARIMA
mse_production_arima = mean_squared_error(df_year_selected_production_arima_y_test_y_pred['Actual'], df_year_selected_production_arima_y_test_y_pred['Predict'])
rmse_production_arima = np.sqrt(mean_squared_error(df_year_selected_production_arima_y_test_y_pred['Actual'], df_year_selected_production_arima_y_test_y_pred['Predict']))
r2_production_arima = r2_score(df_year_selected_production_arima_y_test_y_pred['Actual'], df_year_selected_production_arima_y_test_y_pred['Predict'])

# Evaluasi Model dengan MSE, RMSE, dan R2 Untuk Linear Regression
mse_production_lr = mean_squared_error(y2_test_year_production_lr_scaled, y2_pred_year_production_lr_scaled)
rmse_production_lr = np.sqrt(mean_squared_error(y2_test_year_production_lr_scaled, y2_pred_year_production_lr_scaled))
r2_production_lr = r2_score(y2_test_year_production_lr_scaled, y2_pred_year_production_lr_scaled)

year_selected = X2_test['Year'].values.tolist()
year_selected = [round(i) for i in year_selected]
year_selected = [int(i) for i in year_selected]
dataframe_y2_production_lr = pd.DataFrame(data=[year_selected, y2_test_year_production_lr_scaled.values, np.array(y2_pred_year_production_lr_scaled)], index=['Year', 'Actual', 'Predict'], columns=year_selected).T
dataframe_y2_production_lr = dataframe_y2_production_lr.sort_values(by=['Year'])
y2_pred_year_production_lr_scaled = dataframe_y2_production_lr['Predict'].to_numpy().astype(float)

dataset_ensemble_production = df_year_selected_production_arima_y_test_y_pred.copy()
dataset_ensemble_production.rename(columns={'Predict': 'ARIMA'}, inplace=True)
dataset_ensemble_production['ARIMA'] = df_year_selected_production_arima_y_test_y_pred['Predict'].values
dataset_ensemble_production['Linear Regression'] = y2_pred_year_production_lr_scaled
dataset_ensemble_production['ARIMA DIFF'] = abs(dataset_ensemble_production['Actual'] - dataset_ensemble_production['ARIMA'])
dataset_ensemble_production['LR DIFF'] = abs(dataset_ensemble_production['Actual'] - dataset_ensemble_production['Linear Regression'])
dataset_ensemble_production["Ensemble"] = dataset_ensemble_production.loc[:, ["ARIMA DIFF", "LR DIFF"]].idxmin(axis=1).map(lambda x: 'ARIMA' if x == 'ARIMA DIFF' else 'Linear Regression')
dataset_ensemble_production["Ensemble Value"] = dataset_ensemble_production.apply(lambda row: row['ARIMA'] if row['Ensemble'] == 'ARIMA' else row['Linear Regression'], axis=1)
st.dataframe(dataset_ensemble_production, width=2000)

# Evaluasi Model dengan MSE, RMSE, dan R2 Ensemble
mse_production_ensemble = mean_squared_error(dataset_ensemble_production['Actual'], dataset_ensemble_production['Ensemble Value'])
rmse_production_ensemble = np.sqrt(mean_squared_error(dataset_ensemble_production['Actual'], dataset_ensemble_production['Ensemble Value']))
r2_production_ensemble = r2_score(dataset_ensemble_production['Actual'], dataset_ensemble_production['Ensemble Value'])

# Buatkan DataFrame untuk Evaluasi Model
data_eval_production = pd.DataFrame(data=[['ARIMA', mse_production_arima, rmse_production_arima, r2_production_arima], ['Linear Regression', mse_production_lr, rmse_production_lr, r2_production_lr], ['Ensemble', mse_production_ensemble, rmse_production_ensemble, r2_production_ensemble]], columns=['Model', 'MSE', 'RMSE', 'R2'])
data_eval_production = data_eval_production.set_index('Model')
st.dataframe(data_eval_production, width=2000)

st.markdown("# 6. Evaluasi Model")
st.markdown("## 6.1 Evaluasi Model Tunggal")

# Evaluasi Model dengan MSE, RMSE, dan R2 Untuk Linear Regression
mse_area_harvested_production_dataframe = pd.DataFrame(data=[['Model 1: Linear Regression (Area Harvested -> Production)', mse_area_harvested_production, rmse_area_harvested_production, r2_area_harvested_production]], columns=['Model', 'MSE', 'RMSE', 'R2']) 
mse_year_production_lr_dataframe = pd.DataFrame(data=[['Model 2: Linear Regression (Year -> Production)', mse_year_production_lr, rmse_year_production_lr, r2_year_production_lr]], columns=['Model', 'MSE', 'RMSE', 'R2'])
mse_year_filter_production_dataframe = pd.DataFrame(data=[['Model 3: Linear Regression (Year {} -> Production)'.format(input_area_model3), mse_year_filter_production,rmse_year_filter_production, r2_year_filter_production]], columns=['Model', 'MSE', 'RMSE', 'R2'])
mse_year_yield_lr_dataframe = pd.DataFrame(data=[['Model 4: Linear Regression (Year -> Yield)', mse_year_yield_lr, rmse_year_yield_lr, r2_year_yield_lr]], columns=['Model', 'MSE', 'RMSE', 'R2'])
mse_year_filter_yield_dataframe = pd.DataFrame(data=[['Model 5: Linear Regression (Year {} -> Yield)'.format(input_area_model5), mse_year_filter_yield, rmse_year_filter_yield, r2_year_filter_yield]], columns=['Model', 'MSE', 'RMSE', 'R2'])
mse_year_area_harvested_lr_dataframe = pd.DataFrame(data=[['Model 6: Linear Regression (Year -> Area Harvested)', mse_year_area_harvested_lr, rmse_year_area_harvested_lr, r2_year_area_harvested_lr]], columns=['Model', 'MSE', 'RMSE', 'R2'])
mse_year_filter_area_harvested_dataframe = pd.DataFrame(data=[['Model 7: Linear Regression (Year {} -> Area Harvested)'.format(input_area_model7), mse_year_filter_area_harvested, rmse_year_filter_area_harvested, r2_year_filter_area_harvested]], columns=['Model', 'MSE', 'RMSE', 'R2'])
mse_year_selected_production_arima_dataframe = pd.DataFrame(data=[['Model 8: ARIMA (Year -> Production)', mse_year_selected_production_arima, rmse_year_selected_production_arima, r2_year_selected_production_arima]], columns=['Model', 'MSE', 'RMSE', 'R2'])
mse_year_selected_yield_arima_dataframe = pd.DataFrame(data=[['Model 9: ARIMA (Year -> Yield)', mse_year_selected_yield_arima, rmse_year_selected_yield_arima, r2_year_selected_yield_arima]], columns=['Model', 'MSE', 'RMSE', 'R2'])
mse_year_selected_area_harvested_arima_dataframe = pd.DataFrame(data=[['Model 10: ARIMA (Year -> Area Harvested)', mse_year_selected_area_harvested_arima, rmse_year_selected_area_harvested_arima, r2_year_selected_area_harvested_arima]], columns=['Model', 'MSE', 'RMSE', 'R2'])

mse_area_harvested_production_dataframe = mse_area_harvested_production_dataframe.set_index('Model')
mse_year_production_lr_dataframe = mse_year_production_lr_dataframe.set_index('Model')
mse_year_filter_production_dataframe = mse_year_filter_production_dataframe.set_index('Model')
mse_year_yield_lr_dataframe = mse_year_yield_lr_dataframe.set_index('Model')
mse_year_filter_yield_dataframe = mse_year_filter_yield_dataframe.set_index('Model')
mse_year_area_harvested_lr_dataframe = mse_year_area_harvested_lr_dataframe.set_index('Model')
mse_year_filter_area_harvested_dataframe = mse_year_filter_area_harvested_dataframe.set_index('Model')
mse_year_selected_production_arima_dataframe = mse_year_selected_production_arima_dataframe.set_index('Model')
mse_year_selected_yield_arima_dataframe = mse_year_selected_yield_arima_dataframe.set_index('Model')
mse_year_selected_area_harvested_arima_dataframe = mse_year_selected_area_harvested_arima_dataframe.set_index('Model')

mse_dataframe = pd.concat([mse_area_harvested_production_dataframe, mse_year_production_lr_dataframe, mse_year_filter_production_dataframe, mse_year_yield_lr_dataframe, mse_year_filter_yield_dataframe, mse_year_area_harvested_lr_dataframe, mse_year_filter_area_harvested_dataframe, mse_year_selected_production_arima_dataframe, mse_year_selected_yield_arima_dataframe, mse_year_selected_area_harvested_arima_dataframe])
st.dataframe(mse_dataframe, width=2000)

st.markdown("## 6.2 Evaluasi Model Ensemble")
st.markdown("### 6.2.1 Ensemble LR dan ARIMA Pada (Year -> Area Harvested)")
st.write("Model yang di ensemble dengan teknik voting adalah model 6 (Linear Regression) dan model 10 (ARIMA))")
st.dataframe(data_eval_area_harvested, width=2000)
st.markdown("### 6.2.2 Ensemble LR dan ARIMA Pada (Year -> Yield)")
st.write("Model yang di ensemble dengan teknik voting adalah model 4 (Linear Regression) dan model 9 (ARIMA))")
st.dataframe(data_eval_yield, width=2000)
st.markdown("### 6.2.3 Ensemble LR dan ARIMA Pada (Year -> Production)")
st.write("Model yang di ensemble dengan teknik voting adalah model 2 (Linear Regression) dan model 8 (ARIMA))")
st.dataframe(data_eval_production, width=2000)
st.markdown("## 6.3 Plot Evaluasi")
st.markdown("### 6.3.1 Plot Evaluasi Model Tunggal")

# Buat Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=mse_dataframe.index, y=mse_dataframe['MSE'], mode='lines+markers', name='MSE', line=dict(color='blue', width=2)))
fig.add_trace(go.Scatter(x=mse_dataframe.index, y=mse_dataframe['RMSE'], mode='lines+markers', name='RMSE', line=dict(color='red', width=2)))
fig.add_trace(go.Scatter(x=mse_dataframe.index, y=mse_dataframe['R2'], mode='lines+markers', name='R2', line=dict(color='yellow', width=2)))

fig.update_layout(xaxis_title='Model', yaxis_title='Nilai Evaluasi')
st.plotly_chart(fig)

st.markdown("### 6.3.2 Plot Evaluasi Model Ensemble")

# Buat Plot dalama bentuk bar chart kemudian dari masing-masing data_eval_area_harvested, data_eval_yield, dan data_eval_production
fig = go.Figure()

fig.add_trace(go.Bar(x=data_eval_area_harvested.index, y=data_eval_area_harvested['MSE'], name='MSE - Area Harvested', marker_color='blue'))
fig.add_trace(go.Bar(x=data_eval_area_harvested.index, y=data_eval_area_harvested['RMSE'], name='RMSE - Area Harvested', marker_color='red'))
fig.add_trace(go.Bar(x=data_eval_area_harvested.index, y=data_eval_area_harvested['R2'], name='R2 - Area Harvested', marker_color='yellow'))

fig.add_trace(go.Bar(x=data_eval_yield.index, y=data_eval_yield['MSE'], name='MSE - Yield', marker_color='green'))
fig.add_trace(go.Bar(x=data_eval_yield.index, y=data_eval_yield['RMSE'], name='RMSE - Yield', marker_color='purple'))
fig.add_trace(go.Bar(x=data_eval_yield.index, y=data_eval_yield['R2'], name='R2 - Yield', marker_color='orange'))

fig.add_trace(go.Bar(x=data_eval_production.index, y=data_eval_production['MSE'], name='MSE - Production', marker_color='brown'))
fig.add_trace(go.Bar(x=data_eval_production.index, y=data_eval_production['RMSE'], name='RMSE - Production', marker_color='gray'))
fig.add_trace(go.Bar(x=data_eval_production.index, y=data_eval_production['R2'], name='R2 - Production', marker_color='pink'))

fig.update_layout(barmode='group', xaxis_title='Model', yaxis_title='Nilai Evaluasi')
st.plotly_chart(fig)

st.markdown("# 7. Kesimpulan")
st.markdown("## 7.1 Kesimpulan Model Tunggal")
st.write("1. Model 1: Linear Regression (Area Harvested -> Production) menggunakan fitur Area Harvested untuk memprediksi variabel target Production. Dalam pengujian dengan data testing sebanyak 1394 baris, model menghasilkan evaluasi yang baik dengan MSE 0.0024, RMSE 0.0493, dan R2 0.7396, menunjukkan kemampuan model dalam menjelaskan sekitar 73.96% variabilitas Production. Dengan total data setelah digabungkan sebanyak 6970 baris, model ini memberikan hasil yang cukup representatif.")
st.write("2. Model 2,4,6 : Linear Regression (Year -> Production), (Year -> Yield), (Year -> Area Harvested) menggunakan fitur Year untuk memprediksi variabel target Production, Yield, dan Area Harvested dengan menggunakan data training sebanyak 48 baris dan data testing sebanyak 13 baris, dapat disimpulkan bahwa Model 4 memiliki performa yang terbaik dengan nilai MSE yang rendah (0.0006), RMSE yang kecil (0.0243), dan R2 yang tinggi (0.9934), menunjukkan bahwa model ini memiliki kemampuan yang baik dalam memprediksi nilai Yield berdasarkan tahun. Model 2 juga memiliki performa yang baik dengan R2 sebesar 0.9488 dalam memprediksi Production berdasarkan tahun, sedangkan Model 6 memiliki performa yang lebih rendah dengan R2 sebesar 0.8620 dalam memprediksi Area Harvested berdasarkan tahun.")
st.write("3. Model 3, 5, 7: Linear Regression (Year {} -> Production), (Year {} -> Yield), (Year {} -> Area Harvested) menggunakan fitur Year {} untuk memprediksi variabel target Production, Yield, dan Area Harvested dengan menggunakan data training sebanyak 48 baris dan data testing sebanyak 13 baris, dapat disimpulkan bahwa setiap negara atau area yang akan diprediksi menghasilkan performa prediksi yang relatif baik atau buruk.".format(input_area_model3, input_area_model5, input_area_model7, input_area_model3))
st.write("4. Model 8,9,10 : ARIMA (Year -> Production), (Year -> Yield), (Year -> Area Harvested) menggunakan fitur Year untuk memprediksi variabel target Production, Yield, dan Area Harvested dengan menggunakan data training sebanyak 48 baris dan data testing sebanyak 13 baris, dapat disimpulkan bahwa Model 9 memiliki performa terbaik dengan MSE yang rendah (0.0008), RMSE yang kecil (0.0285), dan R2 yang tinggi (0.9909), menunjukkan bahwa model ini memiliki kemampuan yang baik dalam memprediksi nilai Yield berdasarkan tahun. Model 8 juga memiliki performa yang baik dengan R2 sebesar 0.9641 dalam memprediksi Production berdasarkan tahun, sedangkan Model 10 memiliki performa yang lebih rendah dengan R2 sebesar 0.8935 dalam memprediksi Area Harvested berdasarkan tahun.")

st.markdown("## 7.2 Kesimpulan Model Ensemble")
st.write("1. untuk prediksi Area Harvested menggunakan teknik ensemble dengan menggunakan metode voting classifier, kita dapat menyimpulkan bahwa model ensemble memiliki performa terbaik dengan MSE yang rendah sebesar 0.0043, RMSE sebesar 0.0658, dan R2 sebesar 0.9604. Hal ini menunjukkan bahwa model ensemble dengan teknik voting classifier mampu memberikan prediksi yang lebih akurat dalam memprediksi Area Harvested berdasarkan tahun, dibandingkan dengan model ARIMA dan Linear Regression yang memiliki nilai evaluasi yang sedikit lebih rendah.")
st.write("2. untuk prediksi Yield menggunakan teknik ensemble dengan menggunakan metode voting classifier, dapat disimpulkan bahwa model ensemble memiliki performa terbaik dengan MSE yang rendah sebesar 0.0004, RMSE sebesar 0.0212, dan R2 sebesar 0.9950. Hal ini menunjukkan bahwa model ensemble dengan teknik voting classifier memberikan prediksi yang paling akurat dalam memprediksi Yield berdasarkan tahun, dengan hasil evaluasi yang lebih baik dibandingkan dengan model ARIMA dan Linear Regression.")
st.write("3. untuk prediksi Production menggunakan teknik ensemble dengan menggunakan metode voting classifier, dapat disimpulkan bahwa model ensemble memiliki performa terbaik dengan MSE yang rendah sebesar 0.0014, RMSE sebesar 0.0380, dan R2 sebesar 0.9864. Hal ini menunjukkan bahwa model ensemble dengan teknik voting classifier memberikan prediksi yang paling akurat dalam memprediksi Production berdasarkan tahun, dengan hasil evaluasi yang lebih baik dibandingkan dengan model ARIMA dan Linear Regression.")
st.write("Berdasarkan hasil analisis dan kesimpulan yang telah dijelaskan pada bagian sebelumnya, maka dapat disimpulkan bahwa model ensemble dengan teknik voting classifier memberikan hasil prediksi yang lebih akurat dibandingkan dengan model ARIMA dan Linear Regression. Oleh karena itu, model ensemble dengan teknik voting classifier dapat digunakan untuk memprediksi nilai Area Harvested, Yield, dan Production berdasarkan tahun.")

st.markdown("# 8. Software Development Goals")
st.write("Berdasarkan hasil analisis dan kaitannya pada kasus kasus prediksi dan analisis produksi dan hasil panen padi di dunia menggunakan teknik ensemble learning, dapat ditarik beberapa kesimpulan yang relevan dengan Sustainable Development Goals (SDGs)")
st.write("Model 1, yaitu Linear Regression (Area Harvested -> Production), memberikan hasil yang cukup baik dalam memprediksi produksi padi berdasarkan luas panen. Hal ini penting dalam mencapai SDGs Goal 2 - Zero Hunger, yang bertujuan untuk mencapai ketahanan pangan yang berkelanjutan dan mengakhiri kelaparan di seluruh dunia.")
st.write("Model 2 dan Model 4, yaitu Linear Regression (Year -> Production) dan Linear Regression (Year -> Yield), memberikan hasil yang baik dalam memprediksi produksi dan hasil panen padi berdasarkan tahun. Ini dapat digunakan dalam merencanakan kebijakan pertanian dan pengelolaan sumber daya alam yang berkelanjutan, sesuai dengan SDGs Goal 12 - Responsible Consumption and Production.")
st.write("Model 8 dan Model 9, yaitu ARIMA (Year -> Production) dan ARIMA (Year -> Yield), memberikan hasil yang baik dalam memprediksi produksi dan hasil panen padi berdasarkan tahun. Hal ini dapat digunakan dalam mengidentifikasi tren produksi dan mengambil langkah-langkah untuk meningkatkan efisiensi produksi dan penggunaan sumber daya alam, yang mendukung SDGs Goal 13 - Climate Action dan Goal 15 - Life on Land.")
st.write("Penggunaan teknik ensemble learning dengan metode voting classifier pada Model Ensemble memberikan hasil prediksi yang lebih akurat dibandingkan dengan model tunggal. Hal ini penting dalam mendukung pengambilan keputusan yang tepat dan efektif dalam perencanaan pertanian berkelanjutan, yang relevan dengan SDGs Goal 9 - Industry, Innovation, and Infrastructure dan Goal 17 - Partnerships for the Goals.")

st.markdown("# 9. Reference/Formula")
st.markdown("## 9.1 MinMaxScaler")
st.latex(r'''
\begin{equation}
x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}
\end{equation}
''')
st.markdown("## 9.2 Linear Regression")
st.latex(r'''
\begin{equation}
\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n
\end{equation}
''')
st.latex(r'''
\begin{equation}
\beta_1 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2}
\end{equation}
''')
st.markdown("## 9.3 ARIMA")
st.latex(r'''
\begin{equation}
ARIMA(p,d,q) = AR(p) + I(d) + MA(q)
\end{equation}
''')
st.latex(r'''
\begin{equation}
AR(p) = \phi_1 y_{t-1} + \phi_2 y_{t-2} + ... + \phi_p y_{t-p}
\end{equation}
''')
st.latex(r'''
\begin{equation}
I(d) = y_t - y_{t-d}
\end{equation}
''')
st.latex(r'''
\begin{equation}
MA(q) = \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + ... + \theta_q \epsilon_{t-q}
\end{equation}
''')
st.latex(r'''
\begin{equation}
\epsilon_t = y_t - \hat{y_t}
\end{equation}
''')
st.latex(r'''
\begin{equation}
\hat{y_t} = \mu + \phi_1 y_{t-1} + \phi_2 y_{t-2} + ... + \phi_p y_{t-p} + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + ... + \theta_q \epsilon_{t-q}
\end{equation}
''')
st.markdown("## 9.4 Voting Classifier")
st.latex(r'''
\begin{equation}
\hat{y} = mode(y_1, y_2, ..., y_n)
\end{equation}
''')
st.latex(r'''
\begin{equation}
mode = \begin{cases}
y_{ARIMA} & \text{jika } |y_{ARIMA} - y_{actual}| < |y_{LR} - y_{actual}| \\
y_{LR} & \text{jika } |y_{ARIMA} - y_{actual}| > |y_{LR} - y_{actual}| \\
\end{cases}
\end{equation}
''')
st.markdown("## 9.5 MSE")
st.latex(r'''
\begin{equation}
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y_i})^2
\end{equation}
''')
st.markdown("## 9.6 RMSE")
st.latex(r'''
\begin{equation}
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y_i})^2}
\end{equation}
''')
st.markdown("## 9.7 R2")
st.latex(r'''
\begin{equation}
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y_i})^2}{\sum_{i=1}^{n} (y_i - \bar{y_i})^2}
\end{equation}
''')



