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
st.write(dataset_after_remove)
st.markdown("## 2. Split Dataset Berdasarkan Element")
dataset_area_harvested = dataset_after_remove[dataset_after_remove['Element'] == 'Area harvested']
dataset_yield = dataset_after_remove[dataset_after_remove['Element'] == 'Yield']
dataset_production = dataset_after_remove[dataset_after_remove['Element'] == 'Production']
st.markdown("### 2.1. Dataset Area Harvested")
st.write(dataset_area_harvested)
st.write("Dataset Area Harvested memiliki {} baris dan {} kolom".format(dataset_area_harvested.shape[0], dataset_area_harvested.shape[1]))
st.write("Dataset Area Harvested memiliki {} data duplikat".format(dataset_area_harvested.duplicated().sum()))
st.write("Dataset Area Harvested memiliki {} data yang hilang".format(dataset_area_harvested.isnull().sum().sum()))
st.write("Area harvested  Ini merujuk pada total luas tanah yang benar-benar dipanen untuk suatu tanaman tertentu, dalam hal ini, padi. Area ini tidak termasuk area yang ditanam tetapi kemudian gagal dipanen karena alasan seperti kekeringan, penyakit, atau banjir. Oleh karena itu, 'area harvested' memberikan ukuran lebih akurat tentang sejauh mana tanaman tersebut berhasil tumbuh dan dipanen.")
st.markdown("### 2.2. Dataset Yield")
st.write(dataset_yield)
st.write("Dataset Yield memiliki {} baris dan {} kolom".format(dataset_yield.shape[0], dataset_yield.shape[1]))
st.write("Dataset Yield memiliki {} data duplikat".format(dataset_yield.duplicated().sum()))
st.write("Dataset Yield memiliki {} data yang hilang".format(dataset_yield.isnull().sum().sum()))
st.write("Yield Ini adalah rasio dari total produksi padi terhadap luas tanah yang dipanen. Ini adalah ukuran efisiensi dalam produksi padi dan biasanya diukur dalam ton per hektare atau kilogram per hektare. Tingkat hasil yang tinggi biasanya mengindikasikan teknologi dan praktik pertanian yang efisien dan efektif.")
st.markdown("### 2.3. Dataset Production")
st.write(dataset_production)
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
st.write(marge_describe_value)


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
input_area = st.selectbox('Pilih Area', combined_dataset['Area'].unique())
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
st.write(dataset)

st.markdown("## 4.2. Normalisasi Data")
scaler_value_area = MinMaxScaler()
scaler_value_yield = MinMaxScaler()
scaler_value_production = MinMaxScaler()
scaler_year = MinMaxScaler()

dataset['Value Area Harvested (ha)'] = scaler_value_area.fit_transform(dataset[['Value Area Harvested (ha)']])
dataset['Value Yield (hg/ha)'] = scaler_value_yield.fit_transform(dataset[['Value Yield (hg/ha)']])
dataset['Value Production (ton)'] = scaler_value_production.fit_transform(dataset[['Value Production (ton)']])
dataset['Year'] = scaler_year.fit_transform(dataset[['Year']])
st.write(dataset)

st.markdown("## 4.3. Sum Berdasarkan Tahun (1961-2021)")
st.markdown("### 4.3.1. Original Dataset")
dataset_year = combined_dataset.copy()
dataset_year = dataset_year.groupby(['Year']).sum()
dataset_year = dataset_year.reset_index()
dataset_year = dataset_year.drop(['Area'], axis=1)
st.write(dataset_year)

st.markdown("### 4.3.2. Setelah Normalisasi")
scaler_value_area_year = MinMaxScaler()
scaler_value_yield_year = MinMaxScaler()
scaler_value_production_year = MinMaxScaler()
scaler_year_year = MinMaxScaler()

dataset_year['Value Area Harvested (ha)'] = scaler_value_area_year.fit_transform(dataset_year[['Value Area Harvested (ha)']])
dataset_year['Value Yield (hg/ha)'] = scaler_value_yield_year.fit_transform(dataset_year[['Value Yield (hg/ha)']])
dataset_year['Value Production (ton)'] = scaler_value_production_year.fit_transform(dataset_year[['Value Production (ton)']])
dataset_year['Year'] = scaler_year_year.fit_transform(dataset_year[['Year']])
st.write(dataset_year)

st.markdown("# 5. Modelling")

st.markdown("## 5.1. Linear Regression")
st.markdown("### 5.1.1. Model 1: Menggunakan Area Harvested untuk memprediksi Production")
X1 = dataset[['Value Area Harvested (ha)']]
y1 = dataset[['Value Production (ton)']]

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=0)

model1 = LinearRegression()
model1.fit(X1_train, y1_train)
y1_pred = model1.predict(X1_test)

mse = mean_squared_error(y1_test, y1_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y1_test, y1_pred)

st.write("MSE : {}".format(mse))
st.write("RMSE : {}".format(rmse))
st.write("R2 : {}".format(r2))

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

model2 = LinearRegression()
model2.fit(X2_train, y2_train)
y2_pred = model2.predict(X2_test)

mse = mean_squared_error(y2_test, y2_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y2_test, y2_pred)

st.write("MSE : {}".format(mse))
st.write("RMSE : {}".format(rmse))
st.write("R2 : {}".format(r2))

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


st.markdown("### 5.1.3. Model 4: Menggunakan Year Dengan Filter Negara untuk memprediksi Production")

unique_area = dataset['Area'].unique()
unique_area_inverse = encoder.inverse_transform(unique_area.reshape(-1, 1))
unique_area_inverse = unique_area_inverse.flatten()
unique_area_inverse = np.sort(unique_area_inverse)
unique_area_inverse = unique_area_inverse.tolist()

input_area_model4 = st.selectbox('Pilih Area', unique_area_inverse)

index_input_area_model4 = unique_area_inverse.index(input_area_model4)

dataset_area = dataset[dataset['Area'] == index_input_area_model4]


X3 = dataset_area[['Year']]
y3 = dataset_area[['Value Production (ton)']]

X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=0)

model3 = LinearRegression()
model3.fit(X3_train, y3_train)
y3_pred = model3.predict(X3_test)

mse = mean_squared_error(y3_test, y3_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y3_test, y3_pred)

st.write("MSE : {}".format(mse))
st.write("RMSE : {}".format(rmse))
st.write("R2 : {}".format(r2))

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

st.markdown("### 5.1.4. Model 5: Menggunakan Year untuk memprediksi Yield")

X4 = dataset_year[['Year']]
y4 = dataset_year[['Value Yield (hg/ha)']]

X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.2, random_state=0)

model4 = LinearRegression()
model4.fit(X4_train, y4_train)
y4_pred = model4.predict(X4_test)

mse = mean_squared_error(y4_test, y4_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y4_test, y4_pred)

st.write("MSE : {}".format(mse))
st.write("RMSE : {}".format(rmse))
st.write("R2 : {}".format(r2))

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

st.markdown("### 5.1.5. Model 6: Menggunakan Year Dengan Filter Negara untuk memprediksi Yield")

input_area_model6 = st.selectbox('Pilih Area', unique_area_inverse, key='input_area_model6')

index_input_area_model6 = unique_area_inverse.index(input_area_model6)

dataset_area = dataset[dataset['Area'] == index_input_area_model6]

X5 = dataset_area[['Year']]
y5 = dataset_area[['Value Yield (hg/ha)']]

X5_train, X5_test, y5_train, y5_test = train_test_split(X5, y5, test_size=0.2, random_state=0)

model5 = LinearRegression()
model5.fit(X5_train, y5_train)
y5_pred = model5.predict(X5_test)

mse = mean_squared_error(y5_test, y5_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y5_test, y5_pred)

st.write("MSE : {}".format(mse))
st.write("RMSE : {}".format(rmse))
st.write("R2 : {}".format(r2))

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

st.markdown("### 5.1.6. Model 7: Menggunakan Year untuk memprediksi Area Harvested")

X6 = dataset_year[['Year']]
y6 = dataset_year[['Value Area Harvested (ha)']]


X6_train, X6_test, y6_train, y6_test = train_test_split(X6, y6, test_size=0.2, random_state=0)

model6 = LinearRegression()
model6.fit(X6_train, y6_train)
y6_pred = model6.predict(X6_test)

mse = mean_squared_error(y6_test, y6_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y6_test, y6_pred)

st.write("MSE : {}".format(mse))
st.write("RMSE : {}".format(rmse))
st.write("R2 : {}".format(r2))

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

st.markdown("### 5.1.7. Model 8: Menggunakan Year Dengan Filter Negara untuk memprediksi Area Harvested")

input_area_model8 = st.selectbox('Pilih Area', unique_area_inverse, key='input_area_model8')

index_input_area_model8 = unique_area_inverse.index(input_area_model8)

dataset_area = dataset[dataset['Area'] == index_input_area_model8]

X7 = dataset_area[['Year']]
y7 = dataset_area[['Value Area Harvested (ha)']]

X7_train, X7_test, y7_train, y7_test = train_test_split(X7, y7, test_size=0.2, random_state=0)

model7 = LinearRegression()
model7.fit(X7_train, y7_train)
y7_pred = model7.predict(X7_test)

mse = mean_squared_error(y7_test, y7_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y7_test, y7_pred)

st.write("MSE : {}".format(mse))
st.write("RMSE : {}".format(rmse))
st.write("R2 : {}".format(r2))

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
st.markdown("### 5.2.1. Model 9: Menggunakan Year untuk memprediksi Value Area Harvested (ha)")

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

st.write("MSE : {}".format(model_arima_1_fit.mse))
st.write("RMSE : {}".format(np.sqrt(model_arima_1_fit.mse)))
st.write("R2 : {}".format(r2_score(dataset_forecast_value_area['Value Area Harvested (ha)'], model_arima_1_fit.fittedvalues)))

# Plot Residuals Errors
residuals = pd.DataFrame(model_arima_1_fit.resid)
fig = px.line(x=year, y=residuals[0], title='Residuals Errors')
fig.update_layout(xaxis_title='Lags', yaxis_title='Residuals Errors')
st.plotly_chart(fig)

# Lakukan Forecasting
input_forecast_arima_1 = st.number_input('Input Forecasting', min_value=1, value=1, step=1, key='input_forecast_arima_1')
forecast = model_arima_1_fit.forecast(steps=input_forecast_arima_1)
forecast = scaler_value_area.inverse_transform(np.array(forecast).reshape(-1, 1))
forecast = forecast.flatten().tolist()
forecast = [round(i, 2) for i in forecast]
forecast = [0 if i < 0 else i for i in forecast]
forecast = [int(i) for i in forecast]

# Plot Forecasting
y_pred_arima_1 = scaler_value_area.inverse_transform(np.array(y_pred_arima_1).reshape(-1, 1))

data = pd.DataFrame(data=[data_value_area, y_pred_arima_1.flatten().tolist(), forecast], index=['Actual', 'Predict', 'Forecast'], columns=year).T

fig = go.Figure()
fig.add_trace(go.Scatter(x=year, y=data['Actual'], mode='lines+markers', name='Actual', line=dict(color='gainsboro', width=2), marker=dict(color='gainsboro', symbol='circle', size=5)))
fig.add_trace(go.Scatter(x=year, y=data['Predict'], mode='lines+markers', name='Predict', line=dict(color='yellow', width=2)))
fig.add_trace(go.Scatter(x=[year[-1] + i for i in range(1, input_forecast_arima_1+1)], y=data['Forecast'], mode='lines+markers', name='Forecast', line=dict(color='red', width=2)))

fig.update_layout(xaxis_title='Year', yaxis_title='Value Area Harvested (ha)')
st.plotly_chart(fig)

st.markdown("### 5.2.2. Model 10: Menggunakan Year untuk memprediksi Value Yield (hg/ha)")

dataset_forecast_value_yield = pd.DataFrame(data=[data_value_yield], index=['Value Yield (hg/ha)'], columns=year).T
scaler_value_area = MinMaxScaler(feature_range=(0, 1))
dataset_forecast_value_yield['Value Yield (hg/ha)'] = scaler_value_area.fit_transform(dataset_forecast_value_yield['Value Yield (hg/ha)'].values.reshape(-1, 1))

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

st.write("MSE : {}".format(model_arima_2_fit.mse))
st.write("RMSE : {}".format(np.sqrt(model_arima_2_fit.mse)))
st.write("R2 : {}".format(r2_score(dataset_forecast_value_yield['Value Yield (hg/ha)'], model_arima_2_fit.fittedvalues)))

# Plot Residuals Errors
residuals = pd.DataFrame(model_arima_2_fit.resid)
fig = px.line(x=year, y=residuals[0], title='Residuals Errors')
fig.update_layout(xaxis_title='Lags', yaxis_title='Residuals Errors')
st.plotly_chart(fig)

# Lakukan Forecasting
input_forecast_arima_2 = st.number_input('Input Forecasting', min_value=1, value=1, step=1, key='input_forecast_arima_2')
forecast = model_arima_2_fit.forecast(steps=input_forecast_arima_2)
forecast = scaler_value_area.inverse_transform(np.array(forecast).reshape(-1, 1))
forecast = forecast.flatten().tolist()
forecast = [round(i, 2) for i in forecast]
forecast = [0 if i < 0 else i for i in forecast]
forecast = [int(i) for i in forecast]


# Plot Forecasting
y_pred_arima_2 = scaler_value_area.inverse_transform(np.array(y_pred_arima_2).reshape(-1, 1))

data = pd.DataFrame(data=[data_value_yield, y_pred_arima_2.flatten().tolist(), forecast], index=['Actual', 'Predict', 'Forecast'], columns=year).T

fig = go.Figure()
fig.add_trace(go.Scatter(x=year, y=data['Actual'], mode='lines+markers', name='Actual', line=dict(color='gainsboro', width=2), marker=dict(color='gainsboro', symbol='circle', size=5)))
fig.add_trace(go.Scatter(x=year, y=data['Predict'], mode='lines+markers', name='Predict', line=dict(color='yellow', width=2)))
fig.add_trace(go.Scatter(x=[year[-1] + i for i in range(1, input_forecast_arima_2+1)], y=data['Forecast'], mode='lines+markers', name='Forecast', line=dict(color='red', width=2)))

fig.update_layout(xaxis_title='Year', yaxis_title='Value Yield (hg/ha)')
st.plotly_chart(fig)

st.markdown("### 5.2.3. Model 11: Menggunakan Year untuk memprediksi Value Production (ton)")

dataset_forecast_value_production = pd.DataFrame(data=[data_value_production], index=['Value Production (ton)'], columns=year).T
scaler_value_production = MinMaxScaler(feature_range=(0, 1))
dataset_forecast_value_production['Value Production (ton)'] = scaler_value_production.fit_transform(dataset_forecast_value_production['Value Production (ton)'].values.reshape(-1, 1))

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

st.write("MSE : {}".format(model_arima_3_fit.mse))
st.write("RMSE : {}".format(np.sqrt(model_arima_3_fit.mse)))
st.write("R2 : {}".format(r2_score(dataset_forecast_value_production['Value Production (ton)'], model_arima_3_fit.fittedvalues)))

# Plot Residuals Errors
residuals = pd.DataFrame(model_arima_3_fit.resid)
fig = px.line(x=year, y=residuals[0], title='Residuals Errors')
fig.update_layout(xaxis_title='Lags', yaxis_title='Residuals Errors')
st.plotly_chart(fig)

# Lakukan Forecasting
input_forecast_arima_3 = st.number_input('Input Forecasting', min_value=1, value=1, step=1, key='input_forecast_arima_3')
forecast = model_arima_3_fit.forecast(steps=input_forecast_arima_3)
forecast = scaler_value_production.inverse_transform(np.array(forecast).reshape(-1, 1))
forecast = forecast.flatten().tolist()
forecast = [round(i, 2) for i in forecast]
forecast = [0 if i < 0 else i for i in forecast]
forecast = [int(i) for i in forecast]

# Plot Forecasting
y_pred_arima_3 = scaler_value_production.inverse_transform(np.array(y_pred_arima_3).reshape(-1, 1))

data = pd.DataFrame(data=[data_value_production, y_pred_arima_3.flatten().tolist(), forecast], index=['Actual', 'Predict', 'Forecast'], columns=year).T

fig = go.Figure()
fig.add_trace(go.Scatter(x=year, y=data['Actual'], mode='lines+markers', name='Actual', line=dict(color='gainsboro', width=2), marker=dict(color='gainsboro', symbol='circle', size=5)))
fig.add_trace(go.Scatter(x=year, y=data['Predict'], mode='lines+markers', name='Predict', line=dict(color='yellow', width=2)))
fig.add_trace(go.Scatter(x=[year[-1] + i for i in range(1, input_forecast_arima_3+1)], y=data['Forecast'], mode='lines+markers', name='Forecast', line=dict(color='red', width=2)))

fig.update_layout(xaxis_title='Year', yaxis_title='Value Production (ton)')
st.plotly_chart(fig)

st.markdown("### 5.3. Ensemble Model")
st.markdown("### 5.3.1. Model 12: Menggunakan Year untuk memprediksi Area Harvested (ha)")

year_selected = X6_test['Year'].values.tolist()
year_selected = [round(i) for i in year_selected]
year_selected = [int(i) for i in year_selected]

y_pred_arima_1 = pd.DataFrame(y_pred_arima_1)
y_pred_arima_1.index = year
y_pred_arima_1 = y_pred_arima_1.loc[year_selected]

y6_pred = pd.DataFrame(y6_pred)
y6_pred.index = year_selected


dataset_ensemble_area_harvested = pd.DataFrame()
dataset_ensemble_area_harvested['Year'] = year_selected
dataset_ensemble_area_harvested['Actual'] = y6_test
dataset_ensemble_area_harvested['ARIMA'] = y_pred_arima_1.values
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

scaler = MinMaxScaler()
dataset_ensemble_area_harvested['Actual'] = scaler.fit_transform(np.array(dataset_ensemble_area_harvested['Actual']).reshape(-1, 1))
dataset_ensemble_area_harvested['ARIMA'] = scaler.fit_transform(np.array(dataset_ensemble_area_harvested['ARIMA']).reshape(-1, 1))
dataset_ensemble_area_harvested['Linear Regression'] = scaler.fit_transform(np.array(dataset_ensemble_area_harvested['Linear Regression']).reshape(-1, 1))
dataset_ensemble_area_harvested['ARIMA DIFF'] = scaler.fit_transform(np.array(dataset_ensemble_area_harvested['ARIMA DIFF']).reshape(-1, 1))
dataset_ensemble_area_harvested['LR DIFF'] = scaler.fit_transform(np.array(dataset_ensemble_area_harvested['LR DIFF']).reshape(-1, 1))
dataset_ensemble_area_harvested['Ensemble Value'] = scaler.fit_transform(np.array(dataset_ensemble_area_harvested['Ensemble Value']).reshape(-1, 1))


# Evaluasi Model dengan MSE, RMSE, dan R2 Untuk ARIMA
mse_area_harvested_arima = mean_squared_error(dataset_ensemble_area_harvested['Actual'], dataset_ensemble_area_harvested['ARIMA'])
rmse_area_harvested_arima = np.sqrt(mean_squared_error(dataset_ensemble_area_harvested['Actual'], dataset_ensemble_area_harvested['ARIMA']))
r2_area_harvested_arima = r2_score(dataset_ensemble_area_harvested['Actual'], dataset_ensemble_area_harvested['ARIMA'])

# Evaluasi Model dengan MSE, RMSE, dan R2 Untuk Linear Regression
mse_area_harvested_lr = mean_squared_error(dataset_ensemble_area_harvested['Actual'], dataset_ensemble_area_harvested['Linear Regression'])
rmse_area_harvested_lr = np.sqrt(mean_squared_error(dataset_ensemble_area_harvested['Actual'], dataset_ensemble_area_harvested['Linear Regression']))
r2_area_harvested_lr = r2_score(dataset_ensemble_area_harvested['Actual'], dataset_ensemble_area_harvested['Linear Regression'])

# Evaluasi Model dengan MSE, RMSE, dan R2 Ensemble
mse_area_harvested_ensemble = mean_squared_error(dataset_ensemble_area_harvested['Actual'], dataset_ensemble_area_harvested['Ensemble Value'])
rmse_area_harvested_ensemble = np.sqrt(mean_squared_error(dataset_ensemble_area_harvested['Actual'], dataset_ensemble_area_harvested['Ensemble Value']))
r2_area_harvested_ensemble = r2_score(dataset_ensemble_area_harvested['Actual'], dataset_ensemble_area_harvested['Ensemble Value'])

# Buatkan DataFrame untuk Evaluasi Model
data_eval_area_harvested = pd.DataFrame(data=[['ARIMA', mse_area_harvested_arima, rmse_area_harvested_arima, r2_area_harvested_arima], ['Linear Regression', mse_area_harvested_lr, rmse_area_harvested_lr, r2_area_harvested_lr], ['Ensemble', mse_area_harvested_ensemble, rmse_area_harvested_ensemble, r2_area_harvested_ensemble]], columns=['Model', 'MSE', 'RMSE', 'R2'])
data_eval_area_harvested = data_eval_area_harvested.set_index('Model')
st.write(data_eval_area_harvested)

st.markdown("### 5.3.2. Model 13: Menggunakan Year untuk memprediksi Yield (hg/ha)")

year_selected = X4_test['Year'].values.tolist()
year_selected = [round(i) for i in year_selected]
year_selected = [int(i) for i in year_selected]

y_pred_arima_2 = pd.DataFrame(y_pred_arima_2)
y_pred_arima_2.index = year
y_pred_arima_2 = y_pred_arima_2.loc[year_selected]

y4_pred = pd.DataFrame(y4_pred)
y4_pred.index = year_selected


dataset_ensemble_yield = pd.DataFrame()
dataset_ensemble_yield['Year'] = year_selected
dataset_ensemble_yield['Actual'] = y4_test
dataset_ensemble_yield['ARIMA'] = y_pred_arima_2.values
dataset_ensemble_yield['Linear Regression'] = y4_pred.values
dataset_ensemble_yield['ARIMA DIFF'] = abs(dataset_ensemble_yield['Actual'] - dataset_ensemble_yield['ARIMA'])
dataset_ensemble_yield['LR DIFF'] = abs(dataset_ensemble_yield['Actual'] - dataset_ensemble_yield['Linear Regression'])
dataset_ensemble_yield["Ensemble"] = dataset_ensemble_yield.loc[:, ["ARIMA DIFF", "LR DIFF"]].idxmin(axis=1).map(lambda x: 'ARIMA' if x == 'ARIMA DIFF' else 'Linear Regression')
dataset_ensemble_yield["Ensemble Value"] = dataset_ensemble_yield.apply(lambda row: row['ARIMA'] if row['Ensemble'] == 'ARIMA' else row['Linear Regression'], axis=1)

st.write(dataset_ensemble_yield)


dataset_ensemble_yield = dataset_ensemble_yield.sort_values(by=['Year'])

# Buat Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=dataset_ensemble_yield['Year'], y=dataset_ensemble_yield['Actual'], mode='markers', name='Actual', line=dict(color='gainsboro', width=2), marker=dict(color='gainsboro', symbol='circle', size=5)))
fig.add_trace(go.Scatter(x=dataset_ensemble_yield['Year'], y=dataset_ensemble_yield['ARIMA'], mode='markers', name='ARIMA', line=dict(color='yellow', width=2)))
fig.add_trace(go.Scatter(x=dataset_ensemble_yield['Year'], y=dataset_ensemble_yield['Linear Regression'], mode='markers', name='Linear Regression', line=dict(color='red', width=2)))
fig.add_trace(go.Scatter(x=dataset_ensemble_yield['Year'], y=dataset_ensemble_yield['Ensemble Value'], mode='lines+markers', name='Ensemble', line=dict(color='blue', width=2)))

fig.update_layout(xaxis_title='Year', yaxis_title='Yield (hg/ha)')
st.plotly_chart(fig)

scaler = MinMaxScaler()
dataset_ensemble_yield['Actual'] = scaler.fit_transform(np.array(dataset_ensemble_yield['Actual']).reshape(-1, 1))
dataset_ensemble_yield['ARIMA'] = scaler.fit_transform(np.array(dataset_ensemble_yield['ARIMA']).reshape(-1, 1))
dataset_ensemble_yield['Linear Regression'] = scaler.fit_transform(np.array(dataset_ensemble_yield['Linear Regression']).reshape(-1, 1))
dataset_ensemble_yield['ARIMA DIFF'] = scaler.fit_transform(np.array(dataset_ensemble_yield['ARIMA DIFF']).reshape(-1, 1))
dataset_ensemble_yield['LR DIFF'] = scaler.fit_transform(np.array(dataset_ensemble_yield['LR DIFF']).reshape(-1, 1))
dataset_ensemble_yield['Ensemble Value'] = scaler.fit_transform(np.array(dataset_ensemble_yield['Ensemble Value']).reshape(-1, 1))


# Evaluasi Model dengan MSE, RMSE, dan R2 Untuk ARIMA
mse_yield_arima = mean_squared_error(dataset_ensemble_yield['Actual'], dataset_ensemble_yield['ARIMA'])
rmse_yield_arima = np.sqrt(mean_squared_error(dataset_ensemble_yield['Actual'], dataset_ensemble_yield['ARIMA']))
r2_yield_arima = r2_score(dataset_ensemble_yield['Actual'], dataset_ensemble_yield['ARIMA'])

# Evaluasi Model dengan MSE, RMSE, dan R2 Untuk Linear Regression
mse_yield_lr = mean_squared_error(dataset_ensemble_yield['Actual'], dataset_ensemble_yield['Linear Regression'])
rmse_yield_lr = np.sqrt(mean_squared_error(dataset_ensemble_yield['Actual'], dataset_ensemble_yield['Linear Regression']))
r2_yield_lr = r2_score(dataset_ensemble_yield['Actual'], dataset_ensemble_yield['Linear Regression'])

# Evaluasi Model dengan MSE, RMSE, dan R2 Ensemble
mse_yield_ensemble = mean_squared_error(dataset_ensemble_yield['Actual'], dataset_ensemble_yield['Ensemble Value'])
rmse_yield_ensemble = np.sqrt(mean_squared_error(dataset_ensemble_yield['Actual'], dataset_ensemble_yield['Ensemble Value']))
r2_yield_ensemble = r2_score(dataset_ensemble_yield['Actual'], dataset_ensemble_yield['Ensemble Value'])

# Buatkan DataFrame untuk Evaluasi Model
data_eval_yield = pd.DataFrame(data=[['ARIMA', mse_yield_arima, rmse_yield_arima, r2_yield_arima], ['Linear Regression', mse_yield_lr, rmse_yield_lr, r2_yield_lr], ['Ensemble', mse_yield_ensemble, rmse_yield_ensemble, r2_yield_ensemble]], columns=['Model', 'MSE', 'RMSE', 'R2'])
data_eval_yield = data_eval_yield.set_index('Model')
st.write(data_eval_yield)

st.markdown("### 5.3.3. Model 14: Menggunakan Year untuk memprediksi Production (ton)")

year_selected = X2_test['Year'].values.tolist()
year_selected = [round(i) for i in year_selected]
year_selected = [int(i) for i in year_selected]

y_pred_arima_3 = pd.DataFrame(y_pred_arima_3)
y_pred_arima_3.index = year
y_pred_arima_3 = y_pred_arima_3.loc[year_selected]

y2_pred = pd.DataFrame(y2_pred)
y2_pred.index = year_selected


dataset_ensemble_production = pd.DataFrame()
dataset_ensemble_production['Year'] = year_selected
dataset_ensemble_production['Actual'] = y2_test
dataset_ensemble_production['ARIMA'] = y_pred_arima_3.values
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

scaler = MinMaxScaler()
dataset_ensemble_production['Actual'] = scaler.fit_transform(np.array(dataset_ensemble_production['Actual']).reshape(-1, 1))
dataset_ensemble_production['ARIMA'] = scaler.fit_transform(np.array(dataset_ensemble_production['ARIMA']).reshape(-1, 1))
dataset_ensemble_production['Linear Regression'] = scaler.fit_transform(np.array(dataset_ensemble_production['Linear Regression']).reshape(-1, 1))
dataset_ensemble_production['ARIMA DIFF'] = scaler.fit_transform(np.array(dataset_ensemble_production['ARIMA DIFF']).reshape(-1, 1))
dataset_ensemble_production['LR DIFF'] = scaler.fit_transform(np.array(dataset_ensemble_production['LR DIFF']).reshape(-1, 1))
dataset_ensemble_production['Ensemble Value'] = scaler.fit_transform(np.array(dataset_ensemble_production['Ensemble Value']).reshape(-1, 1))


# Evaluasi Model dengan MSE, RMSE, dan R2 Untuk ARIMA
mse_production_arima = mean_squared_error(dataset_ensemble_production['Actual'], dataset_ensemble_production['ARIMA'])
rmse_production_arima = np.sqrt(mean_squared_error(dataset_ensemble_production['Actual'], dataset_ensemble_production['ARIMA']))
r2_production_arima = r2_score(dataset_ensemble_production['Actual'], dataset_ensemble_production['ARIMA'])

# Evaluasi Model dengan MSE, RMSE, dan R2 Untuk Linear Regression
mse_production_lr = mean_squared_error(dataset_ensemble_production['Actual'], dataset_ensemble_production['Linear Regression'])
rmse_production_lr = np.sqrt(mean_squared_error(dataset_ensemble_production['Actual'], dataset_ensemble_production['Linear Regression']))
r2_production_lr = r2_score(dataset_ensemble_production['Actual'], dataset_ensemble_production['Linear Regression'])

# Evaluasi Model dengan MSE, RMSE, dan R2 Ensemble
mse_production_ensemble = mean_squared_error(dataset_ensemble_production['Actual'], dataset_ensemble_production['Ensemble Value'])
rmse_production_ensemble = np.sqrt(mean_squared_error(dataset_ensemble_production['Actual'], dataset_ensemble_production['Ensemble Value']))
r2_production_ensemble = r2_score(dataset_ensemble_production['Actual'], dataset_ensemble_production['Ensemble Value'])

# Buatkan DataFrame untuk Evaluasi Model
data_eval_production = pd.DataFrame(data=[['ARIMA', mse_production_arima, rmse_production_arima, r2_production_arima], ['Linear Regression', mse_production_lr, rmse_production_lr, r2_production_lr], ['Ensemble', mse_production_ensemble, rmse_production_ensemble, r2_production_ensemble]], columns=['Model', 'MSE', 'RMSE', 'R2'])
data_eval_production = data_eval_production.set_index('Model')
st.write(data_eval_production)

st.write("# 6. Evaluasi Model")


