import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="UCPP",
    page_icon="🚗",
)


st.markdown(
    "<h1 style='text-align: center;'>UCPP : Sistem Prediksi Estimasi Harga Mobil Bekas Dengan Menggunakan Algoritma Random Forest</h1>",
    unsafe_allow_html=True
)

st.header("Data Mobil Bekas")


data = pd.read_csv("data.csv")

st.dataframe(data)

st.write("[Sumber Data](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes)")
st.write("Data diambil dari [Kaggle](https://kaggle.com)")

st.header("Features")

col1, col2, col3 = st.columns(3)

col1.metric(label="Entries", value=len(data))
col2.metric(label="Features", value=len(data.columns))
col3.metric(label="(Label) Clusters", value=1)

st.header("Informasi tentang Variabel")

data_labels = {
    "Variable": ["brand", "model", "transmission", "year", "fuelType", "mileage", "engineSize", "price"],
    "Data Type": ["categorical", "categorical", "categorical", "numerical", "categorical", "numerical", "numerical", "numerical"],
    "Description": [
        "Menunjukkan merek atau pabrikan kendaraan, yang sering kali terkait dengan reputasi, keandalan, dan preferensi konsumen",
        "Menunjukkan model spesifik dari kendaraan, seperti seri atau tipe yang diproduksi oleh pabrikan.",
        "Menunjukkan jenis transmisi kendaraan, seperti manual atau otomatis, yang mempengaruhi pengalaman berkendara dan konsumsi bahan bakar.",
        "Menunjukkan tahun pembuatan kendaraan, yang membantu dalam menentukan usia dan potensi nilai depresiasi kendaraan.",
        "Menunjukkan jenis bahan bakar yang digunakan oleh kendaraan, seperti bensin, diesel, listrik, atau hybrid, yang mempengaruhi efisiensi bahan bakar dan biaya operasional.",
        "Menunjukkan jumlah total jarak yang telah ditempuh kendaraan dalam mil, yang merupakan indikator utama dari penggunaan dan keausan kendaraan.",
        "Menunjukkan ukuran mesin kendaraan dalam liter, yang dapat mempengaruhi kinerja, efisiensi bahan bakar, dan pajak kendaraan.",
        "Menunjukkan harga jual kendaraan dalam mata uang yang berlaku, yang merupakan target atau label dalam banyak analisis harga kendaraan."
    ]
}

data_labels = pd.DataFrame.from_dict(data_labels)
st.dataframe(data_labels)

# Menghapus kolom 'ID' jika ada (dalam kasus ini tidak ada, jadi dihapus dari kode)
# if 'ID' in data.columns:
#     data.drop(columns='ID', inplace=True)

# ------------- Describing and getting information --- #
st.subheader("Deskripsi Data - Numerik")

numerical_data = data.select_dtypes(include=['number'])
st.dataframe(numerical_data.describe().T)

st.subheader("Deskripsi Data - Kategorikal")

categorical_data = data.select_dtypes(include=['object'])
st.dataframe(categorical_data.describe().T)

import streamlit as st
import pandas as pd

# Defining the brand_model_map dictionary
brand_model_map = {
    'audi': ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'Q2', 'Q3', 'Q5', 'Q7', 'Q8', 'RS3', 'RS4', 'RS5', 'RS6', 'RS7', 'S3', 'S4', 'S5', 'S8', 'SQ5', 'SQ7', 'TT', 'R8'],
    'bmw': ['1 Series', '2 Series', '3 Series', '4 Series', '5 Series', '6 Series', '7 Series', '8 Series', 'M2', 'M3', 'M4', 'M5', 'M6', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'Z3', 'Z4'],
    'ford': ['B-MAX', 'C-MAX', 'EcoSport', 'Edge', 'Fiesta', 'Focus', 'Galaxy', 'Grand C-MAX', 'Grand Tourneo Connect', 'KA', 'Ka+', 'Kuga', 'Mondeo', 'Mustang', 'Puma', 'Ranger', 'S-MAX', 'Tourneo Connect', 'Tourneo Custom', 'Transit Tourneo'],
    'hyundi': ['Accent', 'Amica', 'Coupe', 'Elantra', 'Getz', 'I10', 'I20', 'I30', 'I40', 'IX20', 'IX35', 'Kona', 'Santa Fe', 'Tucson', 'Veloster'],
    'merc': ['A Class', 'B Class', 'C Class', 'CL Class', 'CLA Class', 'CLC Class', 'CLK', 'CLS Class', 'E Class', 'G Class', 'GL Class', 'GLA Class', 'GLB Class', 'GLC Class', 'GLE Class', 'GLS Class', 'M Class', 'R Class', 'S Class', 'SL CLASS', 'SLK', 'V Class', 'X-CLASS'],
    'skoda': ['Citigo', 'Fabia', 'Kamiq', 'Karoq', 'Kodiaq', 'Octavia', 'Rapid', 'Roomster', 'Scala', 'Superb', 'Yeti', 'Yeti Outdoor'],
    'toyota': ['Auris', 'Avensis', 'Aygo', 'Camry', 'Corolla', 'GT86', 'Hilux', 'IQ', 'Land Cruiser', 'Prius', 'RAV4', 'Supra', 'Urban Cruiser', 'Verso', 'Verso-S', 'Yaris'],
    'vauxhall': ['Adam', 'Agila', 'Ampera', 'Antara', 'Astra', 'Cascada', 'Combo Life', 'Corsa', 'Crossland X', 'GTC', 'Insignia', 'Meriva', 'Mokka', 'Mokka X', 'Viva', 'Vivaro', 'Zafira', 'Zafira Tourer'],
    'vw': ['Amarok', 'Arteon', 'Beetle', 'Caddy', 'Caddy Life', 'Caddy Maxi', 'Caddy Maxi Life', 'California', 'Caravelle', 'CC', 'Eos', 'Fox', 'Golf', 'Golf SV', 'Jetta', 'Passat', 'Polo', 'Scirocco', 'Sharan', 'T-Cross', 'T-Roc', 'Tiguan', 'Tiguan Allspace', 'Touareg', 'Touran', 'Up']
}
# Streamlit app
st.title('Model')
# Convert the dictionary into a DataFrame
df = pd.DataFrame([(brand, model) for brand, models in brand_model_map.items() for model in models],
                  columns=['Brand', 'Model'])

# Displaying the table using Streamlit
st.write(df)



# Transmission mappings
transmissions = {
    0: 'Automatic',
    1: 'Manual',
    2: 'Other',
    3: 'Semi-Auto'
}

# Convert transmissions to a DataFrame
transmissions_df = pd.DataFrame(list(transmissions.items()), columns=['Code', 'Transmission'])

# Streamlit app
st.title('Transmission')

st.dataframe(transmissions_df)

import streamlit as st
import pandas as pd

# Brand mappings
brand_model_map = {
    'audi': ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'Q2', 'Q3', 'Q5', 'Q7', 'Q8', 'RS3', 'RS4', 'RS5', 'RS6', 'RS7', 'S3', 'S4', 'S5', 'S8', 'SQ5', 'SQ7', 'TT', 'R8'],
    'bmw': ['1 Series', '2 Series', '3 Series', '4 Series', '5 Series', '6 Series', '7 Series', '8 Series', 'M2', 'M3', 'M4', 'M5', 'M6', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'Z3', 'Z4'],
    'ford': ['B-MAX', 'C-MAX', 'EcoSport', 'Edge', 'Fiesta', 'Focus', 'Galaxy', 'Grand C-MAX', 'Grand Tourneo Connect', 'KA', 'Ka+', 'Kuga', 'Mondeo', 'Mustang', 'Puma', 'Ranger', 'S-MAX', 'Tourneo Connect', 'Tourneo Custom', 'Transit Tourneo'],
    'hyundi': ['Accent', 'Amica', 'Coupe', 'Elantra', 'Getz', 'I10', 'I20', 'I30', 'I40', 'IX20', 'IX35', 'Kona', 'Santa Fe', 'Tucson', 'Veloster'],
    'merc': ['A Class', 'B Class', 'C Class', 'CL Class', 'CLA Class', 'CLC Class', 'CLK', 'CLS Class', 'E Class', 'G Class', 'GL Class', 'GLA Class', 'GLB Class', 'GLC Class', 'GLE Class', 'GLS Class', 'M Class', 'R Class', 'S Class', 'SL CLASS', 'SLK', 'V Class', 'X-CLASS'],
    'skoda': ['Citigo', 'Fabia', 'Kamiq', 'Karoq', 'Kodiaq', 'Octavia', 'Rapid', 'Roomster', 'Scala', 'Superb', 'Yeti', 'Yeti Outdoor'],
    'toyota': ['Auris', 'Avensis', 'Aygo', 'Camry', 'Corolla', 'GT86', 'Hilux', 'IQ', 'Land Cruiser', 'Prius', 'RAV4', 'Supra', 'Urban Cruiser', 'Verso', 'Verso-S', 'Yaris'],
    'vauxhall': ['Adam', 'Agila', 'Ampera', 'Antara', 'Astra', 'Cascada', 'Combo Life', 'Corsa', 'Crossland X', 'GTC', 'Insignia', 'Meriva', 'Mokka', 'Mokka X', 'Viva', 'Vivaro', 'Zafira', 'Zafira Tourer'],
    'vw': ['Amarok', 'Arteon', 'Beetle', 'Caddy', 'Caddy Life', 'Caddy Maxi', 'Caddy Maxi Life', 'California', 'Caravelle', 'CC', 'Eos', 'Fox', 'Golf', 'Golf SV', 'Jetta', 'Passat', 'Polo', 'Scirocco', 'Sharan', 'T-Cross', 'T-Roc', 'Tiguan', 'Tiguan Allspace', 'Touareg', 'Touran', 'Up']
}

# Year mappings
years = {
    0: '1970', 1: '1991', 2: '1995', 3: '1996', 4: '1997', 5: '1998', 6: '1999', 7: '2000', 8: '2001', 9: '2002', 10: '2003',
    11: '2004', 12: '2005', 13: '2006', 14: '2007', 15: '2008', 16: '2009', 17: '2010', 18: '2011', 19: '2012', 20: '2013',
    21: '2014', 22: '2015', 23: '2016', 24: '2017', 25: '2018', 26: '2019', 27: '2020'
}
year_list = list(years.values())

# FuelType mappings
fuel_types = {0: 'Diesel', 1: 'Electric', 2: 'Hybrid', 3: 'Other', 4: 'Petrol'}
fuel_type_list = list(fuel_types.values())

# Streamlit app

st.header('Brand dan Model')
brand_data = {'Brand': [], 'Models': []}
for brand, models in brand_model_map.items():
    brand_data['Brand'].append(brand)
    brand_data['Models'].append(', '.join(models))

brand_df = pd.DataFrame(brand_data)
st.dataframe(brand_df)

st.header('Year')
year_df = pd.DataFrame(list(years.items()), columns=['Code', 'Year'])
st.dataframe(year_df)

st.header('FuelType')
fuel_type_df = pd.DataFrame(list(fuel_types.items()), columns=['Code', 'Fuel Type'])
st.dataframe(fuel_type_df)


###################################################
st.sidebar.markdown("# Data")

#################################################### hiding useless parts
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
