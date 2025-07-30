import streamlit as st
import pandas as pd
from arch.unitroot import ZivotAndrews, PhillipsPerron
import plotly.graph_objects as go

# -- Konfigurasi Halaman Utama Streamlit --
st.set_page_config(
    page_title="Aplikasi Uji Stasioneritas",
    page_icon="📊",
    layout="wide"
)

# =============================================================================
# FUNGSI UNTUK APLIKASI UJI ZIVOT-ANDREWS
# =============================================================================
def zivot_andrews_app(df):
    st.header("Uji Zivot-Andrews")
    st.markdown("Uji ini digunakan untuk data time series dengan **satu kali patahan struktural**.")

    # --- Pengaturan Khusus ZA di Sidebar ---
    st.sidebar.subheader("Pengaturan Uji Zivot-Andrews")
    
    # Pilihan Kolom Numerik
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    selected_column = st.sidebar.selectbox(
        "Pilih variabel yang akan diuji",
        options=numeric_cols,
        key='za_column'
    )

    # Pengaturan Model Uji
    model_option = st.sidebar.selectbox(
        "Pilih model patahan struktural",
        options=['Intercept', 'Trend', 'Both'],
        key='za_model',
        help="Pilih jenis patahan yang diuji: 'Intercept' (c), 'Trend' (t), atau 'Both' (ct)."
    )
    model_map = {'Intercept': 'c', 'Trend': 't', 'Both': 'ct'}
    arch_model = model_map[model_option]

    # Pengaturan Lag
    lag_method = st.sidebar.selectbox(
        "Pilih metode penentuan lag",
        options=['AIC', 'BIC', 't-stat'],
        key='za_lag_method',
    )
    max_lags = st.sidebar.number_input(
        "Masukkan jumlah maksimum lag",
        min_value=0,
        value=int(len(df)**(1/3)),
        key='za_max_lags',
    )

    # Tombol untuk menjalankan analisis
    if st.sidebar.button("🚀 Jalankan Uji", key='za_run'):
        series_to_test = df[selected_column].dropna()

        if len(series_to_test) < 20:
            st.warning("Data terlalu sedikit untuk hasil yang andal.")
        else:
            with st.spinner('Menjalankan Uji Zivot-Andrews...'):
                za_test = ZivotAndrews(series_to_test, lags=max_lags, trend=arch_model, method=lag_method)
                break_index = za_test.breakpoint
                break_date = series_to_test.index[break_index]

                st.subheader("🔬 Hasil Uji")
                col1, col2, col3 = st.columns(3)
                col1.metric("Statistik Uji", f"{za_test.stat:.4f}")
                col2.metric("P-value", f"{za_test.pvalue:.4f}")
                col3.metric("Tanggal Patahan", str(break_date.date()) if isinstance(break_date, pd.Timestamp) else str(break_date))

                st.subheader("Kesimpulan Uji")
                if za_test.pvalue < 0.05:
                    st.success("**Tolak Hipotesis Nol ($H_0$)**. Data **stasioner** dengan adanya patahan struktural.")
                else:
                    st.warning("**Gagal Tolak Hipotesis Nol ($H_0$)**. Data **tidak stasioner**.")

                st.subheader("Nilai Kritis")
                crit_values_df = pd.DataFrame({
                    'Tingkat Signifikansi': ['1%', '5%', '10%'],
                    'Nilai Kritis': [za_test.cv_1, za_test.cv_5, za_test.cv_10]
                }).set_index('Tingkat Signifikansi')
                st.table(crit_values_df)
                st.caption(f"Lag yang digunakan dalam model: {za_test.lags}")

                st.header("📈 Visualisasi")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=series_to_test.index, y=series_to_test, mode='lines', name=selected_column))
                fig.add_vline(x=break_date, line_width=2, line_dash="dash", line_color="red", annotation_text="Patahan Terdeteksi", annotation_position="top right")
                fig.update_layout(title=f'Plot "{selected_column}" dengan Patahan Struktural', xaxis_title='Tanggal', yaxis_title='Nilai')
                st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# FUNGSI UNTUK APLIKASI UJI PHILLIPS-PERRON
# =============================================================================
def phillips_perron_app(df):
    st.header("Uji Phillips-Perron (PP)")
    st.markdown("Uji ini digunakan untuk menguji stasioneritas pada data time series secara umum.")

    # --- Pengaturan Khusus PP di Sidebar ---
    st.sidebar.subheader("Pengaturan Uji Phillips-Perron")

    # Pilihan Kolom Numerik
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    selected_column = st.sidebar.selectbox(
        "Pilih variabel yang akan diuji",
        options=numeric_cols,
        key='pp_column'
    )

    # Pengaturan Model Uji (Trend)
    trend_option = st.sidebar.selectbox(
        "Pilih komponen deterministik",
        options=['Constant', 'Constant and Trend', 'No Trend/Constant'],
        index=0,
        key='pp_trend'
    )
    trend_map = {'Constant': 'c', 'Constant and Trend': 'ct', 'No Trend/Constant': 'n'}
    arch_trend = trend_map[trend_option]
    
    # Pengaturan Lag
    lags = st.sidebar.number_input(
        "Masukkan jumlah lag (0 untuk otomatis)",
        min_value=0,
        value=0, 
        key='pp_lags',
        help="Biarkan 0 untuk pemilihan otomatis oleh library."
    )
    lags_to_use = None if lags == 0 else lags

    # Tombol untuk menjalankan analisis
    if st.sidebar.button("🚀 Jalankan Uji", key='pp_run'):
        series_to_test = df[selected_column].dropna()

        if len(series_to_test) < 20:
            st.warning("Data terlalu sedikit untuk hasil yang andal.")
        else:
            with st.spinner('Menjalankan Uji Phillips-Perron...'):
                pp_test = PhillipsPerron(series_to_test, trend=arch_trend, lags=lags_to_use)

                st.subheader("🔬 Hasil Uji")
                col1, col2, col3 = st.columns(3)
                col1.metric("Statistik Uji (τ)", f"{pp_test.stat:.4f}")
                col2.metric("P-value", f"{pp_test.pvalue:.4f}")
                col3.metric("Lags Digunakan", pp_test.lags)
                
                st.subheader("Kesimpulan Uji")
                if pp_test.pvalue < 0.05:
                    st.success("**Tolak Hipotesis Nol ($H_0$)**. Data **stasioner**.")
                else:
                    st.warning("**Gagal Tolak Hipotesis Nol ($H_0$)**. Data **tidak stasioner**.")

                st.subheader("Nilai Kritis")
                crit_values_df = pd.DataFrame.from_dict(
                    pp_test.critical_values, 
                    orient='index', 
                    columns=['Nilai Kritis']
                )
                crit_values_df.index.name = "Tingkat Signifikansi"
                st.table(crit_values_df)
                
                st.header("📊 Visualisasi")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=series_to_test.index, y=series_to_test, mode='lines', name=selected_column))
                fig.update_layout(title=f'Time Series Plot untuk "{selected_column}"', xaxis_title='Tanggal', yaxis_title='Nilai')
                st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# --- LOGIKA UTAMA APLIKASI ---
# =============================================================================

# Judul utama aplikasi
st.title("Aplikasi Uji Stasioneritas Time Series")
st.markdown("---")

# Menu utama di sidebar untuk memilih uji
st.sidebar.title("Navigasi")
pilihan_uji = st.sidebar.selectbox("Pilih Uji Statistik:", 
    ["Uji Zivot-Andrews", "Uji Phillips-Perron"])

st.sidebar.markdown("---")

# Bagian upload data yang umum untuk kedua tes
st.sidebar.header("Unggah Data Anda")
uploaded_file = st.sidebar.file_uploader(
    "Unggah file .csv atau .xlsx",
    type=["csv", "xlsx"]
)

# Jika file sudah diunggah, proses dan panggil fungsi yang sesuai
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success("File berhasil diunggah!")

        # Opsi untuk memilih kolom tanggal dan menjadikannya index
        date_cols = [col for col in df.columns if df[col].dtype in ['datetime64[ns]', 'object']]
        if date_cols:
            date_col = st.sidebar.selectbox("Pilih kolom Tanggal/Waktu sebagai Indeks", date_cols, key='date_col')
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                df.set_index(date_col, inplace=True)
            except Exception as e:
                st.sidebar.error(f"Gagal mengubah kolom tanggal: {e}")
                st.stop()
        
        # Tampilkan pratinjau data di halaman utama
        st.subheader("Pratinjau Data")
        st.dataframe(df.head())
        st.markdown("---")
        
        # Panggil fungsi aplikasi yang sesuai berdasarkan pilihan di sidebar
        if pilihan_uji == "Uji Zivot-Andrews":
            zivot_andrews_app(df)
        elif pilihan_uji == "Uji Phillips-Perron":
            phillips_perron_app(df)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")
        st.exception(e)
else:
    st.info("Silakan unggah file data Anda di sidebar untuk memulai.")
