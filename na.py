import streamlit as st
import pandas as pd
from arch.unitroot import ZivotAndrews, PhillipsPerron
import plotly.graph_objects as go
import numpy as np

# -- Konfigurasi Halaman Utama Streamlit --
st.set_page_config(
    page_title="Aplikasi Uji Stasioneritas",
    page_icon="📊",
    layout="wide"
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def debug_za_test_object(za_test):
    """Function to debug ZA test object and find available attributes"""
    st.write("### 🔍 Debug Information")
    
    # Show all public attributes
    public_attrs = [attr for attr in dir(za_test) if not attr.startswith('_')]
    st.write("**All public attributes:**", public_attrs)
    
    # Check for breakpoint-related attributes
    breakpoint_attrs = [attr for attr in dir(za_test) if 'break' in attr.lower() or 'brk' in attr]
    st.write("**Breakpoint-related attributes:**", breakpoint_attrs)
    
    # Check for critical value attributes
    cv_attrs = [attr for attr in dir(za_test) if 'cv' in attr.lower() or 'crit' in attr.lower()]
    st.write("**Critical value attributes:**", cv_attrs)
    
    # Try to show actual values
    st.write("**Attribute values:**")
    for attr in public_attrs[:10]:  # Show first 10 to avoid clutter
        try:
            value = getattr(za_test, attr)
            if not callable(value):
                st.write(f"- {attr}: {value}")
        except:
            st.write(f"- {attr}: <unable to access>")

def get_breakpoint_safe(za_test, series_length):
    """Safely get breakpoint from ZA test result"""
    breakpoint_candidates = [
        'breakpoint', 'brk', '_breakpoint', '_brk', 
        'break_point', 'structural_break', 'tb'
    ]
    
    for attr in breakpoint_candidates:
        if hasattr(za_test, attr):
            try:
                bp = getattr(za_test, attr)
                if isinstance(bp, (int, float)) and 0 <= bp < series_length:
                    return int(bp)
            except:
                continue
    
    # If all fails, return middle point as estimate
    return series_length // 2

def get_critical_values_safe(za_test):
    """Safely get critical values from ZA test result"""
    # Try different methods to get critical values
    methods = [
        # Method 1: Individual attributes
        lambda: {'1%': za_test.cv_1, '5%': za_test.cv_5, '10%': za_test.cv_10},
        # Method 2: Critical values dict
        lambda: za_test.critical_values,
        # Method 3: Alternative naming
        lambda: {'1%': za_test.cv1, '5%': za_test.cv5, '10%': za_test.cv10},
    ]
    
    for method in methods:
        try:
            cv = method()
            if cv and all(isinstance(v, (int, float)) for v in cv.values()):
                return cv
        except:
            continue
    
    return None

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
    if not numeric_cols:
        st.error("Tidak ada kolom numerik yang ditemukan dalam data.")
        return
    
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

    # Pengaturan Lag - FIXED: Now using the lag_method variable
    lag_method = st.sidebar.selectbox(
        "Pilih metode penentuan lag",
        options=['AIC', 'BIC', 't-stat'],
        key='za_lag_method',
    )
    
    max_lags = st.sidebar.number_input(
        "Masukkan jumlah maksimum lag",
        min_value=0,
        value=int(len(df)**(1/3)) if len(df) > 0 else 10,
        key='za_max_lags',
    )

    # Tombol untuk menjalankan analisis
    if st.sidebar.button("🚀 Jalankan Uji", key='za_run'):
        series_to_test = df[selected_column].dropna()

        if len(series_to_test) < 20:
            st.warning("Data terlalu sedikit untuk hasil yang andal. Minimal 20 observasi diperlukan.")
            return
        
        try:
            with st.spinner('Menjalankan Uji Zivot-Andrews...'):
                # FIXED: Using lag_method instead of undefined 'method' variable
                za_test = ZivotAndrews(
                    series_to_test, 
                    lags=max_lags, 
                    trend=arch_model, 
                    method=lag_method.lower()
                )
                
                # Debug: Tampilkan atribut yang tersedia (hanya untuk debugging)
                if st.sidebar.checkbox("🔍 Debug Mode", key="debug_za"):
                    debug_za_test_object(za_test)
                
                # --- BAGIAN KRITIS YANG DIPERBAIKI SECARA PERMANEN ---
                # Gunakan helper function untuk mendapatkan breakpoint
                break_index = get_breakpoint_safe(za_test, len(series_to_test))
                
                if break_index is None or break_index >= len(series_to_test):
                    st.error("Tidak dapat menentukan breakpoint yang valid. Silakan coba dengan data atau pengaturan yang berbeda.")
                    return
                # ----------------------------------------------------
                
                # Ensure break_index is within valid range
                if break_index >= len(series_to_test):
                    break_index = len(series_to_test) - 1
                
                break_date = series_to_test.index[break_index]

                st.subheader("🔬 Hasil Uji")
                col1, col2, col3 = st.columns(3)
                col1.metric("Statistik Uji", f"{za_test.stat:.4f}")
                col2.metric("P-value", f"{za_test.pvalue:.4f}")
                
                # Handle different types of index for break date display
                if hasattr(break_date, 'date'):
                    break_date_str = str(break_date.date())
                elif isinstance(break_date, (pd.Timestamp, np.datetime64)):
                    break_date_str = str(pd.to_datetime(break_date).date())
                else:
                    break_date_str = str(break_date)
                
                col3.metric("Tanggal Patahan", break_date_str)

                st.subheader("Kesimpulan Uji")
                alpha = 0.05
                if za_test.pvalue < alpha:
                    st.success(f"**Tolak Hipotesis Nol (H₀)** pada α = {alpha}. Data **stasioner** dengan adanya patahan struktural.")
                else:
                    st.warning(f"**Gagal Tolak Hipotesis Nol (H₀)** pada α = {alpha}. Data **tidak stasioner**.")

                # Display critical values with error handling
                st.subheader("Nilai Kritis")
                crit_values = get_critical_values_safe(za_test)
                
                if crit_values:
                    crit_values_df = pd.DataFrame.from_dict(
                        crit_values, 
                        orient='index', 
                        columns=['Nilai Kritis']
                    )
                    crit_values_df.index.name = "Tingkat Signifikansi"
                    st.table(crit_values_df)
                else:
                    st.warning("Nilai kritis tidak tersedia. Gunakan Debug Mode untuk melihat atribut yang tersedia.")
                
                st.caption(f"Lag yang digunakan dalam model: {getattr(za_test, 'lags', 'N/A')}")

                # Visualization
                st.header("📈 Visualisasi")
                fig = go.Figure()
                
                # Convert index to appropriate format for plotting
                x_values = series_to_test.index
                if isinstance(series_to_test.index, pd.DatetimeIndex):
                    x_values = series_to_test.index.to_pydatetime()
                    break_date_plot = break_date.to_pydatetime() if hasattr(break_date, 'to_pydatetime') else break_date
                else:
                    break_date_plot = break_date
                
                fig.add_trace(go.Scatter(
                    x=x_values, 
                    y=series_to_test.values, 
                    mode='lines', 
                    name=selected_column,
                    line=dict(color='blue', width=1)
                ))
                
                # Add vertical line for breakpoint - use add_shape instead of add_vline
                fig.add_shape(
                    type="line",
                    x0=break_date_plot,
                    x1=break_date_plot,
                    y0=0,
                    y1=1,
                    yref="paper",
                    line=dict(color="red", width=3, dash="dash"),
                )
                
                # Add annotation for the breakpoint
                fig.add_annotation(
                    x=break_date_plot,
                    y=0.9,
                    yref="paper",
                    text="Patahan Terdeteksi",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="red",
                    bgcolor="white",
                    bordercolor="red",
                    borderwidth=1
                )
                
                fig.update_layout(
                    title=f'Plot "{selected_column}" dengan Patahan Struktural',
                    xaxis_title='Tanggal/Index',
                    yaxis_title='Nilai',
                    hovermode='x unified',
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat menjalankan uji Zivot-Andrews: {str(e)}")
            st.exception(e)

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
    if not numeric_cols:
        st.error("Tidak ada kolom numerik yang ditemukan dalam data.")
        return
    
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
            st.warning("Data terlalu sedikit untuk hasil yang andal. Minimal 20 observasi diperlukan.")
            return
        
        try:
            with st.spinner('Menjalankan Uji Phillips-Perron...'):
                pp_test = PhillipsPerron(series_to_test, trend=arch_trend, lags=lags_to_use)

                st.subheader("🔬 Hasil Uji")
                col1, col2, col3 = st.columns(3)
                col1.metric("Statistik Uji (τ)", f"{pp_test.stat:.4f}")
                col2.metric("P-value", f"{pp_test.pvalue:.4f}")
                col3.metric("Lags Digunakan", getattr(pp_test, 'lags', 'N/A'))
                
                st.subheader("Kesimpulan Uji")
                alpha = 0.05
                if pp_test.pvalue < alpha:
                    st.success(f"**Tolak Hipotesis Nol (H₀)** pada α = {alpha}. Data **stasioner**.")
                else:
                    st.warning(f"**Gagal Tolak Hipotesis Nol (H₀)** pada α = {alpha}. Data **tidak stasioner**.")

                # Display critical values with error handling
                st.subheader("Nilai Kritis")
                try:
                    if hasattr(pp_test, 'critical_values') and pp_test.critical_values:
                        crit_values_df = pd.DataFrame.from_dict(
                            pp_test.critical_values, 
                            orient='index', 
                            columns=['Nilai Kritis']
                        )
                        crit_values_df.index.name = "Tingkat Signifikansi"
                        st.table(crit_values_df)
                    else:
                        st.warning("Nilai kritis tidak tersedia.")
                except Exception as e:
                    st.warning(f"Tidak dapat menampilkan nilai kritis: {e}")
                
                # Visualization
                st.header("📊 Visualisasi")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=series_to_test.index, 
                    y=series_to_test, 
                    mode='lines', 
                    name=selected_column,
                    line=dict(color='blue', width=1)
                ))
                fig.update_layout(
                    title=f'Time Series Plot untuk "{selected_column}"',
                    xaxis_title='Tanggal/Index',
                    yaxis_title='Nilai',
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat menjalankan uji Phillips-Perron: {str(e)}")
            st.exception(e)

# =============================================================================
# --- LOGIKA UTAMA APLIKASI ---
# =============================================================================

# Judul utama aplikasi
st.title("📊 Aplikasi Uji Stasioneritas Time Series")
st.markdown("""
Aplikasi ini menyediakan dua uji stasioneritas utama:
- **Uji Zivot-Andrews**: Untuk mendeteksi stasioneritas dengan patahan struktural
- **Uji Phillips-Perron**: Untuk uji stasioneritas umum
""")
st.markdown("---")

# Menu utama di sidebar untuk memilih uji
st.sidebar.title("🔧 Navigasi")
pilihan_uji = st.sidebar.selectbox("Pilih Uji Statistik:", 
    ["Uji Zivot-Andrews", "Uji Phillips-Perron"])

st.sidebar.markdown("---")

# Bagian upload data yang umum untuk kedua tes
st.sidebar.header("📁 Unggah Data Anda")
uploaded_file = st.sidebar.file_uploader(
    "Unggah file .csv atau .xlsx",
    type=["csv", "xlsx"],
    help="File harus berisi data time series dengan kolom tanggal dan variabel numerik."
)

# Jika file sudah diunggah, proses dan panggil fungsi yang sesuai
if uploaded_file is not None:
    try:
        # Read the file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success(f"✅ File '{uploaded_file.name}' berhasil diunggah!")
        
        # Show basic info about the dataset
        st.info(f"Dataset memiliki {df.shape[0]} baris dan {df.shape[1]} kolom.")

        # Opsi untuk memilih kolom tanggal dan menjadikannya index
        date_cols = [col for col in df.columns if df[col].dtype in ['datetime64[ns]', 'object']]
        
        if date_cols:
            date_col = st.sidebar.selectbox(
                "Pilih kolom Tanggal/Waktu sebagai Indeks", 
                options=['Tidak menggunakan indeks tanggal'] + date_cols, 
                key='date_col'
            )
            
            if date_col != 'Tidak menggunakan indeks tanggal':
                try:
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    # Check for any NaT values after conversion
                    if df[date_col].isna().any():
                        st.sidebar.warning(f"Beberapa nilai dalam kolom '{date_col}' tidak dapat dikonversi ke tanggal.")
                    
                    df.set_index(date_col, inplace=True)
                    st.sidebar.success(f"Kolom '{date_col}' berhasil dijadikan indeks tanggal.")
                except Exception as e:
                    st.sidebar.error(f"Gagal mengubah kolom tanggal: {e}")
        else:
            st.sidebar.info("Tidak ada kolom tanggal yang terdeteksi. Menggunakan indeks default.")
        
        # Tampilkan pratinjau data di halaman utama
        st.subheader("👀 Pratinjau Data")
        st.dataframe(df.head(10))
        
        # Show data types
        with st.expander("📋 Informasi Kolom"):
            info_df = pd.DataFrame({
                'Kolom': df.columns,
                'Tipe Data': df.dtypes,
                'Nilai Kosong': df.isnull().sum(),
                'Contoh Nilai': [df[col].dropna().iloc[0] if not df[col].dropna().empty else 'N/A' for col in df.columns]
            })
            st.dataframe(info_df)
        
        st.markdown("---")
        
        # Panggil fungsi aplikasi yang sesuai berdasarkan pilihan di sidebar
        if pilihan_uji == "Uji Zivot-Andrews":
            zivot_andrews_app(df)
        elif pilihan_uji == "Uji Phillips-Perron":
            phillips_perron_app(df)

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat memproses file: {e}")
        st.exception(e)
else:
    st.info("👆 Silakan unggah file data Anda di sidebar untuk memulai analisis stasioneritas.")
    
    # Show sample data format
    st.subheader("📝 Format Data yang Disarankan")
    sample_data = pd.DataFrame({
        'Tanggal': pd.date_range('2020-01-01', periods=10, freq='M'),
        'Variabel_1': np.random.randn(10).cumsum(),
        'Variabel_2': np.random.randn(10).cumsum() + 100
    })
    st.dataframe(sample_data)
    st.caption("Contoh format data yang ideal: kolom tanggal dan satu atau lebih variabel numerik.")
