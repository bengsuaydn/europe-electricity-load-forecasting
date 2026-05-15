import streamlit as st
import pandas as pd
import plotly.express as px

# Sayfa konfigürasyonu ve geniş ekran modu aktivasyonu
st.set_page_config(page_title="Enerji Yükü Analizi", layout="wide")

st.title("📊 Enerji Yükü Analiz Dashboard'u")
st.markdown("Veri seti üzerinde gerçekleştirilen öngörüsel analiz ve görselleştirme sonuçları.")

@st.cache_data
def load_data():
    """
    İşlenmiş enerji verisini yükler. 
    Proje dizin yapısına uygun göreceli (relative) dosya yolu kullanılmıştır.
    """
    file_path = 'data/engineered_data.csv' 
    df = pd.read_csv(file_path, sep=';', low_memory=False)
    return df

try:
    # Veri yükleme ve örneklem oluşturma
    df = load_data()
    
    # Performans optimizasyonu için 10.000 satırlık rastgele örneklem (sampling)
    df_sample = df.sample(n=10000, random_state=42)

    # --- VERİ ÖN İŞLEME: Aykırı Değer (Outlier) Analizi ---
    # Grafiklerin genel dağılımını bozmamak adına 0.01 ve 0.98 kuantil aralığı filtrelenmiştir.
    lower_bound = df_sample['Value'].quantile(0.01)
    upper_bound = df_sample['Value'].quantile(0.98)
    df_sample = df_sample[(df_sample['Value'] >= lower_bound) & (df_sample['Value'] <= upper_bound)]
    # -----------------------------------------------------

    # Kullanıcı etkileşimi için yan panel (sidebar) filtreleri
    st.sidebar.header("🔍 Veri Filtreleme Paneli")
    selected_year = st.sidebar.multiselect("Analiz Edilecek Yıl Seçimi", 
                                          options=df_sample['year'].unique(), 
                                          default=df_sample['year'].unique())
    
    df_filtered = df_sample[df_sample['year'].isin(selected_year)]

    # Görselleştirme katmanı: İki sütunlu yerleşim düzeni
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Zaman Serisi Trend Analizi")
        fig1 = px.line(df_filtered.head(1000).reset_index(), y="Value", title="Zaman İçindeki Değişim (İlk 1000 Gözlem)")
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("📦 Saatlik Yük Dağılımı (Box Plot)")
        fig3 = px.box(df_filtered, x="Hour", y="Value", color="Hour", title="Günün Saatlerine Göre Enerji Dalgalanması")
        st.plotly_chart(fig3, use_container_width=True)
        
        st.subheader("🌌 Gecikmeli Veri Korelasyon Analizi")
        fig5 = px.scatter(df_filtered, x="Load_Lag_24h", y="Value", opacity=0.5, 
                          color="IsWeekend", title="Mevcut Yük vs. 24 Saat Önceki Yük")
        st.plotly_chart(fig5, use_container_width=True)

    with col2:
        st.subheader("📊 Aylık Ortalama Yük Dağılımı")
        monthly_avg = df_filtered.groupby("Month")["Value"].mean().reset_index()
        fig2 = px.bar(monthly_avg, x="Month", y="Value", text_auto='.2s', color="Month", title="Aylık Bazda Ortalama Enerji Tüketimi")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("🌳 Zaman Hiyerarşisi Analizi (Treemap)")
        treemap_data = df_filtered.groupby(['year', 'Month', 'DayOfWeek'])['Value'].mean().reset_index()
        treemap_data['Value'] = treemap_data['Value'].abs() 
        fig4 = px.treemap(treemap_data, path=['year', 'Month', 'DayOfWeek'], values='Value', color='Value', title="Yıl/Ay/Gün Bazlı Hiyerarşik Dağılım")
        st.plotly_chart(fig4, use_container_width=True)

        st.subheader("📶 Genel Değer Dağılımı (Histogram)")
        fig6 = px.histogram(df_filtered, x="Value", nbins=50, color_discrete_sequence=['indianred'], title="Enerji Değerleri Frekans Dağılımı")
        st.plotly_chart(fig6, use_container_width=True)

except Exception as e:
    st.error(f"Sistem hatası: Veri okuma veya işleme sırasında bir aksaklık oluştu: {e}")
