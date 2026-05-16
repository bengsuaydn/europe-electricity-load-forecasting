import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Enerji Yükü Analizi", layout="wide")

st.title("📊 Enerji Yükü Dashboard'u")
st.markdown("Aykırı değerlerden arındırılmış, tertemiz kontrol paneli!")

@st.cache_data
def load_data():
    file_path = 'data/engineered_data.txt'
    df = pd.read_csv(file_path, sep=';', low_memory=False)
    return df

try:
    df = load_data()
    
    # 10.000 satırlık örneklem alıyoruz
    df_sample = df.sample(n=10000, random_state=42)

    # --- HAYAT KURTARAN DÜZELTME: Aykırı Değer (Outlier) Temizliği ---
    # Verinin en yüksek %2'sini ve en düşük %1'ini grafiği bozmaması için filtreliyoruz
    alt_sinir = df_sample['Value'].quantile(0.01)
    ust_sinir = df_sample['Value'].quantile(0.98)
    df_sample = df_sample[(df_sample['Value'] >= alt_sinir) & (df_sample['Value'] <= ust_sinir)]
    # -----------------------------------------------------------------

    st.sidebar.header("🔍 Veri Filtreleri")
    secilen_yil = st.sidebar.multiselect("Yıl Seçin", options=df_sample['year'].unique(), default=df_sample['year'].unique())
    
    df_filtered = df_sample[df_sample['year'].isin(secilen_yil)]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 1. Zaman İçinde Değer Değişimi")
        fig1 = px.line(df_filtered.head(1000).reset_index(), y="Value", title="İlk 1000 Satırın Trendi")
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("📦 3. Saatlere Göre Yük Dağılımı")
        fig3 = px.box(df_filtered, x="Hour", y="Value", color="Hour", title="Günün Saatlerine Göre Dalgalanma")
        st.plotly_chart(fig3, use_container_width=True)
        
        st.subheader("🌌 5. Gerçek Değer vs 24 Saat Önceki Yük")
        fig5 = px.scatter(df_filtered, x="Load_Lag_24h", y="Value", opacity=0.5, color="IsWeekend", title="Gecikmeli Veri Korelasyonu")
        st.plotly_chart(fig5, use_container_width=True)

    with col2:
        st.subheader("📊 2. Aylara Göre Ortalama Değer")
        aylik_ortalama = df_filtered.groupby("Month")["Value"].mean().reset_index()
        fig2 = px.bar(aylik_ortalama, x="Month", y="Value", text_auto='.2s', color="Month", title="Aylık Ortalama Yük Değerleri")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("🌳 4. Yıl, Ay ve Haftanın Gününe Göre Hiyerarşi")
        treemap_data = df_filtered.groupby(['year', 'Month', 'DayOfWeek'])['Value'].mean().reset_index()
        treemap_data['Value'] = treemap_data['Value'].abs() 
        fig4 = px.treemap(treemap_data, path=['year', 'Month', 'DayOfWeek'], values='Value', color='Value', title="Zaman Hiyerarşisi Treemap")
        st.plotly_chart(fig4, use_container_width=True)

        st.subheader("📶 6. Değerlerin Genel Dağılımı (Frekans)")
        fig6 = px.histogram(df_filtered, x="Value", nbins=50, color_discrete_sequence=['indianred'], title="Değer (Value) Dağılımı")
        st.plotly_chart(fig6, use_container_width=True)

except Exception as e:
    st.error(f"Dosya okunurken bir hata oluştu: {e}")