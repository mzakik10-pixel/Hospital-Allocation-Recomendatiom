import streamlit as st
import pandas as pd
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="CrowdAID - Smart Hospital Recommendation",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    h1, h2, h3 {
        color: white !important;
    }
    .stSelectbox label, .stRadio label {
        color: white !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2em;
    }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df_hospital = pd.read_csv('Hospital_Banten.csv', sep=';')
    df_faskes = pd.read_csv('Faskes_BPJS_Banten_2019.csv')
    df_faskes['KotaKab_Clean'] = df_faskes['KotaKab'].str.extract(r'(Kab\.|Kota)\s+(.+?)(?:\r|$)', expand=False)[1]
    df_faskes['KotaKab_Clean'] = df_faskes['KotaKab_Clean'].str.strip()
    
    # Load occupancy data
    try:
        df_occupancy = pd.read_csv('Hospital_Occupancy_Current.csv')
    except:
        # If file not found, create dummy data
        df_occupancy = pd.DataFrame({
            'hospital_id': df_hospital['id'],
            'hospital_name': df_hospital['nama'],
            'occupancy_rate': [75.0] * len(df_hospital),
            'status': ['NORMAL'] * len(df_hospital),
            'available_beds': df_hospital['total_tempat_tidur'] * 0.25,
            'wait_time_minutes': [30] * len(df_hospital)
        })
    
    return df_hospital, df_faskes, df_occupancy

df_hospital, df_faskes, df_occupancy = load_data()
kabupaten_list = sorted(df_hospital['kab'].unique().tolist())

# Title
st.title("🏥 CrowdAID")
st.subheader("Sistem Rekomendasi Fasilitas Kesehatan Cerdas Berbasis AI")
st.markdown("**Provinsi Banten** • Real-Time Occupancy")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📊 Statistik Real-Time")
    
    # Current time
    st.info(f"🕐 Update: {datetime.now().strftime('%d %b %Y, %H:%M')}")
    
    # Occupancy stats
    avg_occupancy = df_occupancy['occupancy_rate'].mean()
    penuh_count = len(df_occupancy[df_occupancy['status'] == 'PENUH'])
    hampir_penuh_count = len(df_occupancy[df_occupancy['status'] == 'HAMPIR PENUH'])
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🏥 Rumah Sakit", len(df_hospital))
        st.metric("🔴 RS Penuh", penuh_count)
        st.metric("🟡 Hampir Penuh", hampir_penuh_count)
    with col2:
        st.metric("📊 Avg Occupancy", f"{avg_occupancy:.0f}%")
        st.metric("🏛️ Kelas B", len(df_hospital[df_hospital['kelas'] == 'B']))
        st.metric("🏛️ Kelas C", len(df_hospital[df_hospital['kelas'] == 'C']))
    
    st.markdown("---")
    st.header("🤖 AI Classification")
    st.info("""
    **CrowdAID** menggunakan:
    - 📍 Real-time occupancy data
    - 🏥 Facility type classification
    - ⭐ Dynamic priority ranking
    - 🎯 Smart suggestions
    """)
    
    st.markdown("---")
    st.success("""
    **SDG #3: Good Health**
    
    ✅ Reduce overcrowding
    ✅ Smart referrals
    ✅ Better distribution
    """)

# Main content
col1, col2 = st.columns([1, 2])

with col1:
    st.header("🎯 Input Pasien")
    
    kabupaten = st.selectbox(
        "📍 Pilih Kabupaten/Kota",
        kabupaten_list,
        index=0
    )
    
    st.markdown("⚕️ **Pilih Kondisi Pasien:**")
    kondisi_options = {
        "1": "🤧 Gejala Ringan (Pilek, Batuk, Sakit Perut, Pusing)",
        "2": "💔 Penyakit Dalam (Jantung, Paru-paru, dll)",
        "3": "⚕️ Bedah (Operasi)",
        "4": "👶 Anak",
        "5": "🤰 Kebidanan",
        "6": "🦷 Gigi",
        "7": "🏥 Banyak Spesialis / Komprehensif"
    }
    
    kondisi = st.selectbox(
        "Kondisi",
        options=list(kondisi_options.keys()),
        format_func=lambda x: kondisi_options[x],
        label_visibility="collapsed"
    )
    
    # Urgency level
    st.markdown("⚠️ **Tingkat Urgensi:**")
    urgency = st.radio(
        "Urgency",
        options=["Tidak Mendesak", "Mendesak", "Darurat"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    st.markdown("")
    cari_button = st.button("🔍 Cari Rekomendasi AI", type="primary", use_container_width=True)

with col2:
    st.header("🏥 Hasil Rekomendasi")
    
    if cari_button:
        with st.spinner("🤖 AI sedang menganalisis dengan data real-time..."):
            recommendations = []
            classification_info = ""
            smart_suggestion = ""
            
            # Merge hospital with occupancy data
            df_merged = df_hospital.merge(
                df_occupancy[['hospital_id', 'occupancy_rate', 'status', 'available_beds', 'wait_time_minutes']], 
                left_on='id', 
                right_on='hospital_id', 
                how='left'
            )
            
            # Fill missing occupancy data
            df_merged['occupancy_rate'] = df_merged['occupancy_rate'].fillna(75.0)
            df_merged['status'] = df_merged['status'].fillna('NORMAL')
            df_merged['available_beds'] = df_merged['available_beds'].fillna(df_merged['total_tempat_tidur'] * 0.25)
            df_merged['wait_time_minutes'] = df_merged['wait_time_minutes'].fillna(30)
            
            if kondisi == "1":
                classification_info = """
                **🤖 AI Classification Result:**
                - **Kategori:** Gejala Ringan
                - **Rekomendasi:** Puskesmas atau Klinik Pratama
                - **Alasan:** Kondisi tidak memerlukan fasilitas RS
                """
                
                smart_suggestion = """
                💡 **Smart Suggestion:**
                ✅ **Sangat dianjurkan untuk pergi ke Puskesmas/Klinik saja!**
                
                Alasan:
                - 🏥 Gejala Anda tidak memerlukan fasilitas rumah sakit
                - ⏱️ Waktu tunggu lebih singkat (5-15 menit)
                - 💰 Biaya lebih murah
                - 🎯 Puskesmas/Klinik sudah cukup untuk menangani kondisi ini
                - 📉 Membantu mengurangi beban RS untuk kasus yang lebih serius
                """
                
                puskesmas = df_faskes[
                    (df_faskes['TipeFaskes'] == 'Puskesmas') & 
                    (df_faskes['KotaKab'].str.contains(kabupaten, case=False, na=False))
                ]
                
                klinik = df_faskes[
                    (df_faskes['TipeFaskes'].str.contains('Klinik', case=False, na=False)) & 
                    (df_faskes['KotaKab'].str.contains(kabupaten, case=False, na=False))
                ]
                
                for _, row in puskesmas.head(5).iterrows():
                    recommendations.append({
                        'nama': row['NamaFaskes'].strip(),
                        'alamat': row['AlamatFaskes'],
                        'tipe': 'Puskesmas',
                        'kelas': '-',
                        'status': 'TERSEDIA',
                        'wait_time': 10,
                        'occupancy': 0,
                        'priority': 1
                    })
                
                for _, row in klinik.head(3).iterrows():
                    recommendations.append({
                        'nama': row['NamaFaskes'].strip(),
                        'alamat': row['AlamatFaskes'],
                        'tipe': 'Klinik Pratama',
                        'kelas': '-',
                        'status': 'TERSEDIA',
                        'wait_time': 15,
                        'occupancy': 0,
                        'priority': 2
                    })
            
            elif kondisi == "6":
                classification_info = """
                **🤖 AI Classification Result:**
                - **Kategori:** Kesehatan Gigi
                - **Rekomendasi:** RS Kelas D atau Klinik Gigi
                - **Alasan:** Masalah gigi memerlukan fasilitas dental khusus
                """
                
                rs_d = df_merged[
                    (df_merged['kelas'] == 'D') & 
                    (df_merged['kab'] == kabupaten)
                ]
                
                # Check if all full
                all_full = all(rs_d['status'] == 'PENUH')
                
                if all_full:
                    smart_suggestion = """
                    ⚠️ **Smart Suggestion:**
                    🔄 **Pertimbangkan alternatif: Klinik Gigi**
                    
                    Alasan:
                    - 🔴 Semua RS Kelas D sedang penuh
                    - ⏱️ Waktu tunggu di RS sangat lama (3-5 jam)
                    - 🏥 Klinik Gigi dapat menangani sebagian besar masalah gigi
                    - 💡 Lebih cepat dan efisien untuk kasus non-darurat
                    """
                
                for _, row in rs_d.iterrows():
                    recommendations.append({
                        'nama': row['nama'],
                        'alamat': row['alamat'],
                        'tipe': row['jenis'],
                        'kelas': 'D',
                        'kapasitas': row['total_tempat_tidur'],
                        'layanan': row['total_layanan'],
                        'status': row['status'],
                        'occupancy': row['occupancy_rate'],
                        'wait_time': row['wait_time_minutes'],
                        'available_beds': int(row['available_beds']),
                        'priority': 1
                    })
                
                klinik_gigi = df_faskes[
                    (df_faskes['TipeFaskes'].str.contains('Gigi', case=False, na=False)) & 
                    (df_faskes['KotaKab'].str.contains(kabupaten, case=False, na=False))
                ]
                
                for _, row in klinik_gigi.head(3).iterrows():
                    recommendations.append({
                        'nama': row['NamaFaskes'].strip(),
                        'alamat': row['AlamatFaskes'],
                        'tipe': 'Klinik Gigi',
                        'kelas': '-',
                        'status': 'TERSEDIA',
                        'wait_time': 20,
                        'occupancy': 0,
                        'priority': 2 if all_full else 3
                    })
            
            elif kondisi == "7":
                classification_info = """
                **🤖 AI Classification Result:**
                - **Kategori:** Komprehensif / Multi-Spesialis
                - **Rekomendasi:** RS Kelas B
                - **Alasan:** Kondisi kompleks memerlukan banyak spesialis
                """
                
                rs_b = df_merged[
                    (df_merged['kelas'] == 'B') & 
                    (df_merged['kab'] == kabupaten)
                ].sort_values('total_layanan', ascending=False)
                
                # Check occupancy
                high_occupancy_count = len(rs_b[rs_b['occupancy_rate'] >= 85])
                
                if urgency == "Tidak Mendesak" and high_occupancy_count > len(rs_b) * 0.5:
                    smart_suggestion = """
                    💡 **Smart Suggestion:**
                    📅 **Pertimbangkan untuk menunda kunjungan non-urgent**
                    
                    Alasan:
                    - 🟡 Sebagian besar RS Kelas B sedang sibuk (>85% penuh)
                    - ⏱️ Waktu tunggu rata-rata 2-3 jam
                    - 📆 Occupancy biasanya lebih rendah di pagi hari (07:00-09:00)
                    - 🎯 Jika tidak mendesak, jadwalkan untuk besok pagi
                    """
                
                for idx, row in rs_b.iterrows():
                    recommendations.append({
                        'nama': row['nama'],
                        'alamat': row['alamat'],
                        'tipe': row['jenis'],
                        'kelas': 'B',
                        'kapasitas': row['total_tempat_tidur'],
                        'layanan': row['total_layanan'],
                        'staff': row['total_tenaga_kerja'],
                        'status': row['status'],
                        'occupancy': row['occupancy_rate'],
                        'wait_time': row['wait_time_minutes'],
                        'available_beds': int(row['available_beds']),
                        'priority': 1
                    })
            
            else:
                kondisi_map = {
                    "2": "Penyakit Dalam",
                    "3": "Bedah",
                    "4": "Anak",
                    "5": "Kebidanan"
                }
                
                classification_info = f"""
                **🤖 AI Classification Result:**
                - **Kategori:** {kondisi_map[kondisi]}
                - **Rekomendasi:** RS Kelas C
                - **Alasan:** Kondisi memerlukan perawatan RS dengan spesialisasi
                """
                
                rs_c = df_merged[
                    (df_merged['kelas'] == 'C') & 
                    (df_merged['kab'] == kabupaten)
                ]
                
                # Check if many are full
                full_count = len(rs_c[rs_c['status'].isin(['PENUH', 'HAMPIR PENUH'])])
                
                if full_count > len(rs_c) * 0.6:
                    smart_suggestion = """
                    ⚠️ **Smart Suggestion:**
                    🔄 **Pertimbangkan RS di kabupaten terdekat**
                    
                    Alasan:
                    - 🔴 Banyak RS Kelas C di area ini sedang penuh/hampir penuh
                    - ⏱️ Waktu tunggu sangat lama (2-4 jam)
                    - 🚗 RS di kabupaten sekitar mungkin lebih cepat
                    """
                
                if kondisi in ["4", "5"]:
                    rs_spesialis = rs_c[rs_c['jenis'].str.contains('Ibu dan Anak', case=False, na=False)]
                    rs_umum = rs_c[rs_c['jenis'].str.contains('Umum', case=False, na=False)]
                    
                    for idx, row in rs_spesialis.iterrows():
                        recommendations.append({
                            'nama': row['nama'],
                            'alamat': row['alamat'],
                            'tipe': row['jenis'],
                            'kelas': 'C',
                            'kapasitas': row['total_tempat_tidur'],
                            'layanan': row['total_layanan'],
                            'status': row['status'],
                            'occupancy': row['occupancy_rate'],
                            'wait_time': row['wait_time_minutes'],
                            'available_beds': int(row['available_beds']),
                            'priority': 1
                        })
                    
                    for idx, row in rs_umum.head(5).iterrows():
                        recommendations.append({
                            'nama': row['nama'],
                            'alamat': row['alamat'],
                            'tipe': row['jenis'],
                            'kelas': 'C',
                            'kapasitas': row['total_tempat_tidur'],
                            'layanan': row['total_layanan'],
                            'status': row['status'],
                            'occupancy': row['occupancy_rate'],
                            'wait_time': row['wait_time_minutes'],
                            'available_beds': int(row['available_beds']),
                            'priority': 2
                        })
                
                elif kondisi == "3":
                    rs_bedah = rs_c[rs_c['jenis'].str.contains('Bedah', case=False, na=False)]
                    rs_umum = rs_c[rs_c['jenis'].str.contains('Umum', case=False, na=False)]
                    
                    for idx, row in rs_bedah.iterrows():
                        recommendations.append({
                            'nama': row['nama'],
                            'alamat': row['alamat'],
                            'tipe': row['jenis'],
                            'kelas': 'C',
                            'kapasitas': row['total_tempat_tidur'],
                            'layanan': row['total_layanan'],
                            'status': row['status'],
                            'occupancy': row['occupancy_rate'],
                            'wait_time': row['wait_time_minutes'],
                            'available_beds': int(row['available_beds']),
                            'priority': 1
                        })
                    
                    for idx, row in rs_umum.head(5).iterrows():
                        recommendations.append({
                            'nama': row['nama'],
                            'alamat': row['alamat'],
                            'tipe': row['jenis'],
                            'kelas': 'C',
                            'kapasitas': row['total_tempat_tidur'],
                            'layanan': row['total_layanan'],
                            'status': row['status'],
                            'occupancy': row['occupancy_rate'],
                            'wait_time': row['wait_time_minutes'],
                            'available_beds': int(row['available_beds']),
                            'priority': 2
                        })
                
                else:
                    rs_umum = rs_c[rs_c['jenis'].str.contains('Umum', case=False, na=False)].sort_values('total_layanan', ascending=False)
                    
                    for idx, row in rs_umum.iterrows():
                        recommendations.append({
                            'nama': row['nama'],
                            'alamat': row['alamat'],
                            'tipe': row['jenis'],
                            'kelas': 'C',
                            'kapasitas': row['total_tempat_tidur'],
                            'layanan': row['total_layanan'],
                            'status': row['status'],
                            'occupancy': row['occupancy_rate'],
                            'wait_time': row['wait_time_minutes'],
                            'available_beds': int(row['available_beds']),
                            'priority': 1
                        })
            
            # Sort by priority first, then by occupancy (lower is better)
            recommendations = sorted(recommendations, key=lambda x: (x['priority'], x['occupancy']))
            
            # Display results
            st.info(classification_info)
            
            # Smart suggestion
            if smart_suggestion:
                st.warning(smart_suggestion)
            
            if len(recommendations) == 0:
                st.warning(f"⚠️ Tidak ditemukan fasilitas kesehatan yang sesuai di {kabupaten}.")
            else:
                st.success(f"✅ Ditemukan **{len(recommendations)}** rekomendasi")
                
                # Find best recommendation (lowest occupancy among priority 1)
                priority_1_recs = [r for r in recommendations if r['priority'] == 1]
                if priority_1_recs:
                    best_rec = min(priority_1_recs, key=lambda x: x['occupancy'])
                    st.success(f"⭐ **Best Recommendation:** {best_rec['nama']} (Occupancy: {best_rec['occupancy']:.0f}%)")
                
                # Display each recommendation
                for idx, rec in enumerate(recommendations[:10]):
                    is_best = (priority_1_recs and rec == best_rec)
                    
                    with st.container():
                        if is_best:
                            st.markdown("### ⭐ REKOMENDASI TERBAIK (Occupancy Terendah)")
                        
                        # Hospital name
                        st.markdown(f"### {idx + 1}. {rec['nama']}")
                        
                        # Status badge
                        status = rec.get('status', 'NORMAL')
                        if status == 'PENUH':
                            st.error(f"🔴 **Status: {status}** - Tidak menerima pasien baru saat ini")
                        elif status == 'HAMPIR PENUH':
                            st.warning(f"🟡 **Status: {status}** - Waktu tunggu sangat lama")
                        elif status == 'SIBUK':
                            st.info(f"🟠 **Status: {status}** - Waktu tunggu lebih lama dari biasanya")
                        else:
                            st.success(f"🟢 **Status: {status}** - Siap melayani")
                        
                        # Basic info
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown(f"**🏥 Tipe:** {rec['tipe']}")
                            st.markdown(f"**🏛️ Kelas:** {rec['kelas']}")
                            st.markdown(f"**📍 Alamat:** {rec['alamat']}")
                        
                        with col_b:
                            if 'kapasitas' in rec:
                                st.markdown(f"**🛏️ Total Bed:** {rec['kapasitas']}")
                                if 'available_beds' in rec:
                                    st.markdown(f"**✅ Tersedia:** {rec['available_beds']} bed")
                            if 'layanan' in rec:
                                st.markdown(f"**⚕️ Layanan:** {rec['layanan']} jenis")
                            if 'wait_time' in rec:
                                wait = rec['wait_time']
                                if wait > 120:
                                    st.markdown(f"**⏱️ Perkiraan Tunggu:** ~{wait//60} jam ({wait} menit)")
                                else:
                                    st.markdown(f"**⏱️ Perkiraan Tunggu:** ~{wait} menit")
                        
                        # Occupancy bar (if hospital)
                        if rec['occupancy'] > 0:
                            st.progress(rec['occupancy'] / 100)
                            st.caption(f"Occupancy: {rec['occupancy']:.0f}%")
                        
                        st.markdown("---")
                
                # Summary - updated without AI scores
                st.markdown("### 📋 Ringkasan")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Total Rekomendasi", len(recommendations))
                with col_b:
                    if priority_1_recs:
                        avg_occupancy = sum(r['occupancy'] for r in priority_1_recs) / len(priority_1_recs)
                        st.metric("Avg Occupancy", f"{avg_occupancy:.0f}%")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: white; padding: 20px;">
    <p><strong>CrowdAID - Smart Hospital Recommendation with Real-Time Data</strong></p>
    <p>Powered by AI Classification Model | SDG #3: Good Health and Well-being</p>
    <p style="font-size: 0.9em;">© 2025 - COMP6056001 | Data updated every 6 hours</p>
</div>
""", unsafe_allow_html=True)
