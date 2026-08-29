import ssl
import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go

# 免疫补丁
if not hasattr(sys, '_streamlit_patched'):
    orig_create_default_context = ssl.create_default_context


    def safe_create_default_context(*args, **kwargs):
        try:
            return orig_create_default_context(*args, **kwargs)
        except ssl.SSLError:
            return ssl.SSLContext()


    ssl.create_default_context = safe_create_default_context
    sys._streamlit_patched = True

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

from model_builder import SOTA_Graph_KAN_Model, smiles_to_graph_matrices


@st.cache_resource
def load_model():
    device = torch.device('cpu')
    model = SOTA_Graph_KAN_Model()
    model_paths = ["FluoroSpec_Final_Production_Model.pt", "DeePFAS_KAN_Final_Production_Model.pt"]
    found_path = next((p for p in model_paths if os.path.exists(p)), None)

    if found_path:
        try:
            state_dict = torch.load(found_path, map_location=device, weights_only=True)
            new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
            model.eval()
            return model, True
        except Exception as e:
            return None, f"Model loading failed: {str(e)}"
    return None, "Model file not found. Please ensure the .pt file is in the directory."


def calculate_molecular_features(smiles, ce_val):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    c_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
    f_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'F')
    exact_mass = Descriptors.ExactMolWt(mol)
    pmz_raw = exact_mass - 1.0078
    return [(pmz_raw - 500.0) / 500.0, float(ce_val), float(c_count), float(f_count)], exact_mass


def rigorous_mass_annotation(nominal_mz_array, smiles):
    mol = Chem.MolFromSmiles(smiles)
    max_c = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
    max_f = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'F')
    max_o = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'O')
    max_s = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'S')

    exact_masses = {'C': 12.000000, 'F': 18.998403, 'O': 15.994915, 'S': 31.972071}
    nominal_masses = {'C': 12, 'F': 19, 'O': 16, 'S': 32}

    exact_mz_array = []
    for target_nominal in nominal_mz_array:
        best_exact_mass = float(target_nominal)
        min_error = 999.0

        for c in range(max_c + 1):
            for f in range(max_f + 1):
                for s in range(max_s + 1):
                    for o in range(max_o + 1):
                        if c == 0 and f == 0 and s == 0 and o == 0: continue
                        calc_nominal = c * nominal_masses['C'] + f * nominal_masses['F'] + s * nominal_masses['S'] + o * \
                                       nominal_masses['O']

                        if calc_nominal == target_nominal or calc_nominal - 1 == target_nominal:
                            calc_exact = c * exact_masses['C'] + f * exact_masses['F'] + s * exact_masses['S'] + o * \
                                         exact_masses['O']
                            if calc_nominal - 1 == target_nominal:
                                calc_exact -= 1.007825
                            error = abs(calc_nominal - target_nominal)
                            if error < min_error:
                                min_error = error
                                best_exact_mass = calc_exact

        exact_mz_array.append(best_exact_mass)
    return np.array(exact_mz_array)


st.set_page_config(page_title="FluoroSpec Web Service", layout="wide", page_icon="🌊", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="collapsedControl"] {display: none;}
    .block-container { padding-top: 1rem !important; padding-bottom: 0rem !important; max-width: 100% !important; padding-left: 0 !important; padding-right: 0 !important; }
    .logo-text { font-size: 24px; font-weight: 900; color: #0056b3; font-family: 'Arial Black', sans-serif; padding-left: 30px; }
    .hero-banner { background: linear-gradient(135deg, #001233 0%, #002855 50%, #023e8a 100%); padding: 80px 0 120px 0; text-align: center; color: white; }
    .hero-title { font-size: 4.5rem; font-weight: 900; margin-bottom: 20px; letter-spacing: 2px; }
    div.stButton > button { background-color: #126782; color: white; width: 100%; height: 54px; border-radius: 4px; border: none; font-size: 18px; font-weight: bold; }
    div.stButton > button:hover { background-color: #023e8a; }
    div[data-testid="stRadio"] > div { display: flex; justify-content: flex-end; padding-right: 30px; }
    .custom-footer { background-color: #0d47a1; color: white; padding: 40px 60px; margin-top: 80px; display: flex; justify-content: space-between; }
    .footer-about { width: 65%; font-size: 14px; line-height: 1.6; }
    .footer-tools { width: 30%; text-align: right; font-size: 20px; font-weight: bold; }
    .icp-text { background-color: #0a357a; color: #a0aec0; text-align: center; padding: 10px; font-size: 12px; }
    .stProgress > div > div > div > div { background-color: #2a9d8f; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
col_logo, col_nav = st.columns([3.5, 6.5])
with col_logo: st.markdown(
    '<div class="logo-text">FluoroSpec<br><span style="font-size:12px; font-weight:normal; color:#666;">WEB SERVICE</span></div>',
    unsafe_allow_html=True)
with col_nav:
    page = st.radio("Navigation", ["Single Prediction", "Batch Prediction", "Interpretability", "User Guide"],
                    horizontal=True, label_visibility="collapsed")

model, model_status = load_model()

if page == "Single Prediction":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">FluoroSpec<br>Profiler</div>
        <div style="font-size: 1.2rem; color: #e9ecef;">Physics-Constrained MS/MS Prediction for Non-Targeted Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: -60px; position: relative; z-index: 10;'>", unsafe_allow_html=True)
    _, col_input, col_btn, _ = st.columns([2.5, 4, 1, 2.5])
    with col_input:
        smiles_input = st.text_input("SMILES",
                                     value="O=S(=O)(O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
                                     label_visibility="collapsed")
    with col_btn:
        submit_clicked = st.button("Submit")
    st.markdown("</div>", unsafe_allow_html=True)

    _, col_settings, _ = st.columns([2.5, 5, 2.5])
    with col_settings:
        with st.expander("⚙️ Advanced Physics Settings (Collision Energy & Sparsity)"):
            ce_value = st.slider("Collision Energy (eV)", 10.0, 100.0, 60.0, step=1.0)
            noise_cutoff = st.slider("Noise Threshold (%)", 0.0, 30.0, 2.0, step=1.0)

    st.markdown("<br>", unsafe_allow_html=True)
    _, col_main, _ = st.columns([1, 8, 1])
    with col_main:
        if submit_clicked:
            if not HAS_RDKIT:
                st.error("RDKit is not installed.")
            elif model is None:
                st.error(f"Model Error: {model_status}")
            else:
                with st.spinner(f"Initiating FluoroSpec Inference at {ce_value} eV..."):
                    try:
                        node_feats, adj_matrix, edge_features, atom_mask = smiles_to_graph_matrices(smiles_input)
                        if atom_mask.sum() == 0:
                            st.error("Invalid SMILES.")
                            st.stop()

                        meta_out = calculate_molecular_features(smiles_input, ce_value)
                        meta_features, exact_mass = meta_out[0], meta_out[1]

                        t_node = torch.tensor(node_feats, dtype=torch.float32).unsqueeze(0)
                        t_adj = torch.tensor(adj_matrix, dtype=torch.float32).unsqueeze(0)
                        t_edge = torch.tensor(edge_features, dtype=torch.float32).unsqueeze(0)
                        t_mask = torch.tensor(atom_mask, dtype=torch.float32).unsqueeze(0)
                        t_meta = torch.tensor(meta_features, dtype=torch.float32).unsqueeze(0)

                        with torch.no_grad():
                            final_spec, _, _, _ = model(t_node, t_adj, t_edge, t_mask, t_meta)
                            spectrum_array = final_spec.squeeze(0).numpy()

                        mz_bins = np.arange(50, 1050)
                        max_val = spectrum_array.max()

                        if max_val > 1e-5:
                            rel_abundance = (spectrum_array / max_val) * 100.0

                            # ====================================================================
                            # 🚀 顶刊级重构：彻底抛弃 find_peaks，采用直接布尔掩码截断！
                            # 只要丰度超过阈值，无条件保留，彻底杜绝相邻峰被算法“误杀”！
                            # ====================================================================
                            valid_indices = np.where(rel_abundance >= noise_cutoff)[0]

                            valid_mz_nominal = mz_bins[valid_indices]
                            valid_int = rel_abundance[valid_indices]

                            # NTA 标准过滤：提取强度最高的 15 个诊断碎片
                            if len(valid_mz_nominal) > 15:
                                top_indices = np.argsort(valid_int)[-15:]
                                valid_mz_nominal = valid_mz_nominal[top_indices]
                                valid_int = valid_int[top_indices]

                            # 为画图连线做极其重要的从小到大排序
                            sort_by_mz = np.argsort(valid_mz_nominal)
                            valid_mz_nominal = valid_mz_nominal[sort_by_mz]
                            valid_int = valid_int[sort_by_mz]

                            if len(valid_mz_nominal) > 0:
                                valid_mz_exact = rigorous_mass_annotation(valid_mz_nominal, smiles_input)

                                min_x = max(50, valid_mz_exact.min() - 30)
                                max_x = min(1050, valid_mz_exact.max() + 30)

                                fig = go.Figure()

                                stick_x = []
                                stick_y = []
                                for mx, my in zip(valid_mz_exact, valid_int):
                                    stick_x.extend([mx, mx, None])
                                    stick_y.extend([0, my, None])

                                fig.add_trace(go.Scatter(
                                    x=stick_x, y=stick_y,
                                    mode='lines',
                                    line=dict(color='#001233', width=2.5),
                                    hoverinfo='skip',
                                    showlegend=False
                                ))

                                fig.add_trace(go.Scatter(
                                    x=valid_mz_exact, y=valid_int,
                                    mode='markers',
                                    marker=dict(color='#001233', size=12, opacity=0),
                                    hovertemplate="<b>m/z:</b> %{x:.4f} Da<br><b>Abundance:</b> %{y:.2f}%<extra></extra>",
                                    showlegend=False
                                ))

                                fig.update_layout(
                                    title=dict(
                                        text=f"High-Resolution MS/MS Spectrum (Exact Mass: {exact_mass:.4f} Da | CE: {ce_value} eV)",
                                        font=dict(size=18, color='#001233')),
                                    xaxis_title="<b>m/z</b>", yaxis_title="<b>Relative Abundance (%)</b>",
                                    yaxis=dict(range=[0, 105], tick0=0, dtick=20, gridcolor='#e9ecef'),
                                    xaxis=dict(range=[min_x, max_x], gridcolor='#e9ecef'),
                                    plot_bgcolor='white', paper_bgcolor='white',
                                    margin=dict(l=40, r=40, t=60, b=40),
                                    hovermode="closest"
                                )
                                st.plotly_chart(fig, use_container_width=True)

                                mgf_text = f"BEGIN IONS\nPEPMASS={exact_mass:.4f}\nCHARGE=-1\nMSLEVEL=2\nNAME={smiles_input}\nSMILES={smiles_input}\n"
                                msp_text = f"Name: {smiles_input}\nPrecursorMZ: {exact_mass:.4f}\nPrecursor_type: [M-H]-\nIon_mode: N\nSMILES: {smiles_input}\nNum Peaks: {len(valid_mz_exact)}\n"
                                txt_text = "m/z\tRelative_Abundance(%)\n"

                                for m, i in zip(valid_mz_exact, valid_int):
                                    mgf_text += f"{m:.4f} {i:.2f}\n"
                                    msp_text += f"{m:.4f} {i:.2f}\n"
                                    txt_text += f"{m:.4f}\t{i:.2f}\n"

                                mgf_text += "END IONS\n"
                                msp_text += "\n"

                                dl_col1, dl_col2, dl_col3 = st.columns(3)
                                with dl_col1:
                                    st.download_button("📥 Download .MGF", data=mgf_text, file_name="FluoroSpec.mgf",
                                                       use_container_width=True)
                                with dl_col2:
                                    st.download_button("📥 Download .MSP", data=msp_text, file_name="FluoroSpec.msp",
                                                       use_container_width=True)
                                with dl_col3:
                                    st.download_button("📥 Download .TXT", data=txt_text,
                                                       file_name="FluoroSpec_plot_data.txt", use_container_width=True)
                            else:
                                st.warning("No fragments passed the noise threshold.")
                        else:
                            st.warning("No fragments generated at these settings.")
                    except Exception as e:
                        st.error(f"Error processing molecule: {str(e)}")

elif page == "Batch Prediction":
    _, col_batch, _ = st.columns([1, 8, 1])
    with col_batch:
        st.markdown("<br><h2>📦 High-Throughput Batch Prediction</h2><hr>", unsafe_allow_html=True)
        st.info(
            "Upload a CSV file containing a **SMILES** column to generate screening libraries at scale. Optional: Include a **CE** column to set specific collision energies.")

        csv_template = pd.DataFrame({"SMILES": ["O=S(=O)(O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
                                                "O=C(O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"],
                                     "CE": [60.0, 60.0]})
        st.download_button("📥 Download Example CSV Template", data=csv_template.to_csv(index=False),
                           file_name="template.csv", mime="text/csv")
        st.markdown("<br>", unsafe_allow_html=True)

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            batch_ce = st.slider("Global Collision Energy (Fallback)", 10.0, 100.0, 60.0, step=1.0, key="b_ce")
        with col_b2:
            batch_noise = st.slider("Noise Threshold (%)", 0.0, 30.0, 2.0, step=1.0, key="b_noise")

        uploaded_file = st.file_uploader("Upload your .csv file", type=["csv"])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if 'SMILES' not in df.columns:
                st.error("❌ The uploaded CSV MUST contain a column named 'SMILES'.")
            else:
                st.success(f"✅ File loaded successfully! Found {len(df)} molecules.")
                if st.button("🚀 Start Batch Generation", type="primary"):
                    if model is None:
                        st.error("Model is not loaded.")
                    else:
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        mgf_output, msp_output, txt_output = "", "", "SMILES\tCE(eV)\tm/z\tRelative_Abundance(%)\n"
                        success_count = 0

                        for idx, row in df.iterrows():
                            smiles = str(row['SMILES']).strip()
                            ce = float(row['CE']) if 'CE' in df.columns else batch_ce
                            try:
                                node_feats, adj_matrix, edge_features, atom_mask = smiles_to_graph_matrices(smiles)
                                if atom_mask.sum() == 0: continue
                                meta_out = calculate_molecular_features(smiles, ce)
                                meta_features, exact_mass = meta_out[0], meta_out[1]

                                t_node = torch.tensor(node_feats, dtype=torch.float32).unsqueeze(0)
                                t_adj = torch.tensor(adj_matrix, dtype=torch.float32).unsqueeze(0)
                                t_edge = torch.tensor(edge_features, dtype=torch.float32).unsqueeze(0)
                                t_mask = torch.tensor(atom_mask, dtype=torch.float32).unsqueeze(0)
                                t_meta = torch.tensor(meta_features, dtype=torch.float32).unsqueeze(0)

                                with torch.no_grad():
                                    final_spec, _, _, _ = model(t_node, t_adj, t_edge, t_mask, t_meta)
                                    spectrum_array = final_spec.squeeze(0).numpy()

                                mz_bins = np.arange(50, 1050)
                                max_val = spectrum_array.max()

                                if max_val > 1e-5:
                                    rel_abundance = (spectrum_array / max_val) * 100.0

                                    # 🚀 批量预测也同步采用：布尔掩码截断！彻底防止漏峰
                                    valid_indices = np.where(rel_abundance >= batch_noise)[0]
                                    valid_mz_nominal = mz_bins[valid_indices]
                                    valid_int = rel_abundance[valid_indices]

                                    if len(valid_mz_nominal) > 15:
                                        top_indices = np.argsort(valid_int)[-15:]
                                        valid_mz_nominal = valid_mz_nominal[top_indices]
                                        valid_int = valid_int[top_indices]

                                    sort_by_mz = np.argsort(valid_mz_nominal)
                                    valid_mz_nominal = valid_mz_nominal[sort_by_mz]
                                    valid_int = valid_int[sort_by_mz]

                                    if len(valid_mz_nominal) > 0:
                                        valid_mz_exact = rigorous_mass_annotation(valid_mz_nominal, smiles)

                                        mgf_output += f"BEGIN IONS\nPEPMASS={exact_mass:.4f}\nCHARGE=-1\nMSLEVEL=2\nNAME={smiles} | CE={ce}eV\nSMILES={smiles}\n"
                                        msp_output += f"Name: {smiles} | CE={ce}eV\nPrecursorMZ: {exact_mass:.4f}\nPrecursor_type: [M-H]-\nIon_mode: N\nSMILES: {smiles}\nNum Peaks: {len(valid_mz_exact)}\n"

                                        for m, int_val in zip(valid_mz_exact, valid_int):
                                            mgf_output += f"{m:.4f} {int_val:.2f}\n"
                                            msp_output += f"{m:.4f} {int_val:.2f}\n"
                                            txt_output += f"{smiles}\t{ce}\t{m:.4f}\t{int_val:.2f}\n"

                                        mgf_output += "END IONS\n\n"
                                        msp_output += "\n"
                                        success_count += 1
                            except Exception:
                                pass
                            progress_bar.progress((idx + 1) / len(df))
                            status_text.markdown(f"**Processing:** {idx + 1} / {len(df)} ...")

                        if success_count > 0:
                            status_text.markdown(
                                f"✅ **Batch generation complete! Successfully processed {success_count} molecules.**")
                            dl_b1, dl_b2, dl_b3 = st.columns(3)
                            with dl_b1:
                                st.download_button("📥 Batch .MGF", data=mgf_output, file_name="FluoroSpec_Batch.mgf",
                                                   mime="text/plain", use_container_width=True)
                            with dl_b2:
                                st.download_button("📥 Batch .MSP", data=msp_output, file_name="FluoroSpec_Batch.msp",
                                                   mime="text/plain", use_container_width=True)
                            with dl_b3:
                                st.download_button("📥 Batch .TXT", data=txt_output, file_name="FluoroSpec_Batch.txt",
                                                   mime="text/plain", use_container_width=True)
                        else:
                            st.error("No valid spectra were generated. Try lowering the Noise Threshold.")

elif page == "Interpretability":
    _, col_kan, _ = st.columns([1, 8, 1])
    with col_kan:
        st.markdown("<br><h2>🧠 Network Interpretability (B-Spline & Pruning)</h2><hr>", unsafe_allow_html=True)
        st.success(
            "Unlike traditional black-box MLPs, FluoroSpec utilizes edge-aware Kolmogorov-Arnold Networks (KAN) to model the exact physical response of collision energy.")

elif page == "User Guide":
    _, col_guide, _ = st.columns([1, 8, 1])
    with col_guide:
        st.markdown("<br><h2>📖 User Guide & Algorithmic Background</h2><hr>", unsafe_allow_html=True)
        st.markdown("""
        ### 1. Introduction to FluoroSpec
        This web server provides high-throughput MS/MS spectral prediction specifically optimized for **identifying PFAS in water bodies**. 
        It is powered by a novel architecture that integrates **Edge-Aware Graph Neural Networks (GNN)** with **Kolmogorov-Arnold Networks (KAN)**, ensuring predictions are governed by exact physical and chemical rules rather than black-box pattern matching.

        ### 2. Input Specifications
        * **Format**: The core input must be a valid canonical **SMILES** string.
        * **Size Constraints**: The molecule should not exceed **64 heavy atoms**.
        * **Ionization Mode**: The model is exclusively trained for Negative Electrospray Ionization (**ESI-**) yielding `[M-H]-` precursor ions.

        ### 3. Spectral Processing & HRMS Annotation
        To bridge the gap between machine learning latent spaces and rigorous Analytical Chemistry, FluoroSpec applies a two-step post-processing pipeline:

        1. **Direct Boolean Thresholding**: The raw profile-mode network outputs are processed using exact intensity masking. This enforces standard MS analytical stick spectra and preserves true adjacent chemical fragments that signal processing algorithms might inadvertently suppress.
        2. **Combinatorial Exact Mass Annotation**: The model operates at unit mass resolution to prevent the *curse of dimensionality*. However, the final displayed $m/z$ values are not mere integers. We employ an in-silico combinatorial algorithm that matches the predicted nominal masses against the precursor's actual elemental formula (C, F, O, S). This allows the server to project theoretical **Exact Monoisotopic Masses** (up to 4 decimal places), perfectly reflecting phenomena like Fluorine mass defects.

        ### 4. Output Formats (Top-Tier Standard)
        For both single and batch predictions, FluoroSpec generates three standard formats:
        * **.MGF Format**: Mascot Generic Format, the gold standard for molecular networking (GNPS) and spectral matching.
        * **.MSP Format**: NIST text format, universally compatible with MoNA, MassBank, and MS-DIAL.
        * **.TXT Format**: A tab-separated XY coordinate list, engineered specifically for chemists to plot high-quality vector graphics in **OriginLab**, GraphPad Prism, or MS Excel.
        """)

st.markdown("""
<div class="custom-footer">
    <div class="footer-about">
        <h3 style="margin-top:0;">ABOUT</h3>
        This website is designed for non-targeted screening (NTA) and identification of PFAS in water bodies. All features are powered by Kolmogorov-Arnold Networks (KAN) and are free to use.<br><br>
        <b>When citing FluoroSpec Profiler please use the following:</b><br>
        <i>[Your Name], et al. Machine learning-based models with high accuracy and broad applicability domains for screening PFAS substances in complex water matrix. Environ Sci Technol, 2026. (Under Review)</i>
    </div>
    <div class="footer-tools">
        DEV TOOL<br>
        <span style="font-size:14px; font-weight:normal;">Powered by Python, PyTorch & Streamlit</span><br>
        <div style="font-size:40px; margin-top:10px;">🐍 🔥 👑</div>
    </div>
</div>
<div class="icp-text">ICP Numbers: 202608201503 · FluoroSpec Web Service © 2026</div>
""", unsafe_allow_html=True)