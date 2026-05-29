import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
import networkx.algorithms.community as nx_comm
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
 
# ─────────────────────────────────────────────────────────────
# CONFIG PAGE
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Graphe de Similarité — Patients Diabétiques",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0a0e1a; color: #e2e8f0; }
[data-testid="stSidebar"] { background: #0f1421; border-right: 1px solid #1e2535; }
[data-testid="stSidebar"] * { color: #c8d3e0 !important; }
.main-title { font-family: 'Space Mono', monospace; font-size: 2rem; font-weight: 700;
    background: linear-gradient(135deg, #38bdf8, #6366f1, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; letter-spacing: -1px; margin-bottom: 0.2rem; }
.sub-title { color: #64748b; font-size: 0.9rem; font-family: 'Space Mono', monospace;
    letter-spacing: 2px; text-transform: uppercase; margin-bottom: 1.5rem; }
.metric-card { background: linear-gradient(135deg, #111827, #1a2235);
    border: 1px solid #1e3a5f; border-radius: 12px; padding: 1.2rem 1.5rem;
    text-align: center; transition: transform 0.2s, border-color 0.2s; }
.metric-card:hover { transform: translateY(-2px); border-color: #38bdf8; }
.metric-val { font-family: 'Space Mono', monospace; font-size: 2rem; font-weight: 700; color: #38bdf8; }
.metric-label { font-size: 0.75rem; color: #64748b; letter-spacing: 1px;
    text-transform: uppercase; margin-top: 0.3rem; }
.stTabs [data-baseweb="tab-list"] { background: #0f1421; border-radius: 10px; padding: 4px 8px; gap: 16px; }
.stTabs [data-baseweb="tab"] { background: transparent; border-radius: 8px; color: #64748b;
    font-family: 'Space Mono', monospace; font-size: 0.8rem; padding: 8px 18px; }
.stTabs [aria-selected="true"] { background: #1e2d4a !important; color: #38bdf8 !important; }
.stButton > button { background: linear-gradient(135deg, #1e3a5f, #1e2d4a);
    border: 1px solid #38bdf8; border-radius: 8px; color: #38bdf8;
    font-family: 'Space Mono', monospace; font-size: 0.8rem; letter-spacing: 1px; transition: all 0.2s; }
.stButton > button:hover { background: linear-gradient(135deg, #38bdf8, #6366f1);
    color: #0a0e1a; border-color: transparent; }
.section-divider { height: 1px;
    background: linear-gradient(90deg, transparent, #1e3a5f, transparent); margin: 1.5rem 0; }
.info-box { background: #0f1e3a; border-left: 3px solid #38bdf8; border-radius: 0 8px 8px 0;
    padding: 0.8rem 1.2rem; margin: 0.5rem 0; font-size: 0.9rem; color: #94a3b8; }
.warn-box { background: #1a1200; border-left: 3px solid #E09020; border-radius: 0 8px 8px 0;
    padding: 0.8rem 1.2rem; margin: 0.5rem 0; font-size: 0.9rem; color: #94a3b8; }
.danger-box { background: #1a0a0a; border-left: 3px solid #E24B4A; border-radius: 0 8px 8px 0;
    padding: 0.8rem 1.2rem; margin: 0.5rem 0; font-size: 0.9rem; color: #94a3b8; }
.success-box { background: #0a1a12; border-left: 3px solid #34d399; border-radius: 0 8px 8px 0;
    padding: 0.8rem 1.2rem; margin: 0.5rem 0; font-size: 0.9rem; color: #94a3b8; }
.result-box { background: linear-gradient(135deg, #0f1e3a, #1a1230);
    border: 1px solid #6366f1; border-radius: 12px; padding: 1.5rem; margin: 1rem 0; }
h1,h2,h3,h4 { color: #e2e8f0; }
p { color: #94a3b8; }
</style>
""", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
COLS = ['Age', 'BMI', 'FastingBloodSugar', 'HbA1c',
        'SystolicBP', 'CholesterolTotal', 'SleepQuality']
 
POIDS = np.array([0.10, 0.15, 0.25, 0.25, 0.15, 0.05, 0.05])
 
# ── Mapping visuel basé sur le label textuel (pas sur l'ID) ──────────────────
def get_base_label(profil: str) -> str:
    """Extrait le label principal (avant ' — ')."""
    return profil.split(" — ")[0]
 
PALETTE_PROFIL = {
    "Diabétique sévère":       "#E24B4A",
    "Risque modéré":           "#E09020",
    "Hyperglycémie isolée":    "#38bdf8",
    "Profil sain":             "#34d399",
}
 
ICONS_PROFIL = {
    "Diabétique sévère":       "🔴",
    "Risque modéré":           "🟠",
    "Hyperglycémie isolée":    "🔵",
    "Profil sain":             "🟢",
}
 
def profil_color(profil: str) -> str:
    return PALETTE_PROFIL.get(get_base_label(profil), "#888888")
 
def profil_icon(profil: str) -> str:
    return ICONS_PROFIL.get(get_base_label(profil), "❓")
 
def profil_box_class(profil: str) -> str:
    base = get_base_label(profil)
    if "sévère" in base:
        return "danger-box"
    if "modéré" in base or "isolée" in base:
        return "warn-box"
    return "success-box"
 

import os, io
 
@st.cache_data
def load_data(raw_bytes: bytes, norm_bytes: bytes):
    df_raw  = pd.read_csv(io.BytesIO(raw_bytes))
    df_norm = pd.read_csv(io.BytesIO(norm_bytes))
    return df_raw, df_norm
 
def find_local_csv(name):
    candidates = [
        name,
        os.path.join(os.path.dirname(__file__), name),
        os.path.join(os.path.dirname(__file__), "..", "data", name),
        os.path.join(os.path.dirname(__file__), "data", name),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None
 
def get_data_bytes():
    if "raw_bytes" in st.session_state and "norm_bytes" in st.session_state:
        return st.session_state["raw_bytes"], st.session_state["norm_bytes"]
    path_clean = find_local_csv("diabetes_data_cleaned.csv")
    path_norm  = find_local_csv("diabetes_data_normalized.csv")
    if path_clean and path_norm:
        with open(path_clean, "rb") as f:
            raw_b = f.read()
        with open(path_norm, "rb") as f:
            norm_b = f.read()
        st.session_state["raw_bytes"]  = raw_b
        st.session_state["norm_bytes"] = norm_b
        return raw_b, norm_b
    return None, None
 
@st.cache_data
def compute_similarity(norm_bytes: bytes, threshold_pct=85):
    df_norm = pd.read_csv(io.BytesIO(norm_bytes))
    X = df_norm[COLS].values
    n = len(X)
    sqrt_w = np.sqrt(POIDS)
    X_w = X * sqrt_w
    BLOC = 300
    sim = np.zeros((n, n), dtype=np.float32)
    for i in range(0, n, BLOC):
        for j in range(i, n, BLOC):
            xi = X_w[i:i+BLOC]
            xj = X_w[j:j+BLOC]
            diff = xi[:, None, :] - xj[None, :, :]
            d = np.sqrt((diff**2).sum(axis=2))
            s = 1.0 / (1.0 + d)
            sim[i:i+BLOC, j:j+BLOC] = s
            if i != j:
                sim[j:j+BLOC, i:i+BLOC] = s.T
    idx_upper = np.triu_indices(n, k=1)
    seuil = float(np.percentile(sim[idx_upper], threshold_pct))
    return sim, seuil
 
@st.cache_data
def build_graph(norm_bytes: bytes, threshold_pct=85):
    sim, seuil = compute_similarity(norm_bytes, threshold_pct)
    n = len(sim)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    rows, cols_ = np.where(sim >= seuil)
    mask = rows < cols_
    G.add_edges_from(zip(rows[mask].tolist(), cols_[mask].tolist()))
    return G
 
@st.cache_data
def run_louvain(norm_bytes: bytes, threshold_pct=85):
    G = build_graph(norm_bytes, threshold_pct)
    communities = nx_comm.louvain_communities(G, seed=42, resolution=1.05)
    partition = {}
    for i, comm in enumerate(communities):
        for node in comm:
            partition[node] = i
    Q = nx_comm.modularity(G, communities)
    return partition, Q, len(communities), G
 
@st.cache_data
def get_community_profiles(raw_bytes: bytes, norm_bytes: bytes, threshold_pct=85):
    """Calcule les profils dynamiquement — label basé sur taux + fbs, pas sur l'ID."""
    df_raw = pd.read_csv(io.BytesIO(raw_bytes))
    partition, Q, n_comm, _ = run_louvain(norm_bytes, threshold_pct)
    df = df_raw.copy()
    df['communaute'] = df.index.map(lambda i: partition.get(i, -1))
 
    profiles = []
    for c in sorted(df['communaute'].unique()):
        sub  = df[df['communaute'] == c]
        taux = sub['Diagnosis'].mean()
        fbs  = sub['FastingBloodSugar'].mean()
 
        # Label principal basé sur le taux
        if taux >= 0.7:
            profil = "Diabétique sévère"
        elif taux >= 0.4:
            profil = "Risque modéré"
        elif taux >= 0.2:
            if fbs >= 110:
                profil = "Hyperglycémie isolée"
            else:
                profil = "Risque modéré"
        else:
            profil = "Profil sain"
 
        row = {
            'communaute':   c,
            'taille':       len(sub),
            'taux_diabete': round(taux * 100, 1),
            'profil':       profil,
        }
        for col in COLS:
            row[f'moy_{col}'] = round(sub[col].mean(), 2)
        profiles.append(row)
 
    return pd.DataFrame(profiles), df
 
# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family:Space Mono,monospace; font-size:1.1rem; font-weight:700;
         background:linear-gradient(135deg,#38bdf8,#a78bfa);
         -webkit-background-clip:text; -webkit-text-fill-color:transparent;
         margin-bottom:0.5rem;'>
    🧬 DiabGraphe
    </div>
    <div style='font-size:0.7rem; color:#475569; letter-spacing:1.5px;
         text-transform:uppercase; margin-bottom:1.5rem;'>
    Analyse de réseau clinique
    </div>
    """, unsafe_allow_html=True)
 
    st.markdown("### ⚙️ Paramètres du graphe")
    threshold_pct = st.slider("Seuil de similarité (percentile)",
                               min_value=60, max_value=95, value=75, step=5,
                               help="Détermine quelles connexions entre patients sont conservées. P75 = top 25% des similarités.")
    st.markdown("---")
 
    # Interprétation dynamique
    if threshold_pct < 75:
        badge = "Exploratoire"
        badge_color = "#E09020"
        box_class = "warn-box"
        interpretation = "Trop de patients connectés — les profils perdent en précision"
 
    elif threshold_pct <= 90:
        badge = "Recommandé"
        badge_color = "#34d399"
        box_class = "success-box"
        if threshold_pct == 75:
            interpretation = "P75 recommandé — bon compromis entre connexions et structure des communautés"
        else:
            interpretation = "Bon équilibre — communautés stables et interprétables"
 
    else:
        badge = "Exploratoire"
        badge_color = "#E09020"
        box_class = "warn-box"
        interpretation = "Graphe trop fragmenté — petits groupes, moins représentatifs"
 
    st.markdown(f"""
    <div class='{box_class}'>
        <span style='font-family:Space Mono,monospace; font-size:0.8rem;
             color:{badge_color}; font-weight:700;'>{badge}</span><br>
        <span style='font-size:0.8rem;'>{interpretation}</span>
    </div>
    """, unsafe_allow_html=True)
 

raw_bytes, norm_bytes = get_data_bytes()
 
if raw_bytes is None or norm_bytes is None:
    st.markdown("<div class='main-title'>🧬 DiabGraph</div>", unsafe_allow_html=True)
    st.markdown("<div class='info-box'>📂 <b>Chargez vos deux fichiers CSV</b> pour démarrer.</div>",
                unsafe_allow_html=True)
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        f_clean = st.file_uploader("diabetes_data_cleaned.csv", type="csv", key="up_clean")
    with col_u2:
        f_norm = st.file_uploader("diabetes_data_normalized.csv", type="csv", key="up_norm")
    if f_clean and f_norm:
        st.session_state["raw_bytes"]  = f_clean.read()
        st.session_state["norm_bytes"] = f_norm.read()
        st.rerun()
    else:
        st.stop()
 
df_raw, df_norm = load_data(raw_bytes, norm_bytes)
 
# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class='main-title'>Graphe de Similarité — Patients Diabétiques</div>
<div class='sub-title'>Analyse de réseau · Détection de communautés · Classification</div>
""", unsafe_allow_html=True)
 
# Calcul Q dynamique pour l'afficher dans les métriques
partition_hdr, Q_hdr, n_comm_hdr, _ = run_louvain(norm_bytes, threshold_pct)
 
c1, c2, c3, c4, c5 = st.columns(5)
n_patients = len(df_raw)
n_diabete  = df_raw['Diagnosis'].sum()
pct_diab   = round(n_diabete / n_patients * 100, 1)
 
for col, val, label in zip(
    [c1, c2, c3, c4, c5],
    [n_patients, int(n_diabete), f"{pct_diab}%", n_comm_hdr, f"{Q_hdr:.3f}"],
    ["Patients", "Diabétiques", "Taux diabète", "Communautés", "Modularité Q"]
):
    col.markdown(f"""
    <div class='metric-card'>
        <div class='metric-val'>{val}</div>
        <div class='metric-label'>{label}</div>
    </div>
    """, unsafe_allow_html=True)
 
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Exploration", " Graphe", " Communautés", " Similarité", " Classifier"
])
 
# ══════════════════════════════════════════════════════════════
# TAB 1 — EXPLORATION
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## Exploration des données cliniques")
    col_l, col_r = st.columns([1, 1])
 
    with col_l:
        st.markdown("### Distribution des variables")
        var_choice = st.selectbox("Variable", COLS, key="expl_var")
        fig = go.Figure()
        for diag, name, color in [(0, "Non-Diabétique", "#38bdf8"), (1, "Diabétique", "#E24B4A")]:
            vals = df_raw[df_raw['Diagnosis'] == diag][var_choice]
            fig.add_trace(go.Histogram(x=vals, name=name, nbinsx=40,
                marker_color=color, opacity=0.7, histnorm='probability density'))
        fig.update_layout(barmode='overlay', height=350,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,20,35,0.8)',
            font_color='#94a3b8', legend=dict(font_color='#e2e8f0'),
            xaxis=dict(gridcolor='#1e2535', title=var_choice),
            yaxis=dict(gridcolor='#1e2535', title='Densité'),
            margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
 
    with col_r:
        st.markdown("### Matrice de corrélation")
        corr = df_raw[COLS + ['Diagnosis']].corr()
        fig_corr = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale='RdBu', zmid=0, zmin=-1, zmax=1,
            text=np.round(corr.values, 2), texttemplate="%{text}",
            textfont_size=9, colorbar=dict(tickfont_color='#94a3b8')))
        fig_corr.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,20,35,0.8)', font_color='#94a3b8',
            xaxis=dict(tickangle=-30), margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_corr, use_container_width=True)
 
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("### Analyse de variance (ANOVA) par diagnostic")
 
    diag0 = df_raw[df_raw['Diagnosis'] == 0]
    diag1 = df_raw[df_raw['Diagnosis'] == 1]
    anova_data = []
    for col in COLS:
        f, p = stats.f_oneway(diag0[col], diag1[col])
        anova_data.append({
            'Variable': col,
            'Moy. Non-Diab.': round(diag0[col].mean(), 2),
            'Moy. Diab.': round(diag1[col].mean(), 2),
            'Différence': round(diag1[col].mean() - diag0[col].mean(), 2),
            'F-stat': round(f, 1), 'p-value': round(p, 5),
            'Poids (%)': f"{int(POIDS[COLS.index(col)]*100)}%",
            'Significatif': 'oui' if p < 0.05 else 'non'
        })
    st.dataframe(pd.DataFrame(anova_data), use_container_width=True, hide_index=True)
 
    st.markdown("### Corrélation avec le diagnostic")
    corr_diag = df_raw[COLS].corrwith(df_raw['Diagnosis']).abs().sort_values(ascending=True)
    fig_bar = go.Figure(go.Bar(
        x=corr_diag.values, y=corr_diag.index, orientation='h',
        marker=dict(color=corr_diag.values,
            colorscale=[[0, '#1e3a5f'], [0.5, '#38bdf8'], [1, '#6366f1']], showscale=False),
        text=[f"{v:.3f}" for v in corr_diag.values],
        textposition='outside', textfont_color='#94a3b8'))
    fig_bar.update_layout(height=280, paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,20,35,0.5)', font_color='#94a3b8',
        xaxis=dict(gridcolor='#1e2535', range=[0, 0.6], title='|Corrélation|'),
        yaxis=dict(gridcolor='#1e2535'), margin=dict(l=10, r=80, t=10, b=10))
    st.plotly_chart(fig_bar, use_container_width=True)
 
# ══════════════════════════════════════════════════════════════
# TAB 2 — GRAPHE
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## Graphe de similarité")
    col_a, col_b = st.columns([3, 1])
 
    with col_b:
        n_nodes_show = st.slider("Nœuds à afficher", 100, 1879, 400, 50)
        show_edges   = st.checkbox("Afficher les arêtes", value=True)
        st.markdown("<div class='info-box' style='font-size:0.75rem;'>Couleur = communauté Louvain</div>",
                    unsafe_allow_html=True)
 
    with col_a:
        with st.spinner("Construction du graphe..."):
            partition, Q, n_comm, G = run_louvain(norm_bytes, threshold_pct)
        st.markdown(f"""
        <div class='info-box'>
        Graphe P{threshold_pct} : <b>{G.number_of_nodes():,}</b> nœuds ·
        <b>{G.number_of_edges():,}</b> arêtes · densité <b>{nx.density(G):.4f}</b> ·
        Louvain Q = <b>{Q:.4f}</b>
        </div>""", unsafe_allow_html=True)
 
    with st.spinner("Calcul du layout (spring layout — topologie réelle du graphe)..."):
        pos    = nx.spring_layout(G, seed=42, k=0.3, iterations=50)
        coords = np.zeros((len(df_raw), 2))
        for node, (x, y) in pos.items():
            coords[node] = [x, y]
 
        np.random.seed(42)
        df_part_temp = pd.DataFrame({
            'idx':  range(len(df_raw)),
            'comm': [partition.get(i, -1) for i in range(len(df_raw))],
        })
        sample_idx = []
        for c in df_part_temp['comm'].unique():
            members = df_part_temp[df_part_temp['comm'] == c]['idx'].tolist()
            n_pick  = max(5, round(n_nodes_show * len(members) / len(df_raw)))
            picked  = np.random.choice(members, min(n_pick, len(members)), replace=False)
            sample_idx.extend(picked.tolist())
        sample_idx    = list(set(sample_idx))[:n_nodes_show]
        coords_sample = coords[sample_idx]
        comms_sample  = [partition.get(i, -1) for i in sample_idx]
 
    # Profils pour récupérer les labels dynamiques
    profiles_df_g, _ = get_community_profiles(raw_bytes, norm_bytes, threshold_pct)
    profil_map = dict(zip(profiles_df_g['communaute'], profiles_df_g['profil']))
 
    fig_graph = go.Figure()
 
    if show_edges:
        sim_mat, _ = compute_similarity(norm_bytes, threshold_pct)
        all_weights = [(sim_mat[u, v], u, v)
                       for u in sample_idx for v in sample_idx
                       if u < v and G.has_edge(u, v)]
        if all_weights:
            p75_w   = np.percentile([w for w, _, _ in all_weights], 75)
            idx_map = {node: i for i, node in enumerate(sample_idx)}
            edge_x, edge_y = [], []
            for w, u, v in all_weights:
                if w >= p75_w:
                    ui = idx_map[u]
                    vi = idx_map[v]
                    edge_x += [coords_sample[ui, 0], coords_sample[vi, 0], None]
                    edge_y += [coords_sample[ui, 1], coords_sample[vi, 1], None]
            fig_graph.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines', line=dict(width=0.3, color='rgba(100,120,160,0.15)'),
                hoverinfo='skip', showlegend=False))
 
    # Coloriage par communauté uniquement
    for c in sorted(set(comms_sample)):
        mask   = [i for i, cm in enumerate(comms_sample) if cm == c]
        profil = profil_map.get(c, f"Comm {c}")
        color  = profil_color(profil)
        icon   = profil_icon(profil)
        if not mask:
            continue
        fig_graph.add_trace(go.Scatter(
            x=coords_sample[mask, 0], y=coords_sample[mask, 1],
            mode='markers',
            name=f"{icon} Comm {c} — {profil}",
            marker=dict(size=6, color=color, opacity=0.85, line=dict(width=0)),
            hovertemplate=f"<b>Comm {c}</b><br>{profil}<extra></extra>"
        ))
 
    fig_graph.update_layout(height=520, paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(10,14,26,0.95)',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(font_color='#e2e8f0', bgcolor='rgba(15,20,35,0.9)',
                   bordercolor='#1e2535', borderwidth=1),
        font_color='#94a3b8', margin=dict(l=10, r=10, t=10, b=10),
        hoverlabel=dict(bgcolor='#1a2235', font_color='#e2e8f0'))
    st.plotly_chart(fig_graph, use_container_width=True)
 
    st.markdown(f"""
    <div class='info-box'>
    <b>Spring layout</b> — Les nœuds connectés sont physiquement proches.
    Les clusters visibles correspondent aux communautés Louvain détectées
    sur le graphe de similarité (seuil P{threshold_pct}).
    </div>""", unsafe_allow_html=True)
 
# ══════════════════════════════════════════════════════════════
# TAB 3 — COMMUNAUTÉS
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## Profils des communautés Louvain")
 
    with st.spinner("Calcul des profils..."):
        profiles_df, df_with_comm = get_community_profiles(raw_bytes, norm_bytes, threshold_pct)
 
    # Tri par taux décroissant pour l'affichage
    order = profiles_df.sort_values('taux_diabete', ascending=False)['communaute'].tolist()
 
    # Cartes dynamiques
    cols_badges = st.columns(len(order))
    for i, c in enumerate(order):
        row    = profiles_df[profiles_df['communaute'] == c].iloc[0]
        profil = row['profil']
        color  = profil_color(profil)
        icon   = profil_icon(profil)
        taux   = row['taux_diabete']
        taux_color = "#E24B4A" if taux > 50 else "#E09020" if taux > 20 else "#34d399"
 
        with cols_badges[i]:
            st.markdown(f"""
            <div class='metric-card' style='border-color:{color};'>
                <div style='font-size:1.4rem; margin-bottom:0.4rem;'>{icon}</div>
                <div style='font-family:Space Mono,monospace; font-size:0.7rem;
                     color:{color}; margin-bottom:0.5rem;'>COMM {c}</div>
                <div style='font-size:0.72rem; color:#94a3b8; margin-bottom:0.8rem;
                     line-height:1.4;'>{profil}</div>
                <div style='font-family:Space Mono,monospace; font-size:1.5rem;
                     color:{color};'>{int(row["taille"])}</div>
                <div style='font-size:0.65rem; color:#475569;'>patients</div>
                <div style='font-family:Space Mono,monospace; font-size:1rem;
                     color:{taux_color}; margin-top:0.5rem;'>{taux}% diab.</div>
            </div>""", unsafe_allow_html=True)
 
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
 
    col_h, col_r = st.columns([2, 1])
 
    with col_h:
        st.markdown("### Heatmap des profils cliniques")
        heat_vars  = ['moy_FastingBloodSugar', 'moy_HbA1c', 'moy_BMI', 'moy_Age',
                      'moy_SystolicBP', 'moy_CholesterolTotal', 'moy_SleepQuality']
        var_labels = ['Glycémie', 'HbA1c', 'IMC', 'Âge', 'PA syst.', 'Cholestérol', 'Sommeil']
 
        prof_ordered = profiles_df.set_index('communaute').loc[order]
        matrix = prof_ordered[heat_vars].values.astype(float)
        matrix_norm = np.zeros_like(matrix)
        for j in range(matrix.shape[1]):
            col_vals = matrix[:, j]
            mn, mx   = col_vals.min(), col_vals.max()
            matrix_norm[:, j] = (col_vals - mn) / (mx - mn) if mx > mn else 0.5
 
        text_mat = [[f"{matrix[i,j]:.1f}" for j in range(len(heat_vars))] for i in range(len(order))]
        y_labels = [f"Comm {c} — {profiles_df[profiles_df['communaute']==c].iloc[0]['profil'][:25]}..."
                    if len(profiles_df[profiles_df['communaute']==c].iloc[0]['profil']) > 25
                    else f"Comm {c} — {profiles_df[profiles_df['communaute']==c].iloc[0]['profil']}"
                    for c in order]
 
        fig_heat = go.Figure(go.Heatmap(
            z=matrix_norm, x=var_labels, y=y_labels,
            colorscale='RdYlBu_r',
            text=text_mat, texttemplate="%{text}", textfont_size=11,
            colorbar=dict(tickfont=dict(color='#94a3b8'),
                title=dict(text='Intensité relative', font=dict(color='#94a3b8')))))
        fig_heat.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(10,14,26,0.9)', font_color='#94a3b8',
            xaxis=dict(tickangle=-20), margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_heat, use_container_width=True)
 
    with col_r:
        st.markdown("### Taux de diabète par comm.")
        fig_bar2 = go.Figure()
        for c in order:
            row    = profiles_df[profiles_df['communaute'] == c].iloc[0]
            profil = row['profil']
            fig_bar2.add_trace(go.Bar(
                y=[f"Comm {c}"], x=[row["taux_diabete"]],
                orientation='h', name=f"Comm {c}",
                marker_color=profil_color(profil),
                text=f"{row['taux_diabete']}%",
                textposition='outside', textfont_color='#94a3b8'))
        fig_bar2.update_layout(height=300, barmode='group',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,20,35,0.5)',
            font_color='#94a3b8', showlegend=False,
            xaxis=dict(gridcolor='#1e2535', range=[0, 120], title='% diabétiques'),
            yaxis=dict(gridcolor='#1e2535'), margin=dict(l=10, r=60, t=10, b=10))
        st.plotly_chart(fig_bar2, use_container_width=True)
 
    
    st.markdown("### Radar — Profils cliniques comparés")
    radar_vars   = ['FastingBloodSugar', 'HbA1c', 'BMI', 'Age', 'SystolicBP']
    radar_labels = ['Glycémie', 'HbA1c', 'IMC', 'Âge', 'PA syst.']
 
    fig_radar = go.Figure()
    for c in order:
        row        = profiles_df[profiles_df['communaute'] == c].iloc[0]
        profil     = row['profil']
        vals_radar = [row[f'moy_{v}'] for v in radar_vars]
        all_vals   = [profiles_df[f'moy_{v}'].values for v in radar_vars]
        vals_norm  = [(v - mn) / (mx - mn) if mx > mn else 0.5
                      for v, mn, mx in zip(vals_radar,
                                           [a.min() for a in all_vals],
                                           [a.max() for a in all_vals])]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_norm + [vals_norm[0]], theta=radar_labels + [radar_labels[0]],
            fill='toself', name=f"Comm {c} — {profil}",
            line_color=profil_color(profil), fillcolor=profil_color(profil), opacity=0.25))
    fig_radar.update_layout(height=420, paper_bgcolor='rgba(0,0,0,0)',
        polar=dict(bgcolor='rgba(15,20,35,0.8)',
            radialaxis=dict(visible=True, gridcolor='#1e2535',
                           tickfont_color='#475569', range=[0, 1]),
            angularaxis=dict(gridcolor='#1e2535', tickfont_color='#94a3b8')),
        legend=dict(font_color='#e2e8f0', bgcolor='rgba(15,20,35,0.9)',
                   bordercolor='#1e2535', orientation='h', y=-0.15),
        font_color='#94a3b8', margin=dict(l=40, r=40, t=20, b=40))
    st.plotly_chart(fig_radar, use_container_width=True)
 
# ══════════════════════════════════════════════════════════════
# TAB 4 — SIMILARITÉ
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## Analyse de la matrice de similarité")
 
    with st.spinner("Calcul de la matrice (peut prendre 30-60s)..."):
        sim_full, seuil_val = compute_similarity(norm_bytes, threshold_pct)
 
    idx_upper = np.triu_indices(len(sim_full), k=1)
    vals_all  = sim_full[idx_upper]
 
    col_s1, col_s2 = st.columns(2)
 
    with col_s1:
        st.markdown("### Distribution des similarités")
        p25 = np.percentile(vals_all, 25)
        p90 = np.percentile(vals_all, 90)
 
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=vals_all[::10], nbinsx=60,
            marker_color='#38bdf8', opacity=0.7, name='Similarité'))
        fig_dist.add_vline(x=vals_all.mean(), line_color='#e2e8f0', line_dash='dash',
            annotation_text=f"μ={vals_all.mean():.4f}", annotation_font_color='#e2e8f0')
        fig_dist.add_vline(x=seuil_val, line_color='#E24B4A', line_dash='dashdot',
            annotation_text=f"P{threshold_pct}={seuil_val:.4f}", annotation_font_color='#E24B4A')
        fig_dist.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,20,35,0.5)', font_color='#94a3b8',
            xaxis=dict(gridcolor='#1e2535', title='Similarité'),
            yaxis=dict(gridcolor='#1e2535', title='Fréquence'),
            margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True)
 
        st.markdown(f"""
        <div class='info-box'>
        <b>Statistiques</b><br>
        Paires : <b>{len(vals_all):,}</b> · Moyenne : <b>{vals_all.mean():.4f}</b> ·
        Std : <b>{vals_all.std():.4f}</b><br>
        P25 : <b>{p25:.4f}</b> · Seuil P{threshold_pct} : <b>{seuil_val:.4f}</b> ·
        P90 : <b>{p90:.4f}</b>
        </div>""", unsafe_allow_html=True)
 
    with col_s2:
        st.markdown("### Intra vs Inter groupes")
        labels_diag  = df_raw['Diagnosis'].values
        diab_idx     = np.where(labels_diag == 1)[0]
        non_diab_idx = np.where(labels_diag == 0)[0]
        np.random.seed(42)
        d_s = diab_idx[:300]
        n_s = non_diab_idx[:300]
        sim_dd    = sim_full[np.ix_(d_s, d_s)][np.triu_indices(len(d_s), k=1)]
        sim_nn    = sim_full[np.ix_(n_s, n_s)][np.triu_indices(len(n_s), k=1)]
        sim_inter = sim_full[np.ix_(d_s, n_s)].flatten()
 
        fig_groups = go.Figure()
        for data, name, color in [
            (sim_nn,    "Intra Non-Diab.",   "#38bdf8"),
            (sim_dd,    "Intra Diabétiques", "#E24B4A"),
            (sim_inter, "Inter-groupes",     "#64748b")
        ]:
            fig_groups.add_trace(go.Histogram(x=data[::5],
                name=f"{name} μ={data.mean():.4f}", marker_color=color,
                opacity=0.65, histnorm='probability density', nbinsx=40))
        fig_groups.update_layout(barmode='overlay', height=300,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,20,35,0.5)',
            font_color='#94a3b8', legend=dict(font_color='#e2e8f0', font_size=10),
            xaxis=dict(gridcolor='#1e2535', title='Similarité'),
            yaxis=dict(gridcolor='#1e2535', title='Densité'),
            margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_groups, use_container_width=True)
 
        ratio = ((sim_dd.mean() + sim_nn.mean()) / 2) / sim_inter.mean()
        st.markdown(f"""
<div class='{"success-box" if ratio > 1.05 else "warn-box"}'>
<b>Ratio intra/inter : {ratio:.3f}</b><br>
{"Les patients ont tendance à être plus similaires au sein de leur groupe qu’entre groupes — bonne séparation dans l’espace de similarité." 
 if ratio > 1.05 else 
 "⚠️ Les groupes sont peu différenciés dans l’espace de similarité — forte superposition des profils."}
</div>""", unsafe_allow_html=True)
 
    st.markdown("### Heatmap de la matrice (sous-ensemble 50×50)")
    sub50 = sim_full[:50, :50]
    fig_mini = go.Figure(go.Heatmap(z=sub50, colorscale='YlOrRd', zmin=0.3, zmax=0.85,
        colorbar=dict(tickfont=dict(color='#94a3b8'),
            title=dict(text='Similarité', font=dict(color='#94a3b8')))))
    fig_mini.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(10,14,26,0.9)', font_color='#94a3b8',
        xaxis=dict(title='Patient', showgrid=False),
        yaxis=dict(title='Patient', showgrid=False),
        margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_mini, use_container_width=True)
 
    st.markdown(f"""
    <div class='info-box'>
    <b>Comment lire cette heatmap ?</b><br>
    Plus la cellule est chaude (jaune/rouge), plus deux patients ont des profils similaires. 
    Des blocs colorés sur la diagonale révèlent des groupes de patients aux profils proches.
    </div>""", unsafe_allow_html=True)
 
# ══════════════════════════════════════════════════════════════
# TAB 5 — CLASSIFIER
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## Classifier un nouveau patient")
    st.markdown("""
    <div class='info-box'>
    Entrez les valeurs cliniques d'un nouveau patient. L'algorithme calcule sa similarité
    avec les 1 879 patients existants et l'assigne à la communauté la plus proche (K-NN).
    </div>""", unsafe_allow_html=True)
 
    st.markdown("### Valeurs cliniques du patient")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        age   = st.number_input("Âge (ans)", 18, 100, 55, 1)
        bmi   = st.number_input("BMI (kg/m²)", 10.0, 70.0, 27.5, 0.5)
    with col2:
        fbs   = st.number_input("Glycémie à jeun (mg/dL)", 50, 400, 145, 5)
        hba1c = st.number_input("HbA1c (%)", 3.0, 15.0, 7.8, 0.1)
    with col3:
        sbp   = st.number_input("Pression syst. (mmHg)", 80, 220, 128, 1)
        chol  = st.number_input("Cholestérol total (mg/dL)", 100, 400, 195, 5)
    with col4:
        sleep     = st.number_input("Qualité sommeil (/10)", 0.0, 10.0, 6.0, 0.5)
        k_voisins = st.slider("Nombre de voisins (K)", 5, 50, 20, 5)
 
    def flag(val, lo, hi):
        if val > hi:   return "🔴 ÉLEVÉ"
        elif val < lo: return "🟡 FAIBLE"
        else:          return "🟢 Normal"
 
    flags_data = {
        "Âge": (age, None, None), "BMI": (bmi, 18.5, 24.9),
        "Glycémie": (fbs, 70, 100), "HbA1c": (hba1c, 0, 5.7),
        "PA syst.": (sbp, 90, 120), "Cholestérol": (chol, 0, 200), "Sommeil": (sleep, 6, 10),
    }
    flag_cols = st.columns(7)
    for i, (name, (val, lo, hi)) in enumerate(flags_data.items()):
        with flag_cols[i]:
            f = flag(val, lo, hi) if lo is not None else "—"
            st.markdown(f"""
            <div style='text-align:center; padding:0.4rem; background:#111827;
                 border-radius:8px; border:1px solid #1e2535; font-size:0.75rem;'>
                <div style='color:#64748b; margin-bottom:0.2rem;'>{name}</div>
                <div style='font-family:Space Mono,monospace; color:#e2e8f0; font-size:0.9rem;'>{val}</div>
                <div style='font-size:0.7rem; margin-top:0.2rem;'>{f}</div>
            </div>""", unsafe_allow_html=True)
 
    st.markdown("")
 
    if st.button("Classifier ce patient", use_container_width=False):
        with st.spinner("Calcul de la similarité avec 1879 patients..."):
            df_raw_c, df_norm_c = load_data(raw_bytes, norm_bytes)
            means = df_raw_c[COLS].mean().values
            stds  = df_raw_c[COLS].std().values
            vals_brutes = np.array([age, bmi, fbs, hba1c, sbp, chol, sleep])
            vals_norm   = (vals_brutes - means) / stds
 
            X       = df_norm_c[COLS].values
            sqrt_w  = np.sqrt(POIDS)
            diff    = X * sqrt_w - vals_norm * sqrt_w
            distances  = np.sqrt((diff**2).sum(axis=1))
            similarites = 1.0 / (1.0 + distances)
 
            partition_cls, Q_cls, _, _ = run_louvain(norm_bytes, threshold_pct)
            idx_top = np.argsort(similarites)[::-1][:k_voisins]
 
            votes = {}
            for idx in idx_top:
                comm = partition_cls.get(int(idx), -1)
                votes[comm] = votes.get(comm, 0) + 1
 
            comm_pred  = max(votes, key=votes.get)
            confiance  = votes[comm_pred] / k_voisins * 100
 
            # Va recup le profil dynamique de la communauté prédite
            profiles_cls, _ = get_community_profiles(raw_bytes, norm_bytes, threshold_pct)
            profil_pred = profiles_cls[profiles_cls['communaute'] == comm_pred].iloc[0]['profil']
            color_pred  = profil_color(profil_pred)
            icon_pred   = profil_icon(profil_pred)
            box_pred    = profil_box_class(profil_pred)
 
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        st.markdown("### Résultat de la classification")
 
        col_res, col_vote = st.columns([3, 2])
 
        with col_res:
            st.markdown(f"""
            <div class='result-box' style='border-color:{color_pred};'>
                <div style='display:flex; align-items:center; gap:1rem; margin-bottom:1rem;'>
                    <div style='font-size:2.5rem;'>{icon_pred}</div>
                    <div>
                        <div style='font-family:Space Mono,monospace; font-size:0.8rem;
                             color:{color_pred}; letter-spacing:2px; text-transform:uppercase;'>
                            COMMUNAUTÉ {comm_pred}
                        </div>
                        <div style='font-size:1.1rem; font-weight:600; color:#e2e8f0; margin:0.2rem 0;'>
                            {profil_pred}
                        </div>
                        <div style='font-family:Space Mono,monospace; font-size:1.6rem; color:{color_pred};'>
                            {confiance:.0f}% confiance
                        </div>
                    </div>
                </div>
                <div style='font-size:0.85rem; color:#94a3b8;'>
                    Voisin le plus proche — similarité : <b style='color:#e2e8f0;'>{similarites[idx_top[0]]:.4f}</b><br>
                    Similarité moyenne ({k_voisins} voisins) : <b style='color:#e2e8f0;'>{similarites[idx_top].mean():.4f}</b>
                </div>
            </div>""", unsafe_allow_html=True)
 
            taux_comm = profiles_cls[profiles_cls['communaute'] == comm_pred].iloc[0]['taux_diabete']
            st.markdown(f"""
            <div class='{box_pred}'>
             <b>Profil observé dans cette communauté</b><br>
            Taux de diabète : <b>{taux_comm}%</b> des patients de ce groupe sont diabétiques.<br>
            Ce résultat est basé sur la similarité structurelle du graphe.
            </div>""", unsafe_allow_html=True)
 
        with col_vote:
            st.markdown("### Votes des K voisins")
            fig_votes = go.Figure()
            for c, nb in sorted(votes.items(), key=lambda x: -x[1]):
                p = profiles_cls[profiles_cls['communaute'] == c].iloc[0]['profil']
                fig_votes.add_trace(go.Bar(
                    x=[nb], y=[f"Comm {c}"], orientation='h',
                    marker_color=profil_color(p),
                    text=f"{nb}/{k_voisins} ({nb/k_voisins*100:.0f}%)",
                    textposition='outside', textfont_color='#94a3b8', name=f"Comm {c}"))
            fig_votes.update_layout(height=250, barmode='overlay',
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,20,35,0.5)',
                font_color='#94a3b8', showlegend=False,
                xaxis=dict(gridcolor='#1e2535', range=[0, k_voisins*1.3]),
                yaxis=dict(gridcolor='#1e2535'), margin=dict(l=10, r=80, t=10, b=10))
            st.plotly_chart(fig_votes, use_container_width=True)
 
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        st.markdown("### Lecture clinique indépendante (OMS)")
 
        def clinical_risk(fbs, hba1c, sbp, bmi):
            flags = []
            if fbs >= 126:
                flags.append(("Glycémie", "🔴 Diabète probable", "danger-box"))
            elif fbs >= 100:
                flags.append(("Glycémie", "🟠 Prédiabète", "warn-box"))
            else:
                flags.append(("Glycémie", "🟢 Normal", "success-box"))
 
            if hba1c >= 6.5:
                flags.append(("HbA1c", "🔴 Diabète probable", "danger-box"))
            elif hba1c >= 5.7:
                flags.append(("HbA1c", "🟠 Prédiabète", "warn-box"))
            else:
                flags.append(("HbA1c", "🟢 Normal", "success-box"))
 
            if sbp >= 140:
                flags.append(("PA syst.", "🔴 Hypertension", "danger-box"))
            elif sbp >= 130:
                flags.append(("PA syst.", "🟠 Élevée", "warn-box"))
            else:
                flags.append(("PA syst.", "🟢 Normal", "success-box"))
 
            if bmi >= 30:
                flags.append(("BMI", "🔴 Obésité", "danger-box"))
            elif bmi >= 25:
                flags.append(("BMI", "🟠 Surpoids", "warn-box"))
            else:
                flags.append(("BMI", "🟢 Normal", "success-box"))
 
            return flags
 
        clin_cols = st.columns(4)
        for i, (label, status, box) in enumerate(clinical_risk(fbs, hba1c, sbp, bmi)):
            with clin_cols[i]:
                st.markdown(f"""
                <div class='{box}' style='text-align:center;'>
                    <div style='font-size:0.75rem; color:#64748b;'>{label}</div>
                    <div style='font-size:0.9rem; font-weight:600;'>{status}</div>
                </div>""", unsafe_allow_html=True)
 
        st.markdown("""
        <div class='info-box' style='font-size:0.8rem; margin-top:0.5rem;'>
        Lecture basée sur les guidelines OMS/ADA — indépendante du modèle graphe.
        Le graphe détecte des <b>patterns statistiques</b>, la clinique évalue un <b>risque médical absolu</b>.
        </div>""", unsafe_allow_html=True)
 
        # Z-scores
        st.markdown("### Z-scores du patient")
        z_scores  = list(vals_norm)
        colors_z  = ['#E24B4A' if z > 1 else '#E09020' if z > 0 else '#38bdf8' if z < -1 else '#34d399'
                     for z in z_scores]
        fig_z = go.Figure()
        fig_z.add_trace(go.Bar(x=COLS, y=z_scores, marker_color=colors_z,
            text=[f"{z:+.2f}σ" for z in z_scores],
            textposition='outside', textfont_color='#94a3b8'))
        fig_z.add_hline(y=0,  line_color='#64748b', line_dash='dash')
        fig_z.add_hline(y=1,  line_color='#E24B4A', line_dash='dot',
            annotation_text="+1σ", annotation_font_color='#E24B4A')
        fig_z.add_hline(y=-1, line_color='#38bdf8', line_dash='dot',
            annotation_text="-1σ", annotation_font_color='#38bdf8')
        fig_z.update_layout(height=280, paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,20,35,0.5)', font_color='#94a3b8',
            xaxis=dict(gridcolor='#1e2535'),
            yaxis=dict(gridcolor='#1e2535', title='Z-score (écarts-types)'),
            margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig_z, use_container_width=True)
 
# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align:center; padding:1rem; color:#334155; font-family:Space Mono,monospace;
     font-size:0.7rem; letter-spacing:1px;'>
PROJET GRAPHE DE SIMILARITÉ — PATIENTS DIABÉTIQUES · {len(df_raw)} PATIENTS ·
ALGORITHME LOUVAIN · Q={Q_hdr:.3f} · SEUIL P{threshold_pct}
</div>""", unsafe_allow_html=True)