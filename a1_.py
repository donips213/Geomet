import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import io
import re
import json

try:
    from rasterio.io import MemoryFile
    RASTERIO_DISPONIVEL = True
except ImportError:
    RASTERIO_DISPONIVEL = False

# Importações para a aba de Geometalurgia
try:
    from sklearn.model_selection import KFold, GroupKFold, cross_validate
    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from scipy.stats import skew
    import scipy.cluster.hierarchy as sch
    import statsmodels.api as sm 
    SKLEARN_DISPONIVEL = True
except ImportError:
    SKLEARN_DISPONIVEL = False

# ==========================================================
# CONFIGURAÇÃO INICIAL
# ==========================================================
st.set_page_config(page_title="Geologia 3D & Validação de BD", layout="wide", page_icon="⛏️")

# ==========================================================
# FUNÇÕES DE APOIO - DETECÇÃO AUTOMÁTICA DE COLUNAS
# ==========================================================
MAPEAMENTO_COLUNAS = {
    'id':      ['hole_id', 'holeid', 'shortid', 'id', 'furo', 'hole', 'short_id'],
    'lito':    ['rockcode', 'rock_code', 'lito', 'litologia', 'rocha'],
    'from':    ['from', 'de', 'depth_from', 'prof_de'],
    'to':      ['to', 'ate', 'depth_to', 'prof_ate'],
    'comp':    ['length', 'depth', 'comprimento', 'prof_total', 'max_depth'],
    'x':       ['x', 'easting', 'este', 'coord_x', 'utm_e', 'locationx'],
    'y':       ['y', 'northing', 'norte', 'coord_y', 'utm_n', 'locationy'],
    'z':       ['z', 'elevation', 'elevacao', 'cota', 'rl', 'collar_elev', 'locationz'],
    's_depth': ['depth', 'profundidade', 'prof', 'at', 'distance'],
    'dip':     ['dip', 'mergulho', 'inclinacao'],
    'az':      ['azimute', 'azimuth', 'az', 'bearing', 'rumo'],
    'grade':   ['grade', 'teor', 'au', 'cu', 'valor', 'assay', 'grau'],
}

def identificar_coluna(df, tipo):
    candidatos = MAPEAMENTO_COLUNAS.get(tipo, [])
    for col in df.columns:
        if str(col).lower().strip().replace(" ", "_") in candidatos:
            return col
    return None

def indice_coluna(df, tipo, fallback=0):
    col = identificar_coluna(df, tipo)
    colunas = list(df.columns)
    if col is not None and col in colunas:
        return colunas.index(col)
    return min(fallback, len(colunas) - 1) if colunas else 0

def padronizar_id(nome):
    if pd.isna(nome):
        return nome
    nome = str(nome).strip().upper()
    match = re.match(r"([A-Z]+)(\d+)", nome)
    if match and "-" not in nome:
        return f"{match.group(1)}-{match.group(2)}"
    return nome

# ==========================================================
# FUNÇÕES DE EXPORTAÇÃO
# ==========================================================
def exportar_para_excel(df, nome_aba='Dados'):
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name=nome_aba[:31])
    writer.close()
    return output.getvalue()

def gerar_excel_destacado(df_res, destaques_tabela, nome_aba='Corrigido'):
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df_res.to_excel(writer, index=False, sheet_name=nome_aba[:31])
    workbook = writer.book
    worksheet = writer.sheets[nome_aba[:31]]
    fmt_amarelo = workbook.add_format({'bg_color': '#FFFF00', 'font_color': '#000000'})
    for row_idx, col_idx in destaques_tabela:
        worksheet.write(row_idx + 1, col_idx, df_res.iloc[row_idx, col_idx], fmt_amarelo)
    writer.close()
    return output.getvalue()

def exportar_dxf_polylines(df_3d, id_col='HOLEID'):
    buffer = io.StringIO()
    buffer.write("0\nSECTION\n2\nENTITIES\n")
    for furo, grupo in df_3d.groupby(id_col):
        grupo = grupo.reset_index(drop=True)
        pontos = []
        for i, r in grupo.iterrows():
            if i == 0:
                pontos.append((r['X1'], r['Y1'], r['Z1']))
            pontos.append((r['X2'], r['Y2'], r['Z2']))
        layer = re.sub(r"[^A-Za-z0-9_\-]", "_", str(furo))[:31]
        buffer.write(f"0\nPOLYLINE\n8\n{layer}\n66\n1\n70\n8\n")
        for p in pontos:
            buffer.write(f"0\nVERTEX\n8\n{layer}\n10\n{p[0]:.3f}\n20\n{p[1]:.3f}\n30\n{p[2]:.3f}\n")
        buffer.write("0\nSEQEND\n")
    buffer.write("0\nENDSEC\n0\nEOF\n")
    return buffer.getvalue().encode('utf-8')

def exportar_pontos_genericos(df_3d, id_col='HOLEID', lito_col='LITO'):
    linhas = []
    for furo, grupo in df_3d.groupby(id_col):
        grupo = grupo.reset_index(drop=True)
        pontos = []
        for i, r in grupo.iterrows():
            if i == 0:
                pontos.append((r['X1'], r['Y1'], r['Z1'], r[lito_col]))
            pontos.append((r['X2'], r['Y2'], r['Z2'], r[lito_col]))
        for i, p in enumerate(pontos, start=1):
            linhas.append({'BHID': furo, 'PID': i, 'X': p[0], 'Y': p[1], 'Z': p[2], 'LITO': p[3]})
    return pd.DataFrame(linhas)

def exportar_surpac_str(df_3d, id_col='HOLEID'):
    buffer = io.StringIO()
    for idx_str, (furo, grupo) in enumerate(df_3d.groupby(id_col), start=1):
        grupo = grupo.reset_index(drop=True)
        pontos = []
        for i, r in grupo.iterrows():
            if i == 0:
                pontos.append((r['X1'], r['Y1'], r['Z1']))
            pontos.append((r['X2'], r['Y2'], r['Z2']))
        for p in pontos:
            buffer.write(f"{idx_str},{p[1]:.3f},{p[0]:.3f},{p[2]:.3f},{furo}\n")
        buffer.write("0,0,0,0\n")
    buffer.write("END\n")
    return buffer.getvalue().encode('utf-8')

@st.cache_data(show_spinner="Lendo raster de topografia...")
def carregar_topografia_raster(file_bytes, extensao, fator_decimacao=1):
    with MemoryFile(file_bytes, ext=extensao) as memfile:
        with memfile.open() as src:
            banda = src.read(1).astype(float)
            if src.nodata is not None:
                banda = np.where(banda == src.nodata, np.nan, banda)
            transform = src.transform

    if fator_decimacao > 1:
        banda = banda[::fator_decimacao, ::fator_decimacao]
    altura_d, largura_d = banda.shape

    cols_idx, rows_idx = np.meshgrid(np.arange(largura_d) * fator_decimacao, np.arange(altura_d) * fator_decimacao)
    a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    X = a * (cols_idx + 0.5) + b * (rows_idx + 0.5) + c
    Y = d * (cols_idx + 0.5) + e * (rows_idx + 0.5) + f
    return X, Y, banda

def salvar_projeto(col_map, cores, sel_rejeito, sel_mistura):
    projeto = {
        'col_map': col_map,
        'cores': cores,
        'sel_rejeito': sel_rejeito,
        'sel_mistura': sel_mistura,
    }
    return json.dumps(projeto, indent=2, ensure_ascii=False).encode('utf-8')

# ==========================================================
# VALIDAÇÃO / LIMPEZA DO BANCO DE DADOS
# ==========================================================
def processar_dados_mapeados(df_collar, df_survey, df_lito, cmap, list_rejeito=None, list_mistura=None):
    list_rejeito = list_rejeito or []
    list_mistura = list_mistura or []

    hl_collar, hl_survey, hl_lito = [], [], []
    df_collar['LOG_MODIFICACOES'] = ""
    df_survey['LOG_MODIFICACOES'] = ""
    df_lito['LOG_MODIFICACOES'] = ""

    for df, col_id, hl_list in [(df_collar, cmap['c_id'], hl_collar),
                                 (df_survey, cmap['s_id'], hl_survey),
                                 (df_lito, cmap['l_id'], hl_lito)]:
        col_idx = df.columns.get_loc(col_id)
        log_idx = df.columns.get_loc('LOG_MODIFICACOES')
        for i in range(len(df)):
            orig = str(df.iat[i, col_idx])
            corr = padronizar_id(orig)
            if orig != corr:
                df.iat[i, col_idx] = corr
                df.iat[i, log_idx] += f"[ID: {orig}->{corr}] "
                hl_list.append((i, col_idx))

    from_idx = df_lito.columns.get_loc(cmap['l_from'])
    to_idx = df_lito.columns.get_loc(cmap['l_to'])
    log_idx_lito = df_lito.columns.get_loc('LOG_MODIFICACOES')
    for i in range(len(df_lito)):
        if df_lito.iat[i, from_idx] == df_lito.iat[i, to_idx]:
            v_antigo = df_lito.iat[i, to_idx]
            df_lito.iat[i, to_idx] = v_antigo + 0.1
            df_lito.iat[i, log_idx_lito] += f"[Zero: TO {v_antigo}->{v_antigo + 0.1}] "
            hl_lito.append((i, to_idx))

    lito_idx = df_lito.columns.get_loc(cmap['l_lito'])
    sinonimos = cmap.get('sinonimos_lito', {})
    if sinonimos:
        for i in range(len(df_lito)):
            orig = str(df_lito.iat[i, lito_idx]).upper()
            corr = orig
            for de, para in sinonimos.items():
                corr = corr.replace(de.upper(), para.upper())
            if orig != corr:
                df_lito.iat[i, lito_idx] = corr
                df_lito.iat[i, log_idx_lito] += f"[Lito: {orig}->{corr}] "
                hl_lito.append((i, lito_idx))

    novas_linhas = []
    max_to_por_furo = {}
    litologia_fundo = cmap.get('litologia_fundo', 'SAPROLITO')

    for furo, grupo in df_lito.groupby(cmap['l_id']):
        if pd.isna(furo) or str(furo).strip() == "":
            continue
        profundidades = pd.to_numeric(grupo[cmap['l_to']], errors='coerce')
        if profundidades.isna().all():
            continue
        max_idx = profundidades.idxmax()
        ultimo = grupo.loc[max_idx]
        max_t = ultimo[cmap['l_to']]
        max_to_por_furo[furo] = max_t

        if str(ultimo[cmap['l_lito']]).upper() != litologia_fundo.upper():
            t_novo = max_t + 0.1
            nova_data = {c: None for c in df_lito.columns}
            nova_data.update({
                cmap['l_id']: furo, cmap['l_lito']: litologia_fundo,
                cmap['l_from']: max_t, cmap['l_to']: t_novo,
                'LOG_MODIFICACOES': f'LINHA ADC ({litologia_fundo})'
            })
            novas_linhas.append(nova_data)
            max_to_por_furo[furo] = t_novo

    if novas_linhas:
        df_lito = pd.concat([df_lito, pd.DataFrame(novas_linhas)], ignore_index=True)

    if cmap['c_comp']:
        comp_idx = df_collar.columns.get_loc(cmap['c_comp'])
        id_idx_col = df_collar.columns.get_loc(cmap['c_id'])
        log_idx_col = df_collar.columns.get_loc('LOG_MODIFICACOES')
        for i in range(len(df_collar)):
            furo = str(df_collar.iat[i, id_idx_col])
            if furo in max_to_por_furo:
                lito_max = max_to_por_furo[furo]
                v_ant = pd.to_numeric(df_collar.iat[i, comp_idx], errors='coerce')
                if pd.isna(v_ant) or abs(v_ant - lito_max) > 0.001:
                    df_collar.iat[i, comp_idx] = lito_max
                    df_collar.iat[i, log_idx_col] += f"[Comp: {v_ant}->{lito_max}] "
                    hl_collar.append((i, comp_idx))

    if list_rejeito or list_mistura:
        df_lito['CLASSIFICACAO'] = ""
        class_idx = df_lito.columns.get_loc('CLASSIFICACAO')
        lito_idx = df_lito.columns.get_loc(cmap['l_lito'])
        log_idx_lito = df_lito.columns.get_loc('LOG_MODIFICACOES')
        rejeitos_upper = [str(r).upper() for r in list_rejeito]
        misturas_upper = [str(m).upper() for m in list_mistura]

        for i in range(len(df_lito)):
            val_lito = str(df_lito.iat[i, lito_idx]).upper()
            if val_lito in rejeitos_upper:
                df_lito.iat[i, class_idx] = "REJEITO"
                df_lito.iat[i, log_idx_lito] += "[Class: REJEITO] "
                hl_lito.append((i, class_idx))
            elif val_lito in misturas_upper:
                df_lito.iat[i, class_idx] = "MISTURA"
                df_lito.iat[i, log_idx_lito] += "[Class: MISTURA] "
                hl_lito.append((i, class_idx))

    return [df_collar, df_lito, df_survey], [hl_collar, hl_lito, hl_survey]

def validar_intervalos(df, id_col, from_col, to_col, tol=0.01):
    problemas = []
    froms_all = pd.to_numeric(df[from_col], errors='coerce')
    tos_all = pd.to_numeric(df[to_col], errors='coerce')
    invalidos = df[froms_all >= tos_all]
    for _, r in invalidos.iterrows():
        problemas.append({'FURO': r[id_col], 'TIPO': 'FROM>=TO',
                           'DETALHE': f"FROM={r[from_col]} TO={r[to_col]}"})
    for furo, grupo in df.groupby(id_col):
        grupo = grupo.assign(_f=pd.to_numeric(grupo[from_col], errors='coerce'),
                              _t=pd.to_numeric(grupo[to_col], errors='coerce'))
        grupo = grupo.sort_values('_f').reset_index(drop=True)
        for i in range(len(grupo) - 1):
            f2, t1 = grupo['_f'].iloc[i + 1], grupo['_t'].iloc[i]
            if pd.isna(f2) or pd.isna(t1):
                continue
            diff = f2 - t1
            if diff < -tol:
                problemas.append({'FURO': furo, 'TIPO': 'SOBREPOSICAO',
                                   'DETALHE': f"TO={t1:.2f} > FROM seguinte={f2:.2f}"})
            elif diff > tol:
                problemas.append({'FURO': furo, 'TIPO': 'GAP',
                                   'DETALHE': f"Gap de {diff:.2f} m entre {t1:.2f} e {f2:.2f}"})
    return pd.DataFrame(problemas)

def validar_survey_angulos(df_survey, id_col, dip_col, az_col, depth_col, limiar_dogleg=15.0):
    problemas = []
    for furo, grupo in df_survey.groupby(id_col):
        grupo = grupo.assign(_d=pd.to_numeric(grupo[depth_col], errors='coerce'),
                              _dip=pd.to_numeric(grupo[dip_col], errors='coerce'),
                              _az=pd.to_numeric(grupo[az_col], errors='coerce'))
        grupo = grupo.sort_values('_d').reset_index(drop=True)
        for i in range(len(grupo)):
            dip_i = grupo['_dip'].iloc[i]
            if pd.notna(dip_i) and not (-90 <= dip_i <= 90):
                problemas.append({'FURO': furo, 'TIPO': 'DIP_INVALIDO',
                                   'DETALHE': f"Dip={dip_i} na profundidade {grupo['_d'].iloc[i]}"})
        for i in range(len(grupo) - 1):
            d1, d2 = grupo['_dip'].iloc[i], grupo['_dip'].iloc[i + 1]
            if pd.notna(d1) and pd.notna(d2) and abs(d2 - d1) > limiar_dogleg:
                problemas.append({'FURO': furo, 'TIPO': 'WILD_SHOT',
                                   'DETALHE': f"Variação de {abs(d2 - d1):.1f}° no dip entre "
                                              f"{grupo['_d'].iloc[i]}m e {grupo['_d'].iloc[i + 1]}m"})
    return pd.DataFrame(problemas)

def detectar_inconsistencias_furos(df_collar, df_survey, df_lito, cmap):
    ids_collar = set(df_collar[cmap['c_id']].astype(str))
    ids_survey = set(df_survey[cmap['s_id']].astype(str))
    ids_lito = set(df_lito[cmap['l_id']].astype(str))
    dup_mask = df_collar[cmap['c_id']].astype(str).duplicated(keep=False)
    return {
        'sem_survey': sorted(ids_collar - ids_survey),
        'sem_lito': sorted(ids_collar - ids_lito),
        'lito_sem_collar': sorted(ids_lito - ids_collar),
        'duplicados_collar': sorted(df_collar.loc[dup_mask, cmap['c_id']].astype(str).unique().tolist()),
    }

def detectar_outliers_coordenadas(df_collar, x_col, y_col, z_col, limiar_z=4.0):
    problemas = []
    for col, nome in [(x_col, 'X'), (y_col, 'Y'), (z_col, 'Z')]:
        vals = pd.to_numeric(df_collar[col], errors='coerce')
        media, desvio = vals.mean(), vals.std()
        if desvio and desvio > 0:
            z_scores = (vals - media).abs() / desvio
            for idx in z_scores[z_scores > limiar_z].index:
                problemas.append({'FURO': df_collar.loc[idx, x_col] if False else None,
                                   'COLUNA': nome, 'VALOR': vals[idx], 'Z_SCORE': round(z_scores[idx], 1)})
    return pd.DataFrame(problemas)

# ==========================================================
# DESURVEY
# ==========================================================
def _min_curve_vec(md1, incl1_deg, az1_deg, md2, incl2_deg, az2_deg):
    md1 = np.asarray(md1, dtype=float)
    md2 = np.asarray(md2, dtype=float)
    i1 = np.radians(np.asarray(incl1_deg, dtype=float))
    i2 = np.radians(np.asarray(incl2_deg, dtype=float))
    a1 = np.radians(np.asarray(az1_deg, dtype=float))
    a2 = np.radians(np.asarray(az2_deg, dtype=float))

    cos_dl = np.cos(i2 - i1) - np.sin(i1) * np.sin(i2) * (1 - np.cos(a2 - a1))
    cos_dl = np.clip(cos_dl, -1.0, 1.0)
    dl = np.arccos(cos_dl)
    dl_seguro = np.where(dl < 1e-9, 1.0, dl)
    rf = np.where(dl < 1e-9, 1.0, (2.0 / dl_seguro) * np.tan(dl_seguro / 2.0))

    dmd = (md2 - md1) / 2.0
    d_leste = dmd * (np.sin(i1) * np.sin(a1) + np.sin(i2) * np.sin(a2)) * rf
    d_norte = dmd * (np.cos(a1) * np.sin(i1) + np.cos(a2) * np.sin(i2)) * rf
    d_tvd = dmd * (np.cos(i1) + np.cos(i2)) * rf
    return d_leste, d_norte, d_tvd

def _construir_trajetoria(x0, y0, z0, mds, dips, azs, inv_dip, padrao_vertical):
    if len(mds) == 0:
        if padrao_vertical:
            mds, dips, azs = np.array([0.0, 1.0]), np.array([-90.0, -90.0]), np.array([0.0, 0.0])
        else:
            return None

    dips = np.where(dips > 0, -dips, dips) if inv_dip else dips
    incls = 90.0 + dips 

    ordem = np.argsort(mds)
    mds, incls, azs = mds[ordem], incls[ordem], azs[ordem]

    if mds[0] > 0:
        mds = np.concatenate(([0.0], mds))
        incls = np.concatenate(([incls[0]], incls))
        azs = np.concatenate(([azs[0]], azs))

    traj_md = [0.0]
    traj_x, traj_y, traj_z = [x0], [y0], [z0]
    traj_incl, traj_az = [incls[0]], [azs[0]]

    for i in range(1, len(mds)):
        if mds[i] <= traj_md[-1]:
            continue
        de, dn, dtvd = _min_curve_vec(traj_md[-1], traj_incl[-1], traj_az[-1], mds[i], incls[i], azs[i])
        traj_md.append(float(mds[i]))
        traj_x.append(traj_x[-1] + float(de))
        traj_y.append(traj_y[-1] + float(dn))
        traj_z.append(traj_z[-1] - float(dtvd))
        traj_incl.append(incls[i])
        traj_az.append(azs[i])

    return {
        'md': np.array(traj_md), 'x': np.array(traj_x), 'y': np.array(traj_y), 'z': np.array(traj_z),
        'incl': np.array(traj_incl), 'az': np.array(traj_az),
    }

def _interpolar_posicoes(traj, mds_alvo):
    mds_alvo = np.asarray(mds_alvo, dtype=float)
    idx_bracket = np.clip(np.searchsorted(traj['md'], mds_alvo, side='right') - 1, 0, len(traj['md']) - 2 if len(traj['md']) > 1 else 0)

    md1 = traj['md'][idx_bracket]
    md2 = traj['md'][np.minimum(idx_bracket + 1, len(traj['md']) - 1)]
    incl1 = traj['incl'][idx_bracket]
    incl2 = traj['incl'][np.minimum(idx_bracket + 1, len(traj['md']) - 1)]
    az1 = traj['az'][idx_bracket]
    az2 = traj['az'][np.minimum(idx_bracket + 1, len(traj['md']) - 1)]
    x1 = traj['x'][idx_bracket]
    y1 = traj['y'][idx_bracket]
    z1 = traj['z'][idx_bracket]

    frac = np.where(md2 > md1, (mds_alvo - md1) / np.where(md2 > md1, md2 - md1, 1.0), 0.0)
    frac = np.clip(frac, 0.0, None)  
    incl_alvo = incl1 + (incl2 - incl1) * np.clip(frac, 0, 1)
    az_alvo = az1 + (az2 - az1) * np.clip(frac, 0, 1)
    alem = mds_alvo > traj['md'][-1]
    incl_alvo = np.where(alem, traj['incl'][-1], incl_alvo)
    az_alvo = np.where(alem, traj['az'][-1], az_alvo)

    de, dn, dtvd = _min_curve_vec(md1, incl1, az1, mds_alvo, incl_alvo, az_alvo)
    x = x1 + de
    y = y1 + dn
    z = z1 - dtvd
    return x, y, z

@st.cache_data(show_spinner=False)
def calcular_desurvey_otimizado(df_collar, df_survey, df_lito, col_map, metodo='curvatura_minima'):
    df_collar = df_collar.copy()
    df_survey = df_survey.copy()
    df_lito = df_lito.copy()

    df_collar[col_map['c_id']] = df_collar[col_map['c_id']].astype(str)
    df_survey[col_map['s_id']] = df_survey[col_map['s_id']].astype(str)
    df_lito[col_map['l_id']] = df_lito[col_map['l_id']].astype(str)

    dict_collar = df_collar.drop_duplicates(subset=[col_map['c_id']]).set_index(col_map['c_id'])[[col_map['c_x'], col_map['c_y'], col_map['c_z']]].to_dict('index')
    df_survey_sorted = df_survey.sort_values(by=[col_map['s_id'], col_map['s_depth']])
    dict_survey = {furo: grupo for furo, grupo in df_survey_sorted.groupby(col_map['s_id'])}

    idx_grade = col_map['l_grade']
    resultados = []

    for furo, grupo_lito in df_lito.groupby(col_map['l_id']):
        if furo not in dict_collar:
            continue
        x0 = dict_collar[furo][col_map['c_x']]
        y0 = dict_collar[furo][col_map['c_y']]
        z0 = dict_collar[furo][col_map['c_z']]

        if furo in dict_survey:
            s_group = dict_survey[furo]
            mds = pd.to_numeric(s_group[col_map['s_depth']], errors='coerce').values
            dips = pd.to_numeric(s_group[col_map['s_dip']], errors='coerce').values
            azs = pd.to_numeric(s_group[col_map['s_az']], errors='coerce').values
            valido = ~(np.isnan(mds) | np.isnan(dips) | np.isnan(azs))
            mds, dips, azs = mds[valido], dips[valido], azs[valido]
        else:
            mds, dips, azs = np.array([]), np.array([]), np.array([])

        if len(mds) == 0 and not col_map.get('padrao_vertical', True):
            continue

        froms = pd.to_numeric(grupo_lito[col_map['l_from']], errors='coerce').values
        tos = pd.to_numeric(grupo_lito[col_map['l_to']], errors='coerce').values
        litos = grupo_lito[col_map['l_lito']].values
        if idx_grade:
            grades = pd.to_numeric(grupo_lito[idx_grade], errors='coerce').fillna(0).values
        else:
            grades = np.zeros(len(grupo_lito))

        mask_ok = ~(np.isnan(froms) | np.isnan(tos))
        if not mask_ok.any():
            continue
        froms, tos, litos, grades = froms[mask_ok], tos[mask_ok], litos[mask_ok], grades[mask_ok]

        if metodo == 'tangencial':
            if len(mds) == 0:
                dip_i, az_i = -90.0, 0.0
                dip_arr = np.full(len(froms), dip_i)
                az_arr = np.full(len(froms), az_i)
            else:
                pos = np.clip(np.searchsorted(mds, froms), 0, len(mds) - 1)
                dip_arr, az_arr = dips[pos], azs[pos]
            if col_map['inv_dip']:
                dip_arr = np.where(dip_arr > 0, -dip_arr, dip_arr)
            dip_rad, az_rad = np.radians(dip_arr), np.radians(az_arr)
            x1 = x0 + froms * np.cos(dip_rad) * np.sin(az_rad)
            y1 = y0 + froms * np.cos(dip_rad) * np.cos(az_rad)
            z1 = z0 + froms * np.sin(dip_rad)
            x2 = x0 + tos * np.cos(dip_rad) * np.sin(az_rad)
            y2 = y0 + tos * np.cos(dip_rad) * np.cos(az_rad)
            z2 = z0 + tos * np.sin(dip_rad)
        else:
            traj = _construir_trajetoria(x0, y0, z0, mds, dips, azs,
                                          col_map['inv_dip'], col_map.get('padrao_vertical', True))
            if traj is None:
                continue
            x1, y1, z1 = _interpolar_posicoes(traj, froms)
            x2, y2, z2 = _interpolar_posicoes(traj, tos)

        for i in range(len(froms)):
            resultados.append((furo, froms[i], tos[i], litos[i], grades[i],
                                x1[i], y1[i], z1[i], x2[i], y2[i], z2[i]))

    return pd.DataFrame(resultados, columns=[
        'HOLEID', 'FROM', 'TO', 'LITO', 'GRADE', 'X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2'
    ])

# ==========================================================
# COMPOSITAGEM
# ==========================================================
def compositar_intervalos(df_3d, comprimento=1.0):
    resultados = []
    for furo, grupo in df_3d.groupby('HOLEID'):
        grupo = grupo.sort_values('FROM').reset_index(drop=True)
        prof_min, prof_max = grupo['FROM'].min(), grupo['TO'].max()
        limite = prof_min
        while limite < prof_max - 1e-6:
            limite_sup = min(limite + comprimento, prof_max)
            sub = grupo[(grupo['FROM'] < limite_sup) & (grupo['TO'] > limite)]
            if sub.empty:
                limite = limite_sup
                continue
            pesos, grades_pond, litos, linha_dominante, peso_max = [], [], [], None, -1
            for _, r in sub.iterrows():
                sobreposicao = min(r['TO'], limite_sup) - max(r['FROM'], limite)
                if sobreposicao <= 0:
                    continue
                pesos.append(sobreposicao)
                grades_pond.append(r['GRADE'] * sobreposicao if pd.notna(r['GRADE']) else 0)
                litos.append(r['LITO'])
                if sobreposicao > peso_max:
                    peso_max = sobreposicao
                    linha_dominante = r
            if pesos:
                soma_peso = sum(pesos)
                grade_medio = sum(grades_pond) / soma_peso if soma_peso > 0 else np.nan
                lito_dominante = litos[int(np.argmax(pesos))]
                if linha_dominante is not None and linha_dominante['TO'] > linha_dominante['FROM']:
                    frac_ini = (limite - linha_dominante['FROM']) / (linha_dominante['TO'] - linha_dominante['FROM'])
                    frac_fim = (limite_sup - linha_dominante['FROM']) / (linha_dominante['TO'] - linha_dominante['FROM'])
                    frac_ini, frac_fim = np.clip(frac_ini, 0, 1), np.clip(frac_fim, 0, 1)
                    x1c = linha_dominante['X1'] + frac_ini * (linha_dominante['X2'] - linha_dominante['X1'])
                    y1c = linha_dominante['Y1'] + frac_ini * (linha_dominante['Y2'] - linha_dominante['Y1'])
                    z1c = linha_dominante['Z1'] + frac_ini * (linha_dominante['Z2'] - linha_dominante['Z1'])
                    x2c = linha_dominante['X1'] + frac_fim * (linha_dominante['X2'] - linha_dominante['X1'])
                    y2c = linha_dominante['Y1'] + frac_fim * (linha_dominante['Y2'] - linha_dominante['Y1'])
                    z2c = linha_dominante['Z1'] + frac_fim * (linha_dominante['Z2'] - linha_dominante['Z1'])
                else:
                    x1c = y1c = z1c = x2c = y2c = z2c = np.nan
                resultados.append({
                    'HOLEID': furo, 'FROM': limite, 'TO': limite_sup,
                    'LITO_DOMINANTE': lito_dominante, 'GRADE_COMPOSITO': grade_medio,
                    'X1': x1c, 'Y1': y1c, 'Z1': z1c, 'X2': x2c, 'Y2': y2c, 'Z2': z2c,
                })
            limite = limite_sup
    return pd.DataFrame(resultados)

# ==========================================================
# INTERFACE PRINCIPAL
# ==========================================================
st.title("Visualizador 3D & Validação de BD ⛏️")

with st.sidebar.expander("💾 Projeto (salvar/carregar mapeamento)", expanded=False):
    projeto_upload = st.file_uploader("Carregar projeto (.json)", type=['json'], key="upl_projeto")
    projeto_carregado = None
    if projeto_upload is not None:
        try:
            projeto_carregado = json.load(projeto_upload)
            st.success("Projeto carregado — os campos abaixo serão pré-preenchidos quando possível.")
        except Exception as e:
            st.error(f"Não foi possível ler o projeto: {e}")

col1, col2, col3 = st.columns(3)
with col1:
    u_collar = st.file_uploader("1. Collar", type=['csv', 'xlsx'])
with col2:
    u_survey = st.file_uploader("2. Survey", type=['csv', 'xlsx'])
with col3:
    u_lito = st.file_uploader("3. Lito/Assay", type=['csv', 'xlsx'])

u_topo = st.file_uploader(
    "4. Topografia (opcional) — CSV/TXT com colunas X,Y,Z, ou raster GeoTIFF/ASCII Grid (.tif, .tiff, .asc)",
    type=['csv', 'txt', 'tif', 'tiff', 'asc']
)

if u_collar and u_survey and u_lito:

    st.markdown("### ⚙️ Configuração de Leitura (Pular Metadados)")
    c1, c2, c3 = st.columns(3)

    @st.cache_data(show_spinner=False)
    def preview_raw(file_bytes, file_name):
        buffer = io.BytesIO(file_bytes)
        if file_name.endswith('.xlsx'):
            return pd.read_excel(buffer, header=None, nrows=10)
        try:
            return pd.read_csv(buffer, sep=None, engine='python', encoding='utf-8', header=None, nrows=10)
        except Exception:
            buffer.seek(0)
            return pd.read_csv(buffer, sep=None, engine='python', encoding='latin1', header=None, nrows=10)

    def get_header(u_file, col_st, titulo):
        with col_st:
            st.write(f"**Prévia: {titulo}**")
            df_raw = preview_raw(u_file.getvalue(), u_file.name)
            st.dataframe(df_raw.head(3), use_container_width=True)
            return st.number_input(f"Linha do Cabeçalho ({titulo})", min_value=0, max_value=max(0, len(df_raw) - 1), value=0, key=f"h_{titulo}")

    h_collar = get_header(u_collar, c1, "Collar")
    h_survey = get_header(u_survey, c2, "Survey")
    h_lito = get_header(u_lito, c3, "Lito")

    @st.cache_data(show_spinner="Carregando e lendo arquivo...")
    def load_final(file_bytes, file_name, header_idx):
        buffer = io.BytesIO(file_bytes)
        if file_name.endswith('.xlsx'):
            df = pd.read_excel(buffer, header=header_idx)
        else:
            try:
                df = pd.read_csv(buffer, sep=None, engine='python', encoding='utf-8', header=header_idx)
            except Exception:
                buffer.seek(0)
                df = pd.read_csv(buffer, sep=None, engine='python', encoding='latin1', header=header_idx)
        df.columns = [str(c).strip() for c in df.columns]
        return df.reset_index(drop=True)

    df_collar = load_final(u_collar.getvalue(), u_collar.name, h_collar)
    df_survey = load_final(u_survey.getvalue(), u_survey.name, h_survey)
    df_lito = load_final(u_lito.getvalue(), u_lito.name, h_lito)

    df_topo = None
    topo_grid = None
    if u_topo is not None:
        ext_topo = "." + u_topo.name.split(".")[-1].lower()
        if ext_topo in (".tif", ".tiff", ".asc"):
            if not RASTERIO_DISPONIVEL:
                st.error("Leitura de raster requer a biblioteca `rasterio`. Adicione `rasterio` ao "
                         "requirements.txt do projeto e reinicie o app para habilitar este recurso.")
            else:
                altura_px = largura_px = None
                try:
                    with MemoryFile(u_topo.getvalue(), ext=ext_topo) as _mf:
                        with _mf.open() as _src:
                            altura_px, largura_px = _src.height, _src.width
                except Exception as e:
                    st.error(f"Não foi possível ler o raster: {e}")

                if altura_px and largura_px:
                    st.caption(f"Raster com {largura_px} x {altura_px} pixels.")
                    sugestao = max(1, int(np.ceil(max(altura_px, largura_px) / 300)))
                    fator_dec = st.slider(
                        "Fator de decimação da topografia (reduz pontos para manter a performance)",
                        1, 50, sugestao
                    )
                    try:
                        X_topo, Y_topo, Z_topo = carregar_topografia_raster(u_topo.getvalue(), ext_topo, fator_dec)
                        topo_grid = (X_topo, Y_topo, Z_topo)
                        st.success(f"Topografia carregada: grade {Z_topo.shape[1]} x {Z_topo.shape[0]} pontos após decimação.")
                    except Exception as e:
                        st.error(f"Erro ao processar o raster: {e}")
        else:
            try:
                df_topo = pd.read_csv(u_topo, sep=None, engine='python')
            except Exception as e:
                st.warning(f"Não foi possível ler a topografia: {e}")

    st.markdown("---")
    validar_dados = st.checkbox("✅ Executar validação e limpeza automática de BD", value=False)
    st.markdown("---")

    # Mapeamento
    st.sidebar.header("Mapeamento de Colunas")
    col_map = {}
    pj = projeto_carregado.get('col_map', {}) if projeto_carregado else {}

    def idx_ou_projeto(df, tipo, chave_pj, fallback):
        if pj.get(chave_pj) in df.columns:
            return list(df.columns).index(pj[chave_pj])
        return indice_coluna(df, tipo, fallback)

    st.sidebar.subheader("Tabela Collar")
    col_map['c_id'] = st.sidebar.selectbox("Furo ID (Collar)", df_collar.columns, index=idx_ou_projeto(df_collar, 'id', 'c_id', 0))
    col_map['c_x'] = st.sidebar.selectbox("Easting (X)", df_collar.columns, index=idx_ou_projeto(df_collar, 'x', 'c_x', 1 if len(df_collar.columns) > 1 else 0))
    col_map['c_y'] = st.sidebar.selectbox("Northing (Y)", df_collar.columns, index=idx_ou_projeto(df_collar, 'y', 'c_y', 2 if len(df_collar.columns) > 2 else 0))
    col_map['c_z'] = st.sidebar.selectbox("Elevation (Z)", df_collar.columns, index=idx_ou_projeto(df_collar, 'z', 'c_z', 3 if len(df_collar.columns) > 3 else 0))
    opcoes_comp = ["Nenhuma"] + list(df_collar.columns)
    col_comp_auto = identificar_coluna(df_collar, 'comp')
    col_map['c_comp'] = st.sidebar.selectbox("Comprimento Máx", opcoes_comp, index=opcoes_comp.index(col_comp_auto) if col_comp_auto in opcoes_comp else 0)
    if col_map['c_comp'] == "Nenhuma":
        col_map['c_comp'] = None

    st.sidebar.subheader("Tabela Survey")
    col_map['s_id'] = st.sidebar.selectbox("Furo ID (Survey)", df_survey.columns, index=idx_ou_projeto(df_survey, 'id', 's_id', 0))
    col_map['s_depth'] = st.sidebar.selectbox("Profundidade", df_survey.columns, index=idx_ou_projeto(df_survey, 's_depth', 's_depth', 1 if len(df_survey.columns) > 1 else 0))
    col_map['s_dip'] = st.sidebar.selectbox("Mergulho (Dip)", df_survey.columns, index=idx_ou_projeto(df_survey, 'dip', 's_dip', 2 if len(df_survey.columns) > 2 else 0))
    col_map['s_az'] = st.sidebar.selectbox("Azimute", df_survey.columns, index=idx_ou_projeto(df_survey, 'az', 's_az', 3 if len(df_survey.columns) > 3 else 0))

    st.sidebar.subheader("Tabela Lito/Assay")
    col_map['l_id'] = st.sidebar.selectbox("Furo ID (Lito)", df_lito.columns, index=idx_ou_projeto(df_lito, 'id', 'l_id', 0))
    col_map['l_from'] = st.sidebar.selectbox("De (From)", df_lito.columns, index=idx_ou_projeto(df_lito, 'from', 'l_from', 1 if len(df_lito.columns) > 1 else 0))
    col_map['l_to'] = st.sidebar.selectbox("Até (To)", df_lito.columns, index=idx_ou_projeto(df_lito, 'to', 'l_to', 2 if len(df_lito.columns) > 2 else 0))
    col_map['l_lito'] = st.sidebar.selectbox("Litologia", df_lito.columns, index=idx_ou_projeto(df_lito, 'lito', 'l_lito', 3 if len(df_lito.columns) > 3 else 0))
    opcoes_grade = ["Nenhuma"] + list(df_lito.columns)
    col_grade_auto = identificar_coluna(df_lito, 'grade')
    col_map['l_grade'] = st.sidebar.selectbox("Teor Principal (Au)", opcoes_grade, index=opcoes_grade.index(col_grade_auto) if col_grade_auto in opcoes_grade else 0)
    if col_map['l_grade'] == "Nenhuma":
        col_map['l_grade'] = None

    with st.sidebar.expander("Colunas numéricas extras (correlação)", expanded=False):
        numericas_lito = df_lito.select_dtypes(include=np.number).columns.tolist()
        cols_correlacao = st.multiselect("Selecione elementos/variáveis para matriz de correlação", numericas_lito)
        
    with st.sidebar.expander("Variáveis Iniciais Geometalurgia (Tab 8)", expanded=False):
        st.caption("Configure os alvos principais aqui. Mais variáveis serão selecionadas dentro da própria aba.")
        opcoes_lito_num = ["Nenhuma"] + df_lito.select_dtypes(include=np.number).columns.tolist()
        col_map['l_rendimento'] = st.selectbox("Rendimento (%)", opcoes_lito_num, index=0)
        col_map['l_rec_au'] = st.selectbox("Recuperação de Au (%)", opcoes_lito_num, index=0)

    st.sidebar.subheader("Configurações 3D")
    col_map['inv_dip'] = st.sidebar.checkbox("Inverter Dip", value=True)
    col_map['padrao_vertical'] = st.sidebar.checkbox("Furo Vertical se faltar Survey", value=True)
    metodo_desurvey = st.sidebar.radio("Método de Desurvey", ["Curvatura Mínima (recomendado)", "Tangencial (legado)"], index=0)
    metodo_desurvey_key = 'curvatura_minima' if metodo_desurvey.startswith('Curvatura') else 'tangencial'
    z_scale = st.sidebar.slider("Exagero Z", 0.1, 5.0, 1.0)

    with st.sidebar.expander("Sinônimos de Litologia (editável)", expanded=False):
        st.caption("Um par 'DE=PARA' por linha, ex.: SOLO=SEDIMENTO")
        texto_sinonimos = st.text_area("Substituições", value="SOLO=SEDIMENTO\nEMBASAMENTO=SAPROLITO", height=100)
        sinonimos_lito = {}
        for linha in texto_sinonimos.splitlines():
            if '=' in linha:
                de, para = linha.split('=', 1)
                sinonimos_lito[de.strip()] = para.strip()
        col_map['sinonimos_lito'] = sinonimos_lito
        col_map['litologia_fundo'] = st.text_input("Litologia esperada no fundo do furo", value="SAPROLITO")

    # 4. VALIDAÇÃO
    if validar_dados:
        st.markdown("### 🏷️ Classificação de Material (Opcional)")
        todas_litos = df_lito[col_map['l_lito']].dropna().astype(str).unique().tolist()

        cr, cm = st.columns(2)
        with cr:
            sel_rejeito = st.multiselect("Litologias de REJEITO (Estéril)", todas_litos)
        with cm:
            sel_mistura = st.multiselect("Litologias de MISTURA (Minério)", todas_litos)

        with st.spinner("Auditando..."):
            dfs_out, destaques = processar_dados_mapeados(df_collar.copy(), df_survey.copy(), df_lito.copy(), col_map, sel_rejeito, sel_mistura)
            df_collar, df_lito, df_survey = dfs_out[0], dfs_out[1], dfs_out[2]
            with st.expander("📥 Planilhas Corrigidas", expanded=True):
                st.success("Validação concluída!")
                nomes = ["Collar", "Lito_Assay", "Survey"]
                cols_down = st.columns(3)
                for i, df_res in enumerate(dfs_out):
                    with cols_down[i]:
                        excel_bytes = gerar_excel_destacado(df_res, destaques[i])
                        st.download_button(f"Baixar {nomes[i]}", excel_bytes, f"{nomes[i]}_Auditado.xlsx", key=f"dl_{nomes[i]}")

        st.markdown("### 🔎 Checagens Estruturais Avançadas")
        with st.expander("Ver problemas de intervalo, survey e furos", expanded=False):
            prob_intervalos = validar_intervalos(df_lito, col_map['l_id'], col_map['l_from'], col_map['l_to'])
            prob_survey = validar_survey_angulos(df_survey, col_map['s_id'], col_map['s_dip'], col_map['s_az'], col_map['s_depth'])
            inconsist = detectar_inconsistencias_furos(df_collar, df_survey, df_lito, col_map)

            t_int, t_srv, t_furo = st.tabs(["Intervalos (Lito)", "Ângulos de Survey", "Furos entre tabelas"])
            with t_int:
                if prob_intervalos.empty:
                    st.success("Nenhuma sobreposição, gap ou FROM≥TO detectado.")
                else:
                    st.warning(f"{len(prob_intervalos)} problema(s) encontrado(s).")
                    st.dataframe(prob_intervalos, use_container_width=True)
            with t_srv:
                if prob_survey.empty:
                    st.success("Nenhum ângulo inválido ou variação abrupta detectada.")
                else:
                    st.warning(f"{len(prob_survey)} problema(s) encontrado(s).")
                    st.dataframe(prob_survey, use_container_width=True)
            with t_furo:
                cfa, cfb = st.columns(2)
                with cfa:
                    st.write(f"**Furos no Collar sem Survey ({len(inconsist['sem_survey'])})**")
                    st.write(", ".join(inconsist['sem_survey'][:200]) or "—")
                    st.write(f"**Furos no Collar sem Lito ({len(inconsist['sem_lito'])})**")
                    st.write(", ".join(inconsist['sem_lito'][:200]) or "—")
                with cfb:
                    st.write(f"**Furos na Lito ausentes no Collar ({len(inconsist['lito_sem_collar'])})**")
                    st.write(", ".join(inconsist['lito_sem_collar'][:200]) or "—")
                    st.write(f"**IDs duplicados no Collar ({len(inconsist['duplicados_collar'])})**")
                    st.write(", ".join(inconsist['duplicados_collar'][:200]) or "—")

    # 5. CÁLCULO E RENDERIZAÇÃO
    with st.spinner("Calculando modelo 3D (Desurvey)..."):
        df_3d = calcular_desurvey_otimizado(df_collar, df_survey, df_lito, col_map, metodo=metodo_desurvey_key)

    if not df_3d.empty:
        litos_unicas = df_3d['LITO'].unique()
        if 'lito_colors' not in st.session_state:
            cores_padrao = px.colors.qualitative.Plotly
            cores_projeto = projeto_carregado.get('cores', {}) if projeto_carregado else {}
            st.session_state.lito_colors = {
                lito: cores_projeto.get(str(lito), cores_padrao[i % len(cores_padrao)])
                for i, lito in enumerate(litos_unicas)
            }
        else:
            for i, lito in enumerate(litos_unicas):
                if lito not in st.session_state.lito_colors:
                    cores_padrao = px.colors.qualitative.Plotly
                    st.session_state.lito_colors[lito] = cores_padrao[i % len(cores_padrao)]

        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
            ["🌍 Visualizador 3D", "📐 Seção 2D", "📊 Estatísticas Profundas", "🧪 Compositagem",
             "🎨 Cores Litologia", "💾 Exportar", "🗺️ Mapas Temáticos", "⚙️ Geometalurgia"])

        with tab1:
            c1, c2, c3 = st.columns([1, 1, 1])
            furo_sel = c1.multiselect("Filtrar Furo(s)", df_3d['HOLEID'].unique(), default=[])
            tipo_cor = c2.radio("Cor:", ["Litologia", "Teor"], horizontal=True)
            espessura_fixa = c3.slider("Espessura da linha", 1, 15, 5)

            df_plot = df_3d if not furo_sel else df_3d[df_3d['HOLEID'].isin(furo_sel)]
            fig3d = go.Figure()

            if topo_grid is not None:
                X_topo, Y_topo, Z_topo = topo_grid
                fig3d.add_trace(go.Surface(x=X_topo, y=Y_topo, z=Z_topo, opacity=0.6, colorscale='earth', showscale=False, name='Topografia (raster)'))
            elif df_topo is not None and {'X', 'Y', 'Z'}.issubset(set(c.upper() for c in df_topo.columns)):
                cx = [c for c in df_topo.columns if c.upper() == 'X'][0]
                cy = [c for c in df_topo.columns if c.upper() == 'Y'][0]
                cz = [c for c in df_topo.columns if c.upper() == 'Z'][0]
                fig3d.add_trace(go.Mesh3d(x=df_topo[cx], y=df_topo[cy], z=df_topo[cz], opacity=0.3, color='saddlebrown', name='Topografia (pontos XYZ)'))
            elif len(df_collar) >= 3:
                fig3d.add_trace(go.Mesh3d(x=df_collar[col_map['c_x']], y=df_collar[col_map['c_y']], z=df_collar[col_map['c_z']], opacity=0.3, color='gray', name='Topografia (proxy: collars)'))

            if tipo_cor == "Litologia":
                for lito in df_plot['LITO'].unique():
                    sub = df_plot[df_plot['LITO'] == lito]
                    x_vals, y_vals, z_vals, texts = [], [], [], []
                    for r in sub.itertuples():
                        x_vals.extend([r.X1, r.X2, None])
                        y_vals.extend([r.Y1, r.Y2, None])
                        z_vals.extend([r.Z1, r.Z2, None])
                        txt = f"Furo: {r.HOLEID}<br>De: {r.FROM} Até: {r.TO}<br>Lito: {r.LITO}"
                        texts.extend([txt, txt, None])

                    fig3d.add_trace(go.Scatter3d(
                        x=x_vals, y=y_vals, z=z_vals, mode='lines',
                        line=dict(width=espessura_fixa, color=st.session_state.lito_colors.get(lito, 'grey')),
                        name=str(lito), text=texts, hoverinfo='text'
                    ))
            else:
                x_vals, y_vals, z_vals, texts, colors = [], [], [], [], []
                max_grade = df_plot['GRADE'].max() if (col_map['l_grade'] and df_plot['GRADE'].max() > 0) else 1
                for r in df_plot.itertuples():
                    x_vals.extend([r.X1, r.X2, None])
                    y_vals.extend([r.Y1, r.Y2, None])
                    z_vals.extend([r.Z1, r.Z2, None])
                    colors.extend([r.GRADE, r.GRADE, r.GRADE])
                    txt = f"Furo: {r.HOLEID}<br>Lito: {r.LITO}<br>Teor: {r.GRADE}"
                    texts.extend([txt, txt, None])

                fig3d.add_trace(go.Scatter3d(
                    x=x_vals, y=y_vals, z=z_vals, mode='lines',
                    line=dict(width=espessura_fixa, color=colors, colorscale='Viridis', cmin=0, cmax=max_grade),
                    text=texts, hoverinfo='text', showlegend=False
                ))

            fig3d.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectratio=dict(x=1, y=1, z=z_scale)), height=700, margin=dict(l=0, r=0, b=0, t=0))
            st.plotly_chart(fig3d, use_container_width=True)

            st.download_button("📥 Exportar Dados Visíveis para Excel", exportar_para_excel(df_plot, "Dados_3D_Filtrados"), "dados_3D_filtrados.xlsx", key="dl_3d")

        with tab2:
            st.subheader("Gerador de Seções 2D")
            c1, c2 = st.columns([1, 2])
            with c1:
                x_min, x_max = float(df_collar[col_map['c_x']].min()), float(df_collar[col_map['c_x']].max())
                y_min, y_max = float(df_collar[col_map['c_y']].min()), float(df_collar[col_map['c_y']].max())
                p1_x = st.slider("Ponto A (X)", x_min, x_max, x_min)
                p1_y = st.slider("Ponto A (Y)", y_min, y_max, y_min)
                p2_x = st.slider("Ponto B (X)", x_min, x_max, x_max)
                p2_y = st.slider("Ponto B (Y)", y_min, y_max, y_max)
                tolerancia = st.number_input("Tolerância Envelope (m)", min_value=1.0, value=50.0)

            with c2:
                fig_mapa = go.Figure()
                fig_mapa.add_trace(go.Scatter(x=df_collar[col_map['c_x']], y=df_collar[col_map['c_y']], mode='markers', text=df_collar[col_map['c_id']], name="Furos"))
                fig_mapa.add_trace(go.Scatter(x=[p1_x, p2_x], y=[p1_y, p2_y], mode='lines', line=dict(color='red', width=3, dash='dash'), name="Corte"))
                fig_mapa.update_layout(height=400, yaxis=dict(scaleanchor="x", scaleratio=1))
                st.plotly_chart(fig_mapa, use_container_width=True)

            p1, p2 = np.array([p1_x, p1_y]), np.array([p2_x, p2_y])
            linha_vetor = p2 - p1
            compr_linha = np.linalg.norm(linha_vetor)
            fig_2d = go.Figure()
            dados_secao = []

            if compr_linha > 0:
                dir_vetor = linha_vetor / compr_linha
                vetor_normal = np.array([-dir_vetor[1], dir_vetor[0]])

                for lito in df_3d['LITO'].unique():
                    sub = df_3d[df_3d['LITO'] == lito]
                    x_proj, z_proj, txts = [], [], []

                    for r in sub.itertuples():
                        p1_xy, p2_xy = np.array([r.X1, r.Y1]), np.array([r.X2, r.Y2])
                        if abs(np.dot(p1_xy - p1, vetor_normal)) <= tolerancia or abs(np.dot(p2_xy - p1, vetor_normal)) <= tolerancia:
                            proj_x1 = np.dot(p1_xy - p1, dir_vetor)
                            proj_x2 = np.dot(p2_xy - p1, dir_vetor)
                            x_proj.extend([proj_x1, proj_x2, None])
                            z_proj.extend([r.Z1, r.Z2, None])
                            t = f"{r.HOLEID} ({r.LITO})"
                            txts.extend([t, t, None])

                            dados_secao.append({
                                'HOLEID': r.HOLEID, 'LITO': r.LITO, 'GRADE': r.GRADE,
                                'DIST_INICIAL': proj_x1, 'Z_INICIAL': r.Z1,
                                'DIST_FINAL': proj_x2, 'Z_FINAL': r.Z2
                            })

                    if x_proj:
                        fig_2d.add_trace(go.Scatter(
                            x=x_proj, y=z_proj, mode='lines', line=dict(color=st.session_state.lito_colors.get(lito, 'grey'), width=6),
                            text=txts, hoverinfo='text', name=str(lito)
                        ))

            fig_2d.update_layout(xaxis_title="Distância (m)", yaxis_title="Elevação (Z)", yaxis=dict(scaleanchor="x", scaleratio=1), height=500)
            st.plotly_chart(fig_2d, use_container_width=True)

            if dados_secao:
                st.download_button("📥 Exportar Dados da Seção para Excel", exportar_para_excel(pd.DataFrame(dados_secao), "Dados_Secao_2D"), "dados_secao_2d.xlsx", key="dl_2d")

        with tab3:
            st.subheader("📊 Análise Exploratória Profunda (EDA)")
            df_3d['COMPR'] = df_3d['TO'] - df_3d['FROM']

            eda1, eda2, eda3, eda4 = st.tabs(["1. Resumo & Qualidade", "2. Análise Espacial (Geometria)", "3. Geologia & Teores", "4. QA Survey / Furo-a-furo"])

            with eda1:
                st.markdown("##### 🔍 Funil de Dados (Furos Perdidos)")
                col_f1, col_f2, col_f3, col_f4 = st.columns(4)
                unicos_collar = df_collar[col_map['c_id']].nunique()
                unicos_survey = df_survey[col_map['s_id']].nunique()
                unicos_lito = df_lito[col_map['l_id']].nunique()
                unicos_3d = df_3d['HOLEID'].nunique()

                col_f1.metric("Furos no Collar", unicos_collar)
                col_f2.metric("Furos no Survey", unicos_survey)
                col_f3.metric("Furos na Lito", unicos_lito)
                col_f4.metric("Desenhados no 3D", unicos_3d, delta=f"{unicos_3d - unicos_collar} furos", delta_color="inverse")

                furos_originais = set(df_collar[col_map['c_id']].astype(str).unique())
                furos_gerados = set(df_3d['HOLEID'].astype(str).unique())
                furos_perdidos = furos_originais - furos_gerados
                if furos_perdidos:
                    with st.expander(f"⚠️ Ver lista dos {len(furos_perdidos)} furos que não foram desenhados"):
                        st.write("Motivos comuns: Falta de descrição de litologia na tabela Lito ou falha grave nas coordenadas (X, Y, Z).")
                        st.write(", ".join(sorted(list(furos_perdidos))))

                st.markdown("---")
                st.markdown("##### 📏 Metragem e Volumetria")
                m1, m2 = st.columns(2)
                m1.metric("Total de Furos Validados", df_3d['HOLEID'].nunique())
                m2.metric("Metragem Total Interpretada", f"{df_3d['COMPR'].sum():.2f} m")

                lito_soma = df_3d.groupby('LITO')['COMPR'].sum().reset_index().sort_values('COMPR', ascending=False)
                c_d1, c_d2 = st.columns(2)
                with c_d1:
                    st.plotly_chart(px.bar(lito_soma, x='LITO', y='COMPR', color='LITO', title="Metragem por Litologia", color_discrete_map=st.session_state.lito_colors), use_container_width=True)
                with c_d2:
                    st.plotly_chart(px.pie(lito_soma, names='LITO', values='COMPR', color='LITO', title="Proporção da Geologia", color_discrete_map=st.session_state.lito_colors), use_container_width=True)

            with eda2:
                st.markdown("##### 🧭 Geometria Espacial da Campanha de Sondagem")
                max_depths = df_3d.groupby('HOLEID')['TO'].max().reset_index()
                df_mapa = df_collar.merge(max_depths, left_on=col_map['c_id'], right_on='HOLEID', how='inner')

                c_esp1, c_esp2 = st.columns(2)
                with c_esp1:
                    fig_mapa2 = px.scatter(df_mapa, x=col_map['c_x'], y=col_map['c_y'], color='TO', title="Visão em Planta (Heatmap de Profundidade)", hover_data=[col_map['c_id']], color_continuous_scale='Viridis')
                    fig_mapa2.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1))
                    st.plotly_chart(fig_mapa2, use_container_width=True)
                    st.download_button("📥 Exportar Mapa para Excel", exportar_para_excel(df_mapa, "Mapa_Profundidade"), "mapa_profundidade.xlsx", key="dl_mapa")

                with c_esp2:
                    st.plotly_chart(px.histogram(max_depths, x='TO', nbins=30, title="Distribuição das Profundidades Finais (m)", labels={'TO': 'Profundidade Final (m)', 'count': 'Qtd de Furos'}), use_container_width=True)

                st.markdown("---")
                st.markdown("##### 📌 Outliers de coordenadas do Collar")
                out_coords = detectar_outliers_coordenadas(df_collar, col_map['c_x'], col_map['c_y'], col_map['c_z'])
                if out_coords.empty:
                    st.success("Nenhum outlier de coordenada acima do limiar (Z-score > 4).")
                else:
                    st.warning(f"{len(out_coords)} valor(es) de coordenada fora do padrão — confira possíveis erros de digitação/unidade.")
                    st.dataframe(out_coords, use_container_width=True)

            with eda3:
                st.markdown("##### 💎 Análise de Teores e Comprimento de Amostras")
                st.plotly_chart(px.histogram(df_3d, x='COMPR', nbins=50, title="Comprimento das Amostras (QA/QC)", labels={'COMPR': 'Tamanho da Amostra (m)'}), use_container_width=True)

                if col_map['l_grade']:
                    st.plotly_chart(px.box(df_3d, x='LITO', y='GRADE', color='LITO', title="Distribuição de Teor por Rocha", color_discrete_map=st.session_state.lito_colors), use_container_width=True)
                    c_teo3, c_teo4 = st.columns(2)
                    with c_teo3:
                        fig_scatter = px.scatter(df_3d, x='GRADE', y='Z1', color='LITO', title="Teor vs. Elevação (Z)", labels={'Z1': 'Elevação / Cota Z', 'GRADE': 'Teor'}, color_discrete_map=st.session_state.lito_colors, opacity=0.7)
                        st.plotly_chart(fig_scatter, use_container_width=True)
                    with c_teo4:
                        fig_ecdf = px.ecdf(df_3d, x='GRADE', color='LITO', title="Curva de Distribuição Acumulada", color_discrete_map=st.session_state.lito_colors)
                        st.plotly_chart(fig_ecdf, use_container_width=True)

                    st.markdown("---")
                    st.markdown("##### 📈 Curva Teor x Tonelagem (aproximada)")
                    st.caption("Tonelagem aproximada pelo comprimento amostrado × densidade informada. Para tonelagem real, use um modelo de blocos com volumes por domínio.")
                    densidade = st.number_input("Densidade (t/m³)", min_value=0.1, value=2.7, step=0.1)
                    grade_validos = df_3d['GRADE'].dropna()
                    if not grade_validos.empty:
                        cutoffs = np.linspace(grade_validos.min(), grade_validos.quantile(0.98), 40)
                        tonelagens, teores_medios = [], []
                        for co in cutoffs:
                            acima = df_3d[df_3d['GRADE'] >= co]
                            ton = acima['COMPR'].sum() * densidade
                            teor_medio = np.average(acima['GRADE'], weights=acima['COMPR']) if len(acima) > 0 else 0
                            tonelagens.append(ton)
                            teores_medios.append(teor_medio)
                        fig_gt = make_subplots(specs=[[{"secondary_y": True}]])
                        fig_gt.add_trace(go.Scatter(x=cutoffs, y=tonelagens, name="Tonelagem (t)", line=dict(color='steelblue')), secondary_y=False)
                        fig_gt.add_trace(go.Scatter(x=cutoffs, y=teores_medios, name="Teor Médio", line=dict(color='firebrick')), secondary_y=True)
                        fig_gt.update_layout(title="Curva Teor x Tonelagem", xaxis_title="Cutoff de Teor")
                        fig_gt.update_yaxes(title_text="Tonelagem aproximada (t)", secondary_y=False)
                        fig_gt.update_yaxes(title_text="Teor Médio Acima do Cutoff", secondary_y=True)
                        st.plotly_chart(fig_gt, use_container_width=True)

            with eda4:
                st.markdown("##### 📡 QA de Survey — Dip/Azimute por Profundidade")
                furo_qa = st.selectbox("Selecione um furo", sorted(df_survey[col_map['s_id']].astype(str).unique()))
                sub_survey = df_survey[df_survey[col_map['s_id']].astype(str) == furo_qa].sort_values(col_map['s_depth'])
                if not sub_survey.empty:
                    fig_qa = make_subplots(rows=1, cols=2, subplot_titles=("Dip vs Profundidade", "Azimute vs Profundidade"))
                    fig_qa.add_trace(go.Scatter(x=pd.to_numeric(sub_survey[col_map['s_dip']], errors='coerce'), y=pd.to_numeric(sub_survey[col_map['s_depth']], errors='coerce'), mode='lines+markers', name='Dip'), row=1, col=1)
                    fig_qa.add_trace(go.Scatter(x=pd.to_numeric(sub_survey[col_map['s_az']], errors='coerce'), y=pd.to_numeric(sub_survey[col_map['s_depth']], errors='coerce'), mode='lines+markers', name='Azimute'), row=1, col=2)
                    fig_qa.update_yaxes(autorange='reversed', title_text='Profundidade (m)')
                    fig_qa.update_layout(height=450, showlegend=False)
                    st.plotly_chart(fig_qa, use_container_width=True)

                st.markdown("##### 📉 Perfil de Teor por Profundidade (furo a furo)")
                if col_map['l_grade']:
                    furo_perfil = st.selectbox("Selecione um furo para o perfil de teor", sorted(df_3d['HOLEID'].unique()), key="furo_perfil")
                    sub_3d = df_3d[df_3d['HOLEID'] == furo_perfil].copy()
                    sub_3d['MEIO'] = (sub_3d['FROM'] + sub_3d['TO']) / 2
                    fig_perfil = go.Figure()
                    fig_perfil.add_trace(go.Scatter(x=sub_3d['GRADE'], y=sub_3d['MEIO'], mode='lines+markers', line=dict(color='firebrick')))
                    fig_perfil.update_yaxes(autorange='reversed', title_text='Profundidade (m)')
                    fig_perfil.update_xaxes(title_text='Teor')
                    fig_perfil.update_layout(height=450, title=f"Perfil de Teor — {furo_perfil}")
                    st.plotly_chart(fig_perfil, use_container_width=True)
                else:
                    st.info("Selecione uma coluna de Teor no mapeamento lateral para habilitar este gráfico.")

        with tab4:
            st.subheader("🧪 Compositagem em Intervalos Regulares")
            comprimento_comp = st.number_input("Comprimento do composito (m)", min_value=0.1, value=1.0, step=0.1)
            if st.button("Gerar Compositos"):
                with st.spinner("Compositando..."):
                    df_comp = compositar_intervalos(df_3d, comprimento=comprimento_comp)
                st.session_state['df_comp'] = df_comp
            if 'df_comp' in st.session_state and not st.session_state['df_comp'].empty:
                df_comp = st.session_state['df_comp']
                st.dataframe(df_comp.head(50), use_container_width=True)
                if col_map['l_grade']:
                    st.plotly_chart(px.histogram(df_comp, x='GRADE_COMPOSITO', nbins=40, title="Distribuição do Teor Compositado"), use_container_width=True)
                st.download_button("📥 Baixar Compositos (Excel)", exportar_para_excel(df_comp, "Compositos"), "compositos.xlsx", key="dl_comp")

        with tab5:
            st.subheader("Cores")
            cols_cor = st.columns(4)
            for i, lito in enumerate(litos_unicas):
                with cols_cor[i % 4]:
                    st.session_state.lito_colors[lito] = st.color_picker(f"{lito}", st.session_state.lito_colors.get(lito, '#808080'))

        with tab6:
            st.subheader("Exportar Banco de Dados 3D")
            st.download_button("📥 Baixar Tabela 3D Completa (Excel)", exportar_para_excel(df_3d, "BD_3D"), 'bd_3D.xlsx')
            st.dataframe(df_3d.head(20), use_container_width=True)

            st.markdown("---")
            st.markdown("##### 🔗 Exportação para CAD de Mineração (formatos genéricos)")
            st.caption("Pontos de partida para importação no Deswik, Datamine Studio ou Surpac — confira sempre o wizard de importação da sua versão.")
            ce1, ce2, ce3 = st.columns(3)
            with ce1:
                st.download_button("📥 DXF (polylines 3D)", exportar_dxf_polylines(df_3d), "furos_3d.dxf", key="dl_dxf")
            with ce2:
                st.download_button("📥 Pontos genéricos (Excel)", exportar_para_excel(exportar_pontos_genericos(df_3d), "Pontos"), "pontos_furos.xlsx", key="dl_pontos")
            with ce3:
                st.download_button("📥 String file (aprox. Surpac)", exportar_surpac_str(df_3d), "furos.str", key="dl_str")

            st.markdown("---")
            st.markdown("##### 💾 Salvar configuração do projeto")
            cores_serializaveis = {str(k): v for k, v in st.session_state.lito_colors.items()}
            st.download_button("📥 Baixar mapeamento + cores (.json)",
                                salvar_projeto(col_map, cores_serializaveis, [], []),
                                "projeto_geologia.json", key="dl_projeto")

        with tab7:
            st.subheader("🗺️ Gerador de Mapas Temáticos e Filtros Cruzados")
            st.markdown("Filtre as amostras por Múltiplas Categorias para criar mapas específicos.")

            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                sel_litos = st.multiselect("Selecione Litologias", litos_unicas, default=litos_unicas)
            with f_col2:
                sel_furos = st.multiselect("Selecione Furos", df_3d['HOLEID'].unique(), default=[])
            with f_col3:
                z_min, z_max = float(df_3d['Z2'].min()), float(df_3d['Z1'].max())
                sel_z = st.slider("Filtro de Elevação (Cota Z)", z_min, z_max, (z_min, z_max))

            df_tema = df_3d[df_3d['LITO'].isin(sel_litos)]
            if sel_furos:
                df_tema = df_tema[df_tema['HOLEID'].isin(sel_furos)]
            df_tema = df_tema[(df_tema['Z1'] >= sel_z[0]) & (df_tema['Z2'] <= sel_z[1])]

            if col_map['l_grade']:
                grade_min, grade_max = float(df_3d['GRADE'].min()), float(df_3d['GRADE'].max())
                sel_grade = st.slider("Cut-off de Teor", grade_min, grade_max, grade_min)
                df_tema = df_tema[df_tema['GRADE'] >= sel_grade]

            if df_tema.empty:
                st.warning("Nenhum dado encontrado para os filtros selecionados.")
            else:
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    fig_tema1 = px.scatter(df_tema, x='X1', y='Y1', color='LITO', title="Projeção em Planta (X vs Y)", hover_data=['HOLEID', 'GRADE'], color_discrete_map=st.session_state.lito_colors)
                    fig_tema1.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1), xaxis_title="Easting (X)", yaxis_title="Northing (Y)")
                    st.plotly_chart(fig_tema1, use_container_width=True)

                with m_col2:
                    fig_tema2 = px.scatter(df_tema, x='X1', y='Z1', color='LITO', title="Projeção Perfil Leste-Oeste (X vs Z)", hover_data=['HOLEID', 'GRADE'], color_discrete_map=st.session_state.lito_colors)
                    fig_tema2.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1), xaxis_title="Easting (X)", yaxis_title="Elevação (Z)")
                    st.plotly_chart(fig_tema2, use_container_width=True)

            st.download_button("📥 Baixar Dados Deste Mapa Temático (Excel)", exportar_para_excel(df_tema, "Mapa_Tematico"), "mapa_tematico_filtrado.xlsx", key="dl_tema")

        with tab8:
            st.header("⚙️ Modelagem Geometalúrgica do Ouro (Machine Learning)")
            st.markdown("Implementação interativa de fluxos preditivos e análise estatística dinâmica.")
            
            if not SKLEARN_DISPONIVEL or 'sm' not in globals():
                st.error("⚠️ As bibliotecas scikit-learn, scipy e statsmodels não estão instaladas. Execute `pip install scikit-learn scipy statsmodels` para utilizar esta aba.")
            else:
                # ---------------------------------------------------------
                # 1. ADICIONAR PLANILHAS EXTRAS (MESCLAGEM GEOMETALÚRGICA)
                # ---------------------------------------------------------
                st.subheader("📥 1. Mesclar Planilhas Geometalúrgicas Extras")
                st.info("Caso possua dados adicionais (ex: Wi, Recuperação), faça o upload aqui. O sistema usará as colunas de ID (Furo), De (From) e Até (To) para unir à base principal.")
                
                arq_extra = st.file_uploader("Upload da Planilha Extra", type=['csv', 'xlsx'], key="up_extra_geomet")
                
                if 'df_geomet_base' not in st.session_state:
                    st.session_state.df_geomet_base = df_lito.copy()
                
                if arq_extra:
                    try:
                        df_extra = load_final(arq_extra.getvalue(), arq_extra.name, 0)
                        
                        id_ext = identificar_coluna(df_extra, 'id')
                        from_ext = identificar_coluna(df_extra, 'from')
                        to_ext = identificar_coluna(df_extra, 'to')
                        
                        if id_ext and from_ext and to_ext:
                            df_extra.rename(columns={id_ext: col_map['l_id'], from_ext: col_map['l_from'], to_ext: col_map['l_to']}, inplace=True)
                            
                            st.session_state.df_geomet_base = pd.merge(
                                st.session_state.df_geomet_base, 
                                df_extra, 
                                on=[col_map['l_id'], col_map['l_from'], col_map['l_to']], 
                                how='left'
                            )
                            st.success(f"Planilha extra combinada com sucesso! Novas colunas adicionadas à base.")
                        else:
                            st.warning("⚠️ Não foi possível identificar automaticamente as colunas de Furo, De, Até na planilha extra para mesclar.")
                    except Exception as e:
                        st.error(f"Erro ao processar planilha extra: {e}")

                # ---------------------------------------------------------
                # 1.5. ENGENHARIA DE FEATURES (RAZÕES)
                # ---------------------------------------------------------
                df_ml = st.session_state.df_geomet_base.copy()
                colunas_numericas_disp = df_ml.select_dtypes(include=np.number).columns.tolist()
                
                st.markdown("---")
                st.subheader("🧮 2. Criação de Novas Variáveis (Razões)")
                st.info("Crie razões entre elementos (ex: Cu / S) para atuar como proxies mineralógicos no seu depósito.")
                
                c_num, c_den, c_btn = st.columns([2, 2, 1])
                with c_num:
                    num_col = st.selectbox("Numerador", colunas_numericas_disp, key="num_ratio")
                with c_den:
                    den_col = st.selectbox("Denominador", colunas_numericas_disp, key="den_ratio")
                with c_btn:
                    st.write("") 
                    if st.button("Criar Razão"):
                        nome_razao = f"{num_col}/{den_col}"
                        st.session_state.df_geomet_base[nome_razao] = np.where(
                            st.session_state.df_geomet_base[den_col] != 0, 
                            st.session_state.df_geomet_base[num_col] / st.session_state.df_geomet_base[den_col], 
                            np.nan
                        )
                        st.success(f"Razão `{nome_razao}` adicionada ao banco de dados!")
                        st.rerun()
                
                df_ml = st.session_state.df_geomet_base.copy()
                colunas_numericas_disp = df_ml.select_dtypes(include=np.number).columns.tolist()

                # ---------------------------------------------------------
                # 2. SELEÇÃO DINÂMICA DE VARIÁVEIS PARA ESTATÍSTICA
                # ---------------------------------------------------------
                st.markdown("---")
                st.subheader("📊 3. Estatística Descritiva e Assimetria")
                
                cols_estat = st.multiselect(
                    "Selecione as variáveis para calcular Média, Variância, Desvio Padrão, Mediana, Quartis e CV:", 
                    colunas_numericas_disp, 
                    default=[c for c in [col_map.get('l_grade', '')] if c in colunas_numericas_disp]
                )
                
                if cols_estat:
                    stats = []
                    for c in cols_estat:
                        serie = df_ml[c].dropna()
                        n = len(serie)
                        if n > 0:
                            mean_v = serie.mean()
                            var_v = serie.var()   
                            std_v = serie.std()   
                            median_v = serie.median() 
                            q1_v = serie.quantile(0.25) 
                            q3_v = serie.quantile(0.75) 
                            cv_v = (std_v / mean_v) * 100 if mean_v != 0 else np.nan 
                            skew_v = serie.skew() 
                            
                            stats.append({
                                'Variável': c,
                                'N Amostras': n,
                                'Média (x̄)': mean_v,
                                'Variância (s²)': var_v,
                                'Desvio Padr. (s)': std_v,
                                'CV (%)': cv_v,
                                'Q1 (25%)': q1_v,
                                'Mediana (50%)': median_v,
                                'Q3 (75%)': q3_v,
                                'Assimetria': skew_v
                            })
                    
                    df_stats_vis = pd.DataFrame(stats).set_index('Variável')
                    st.dataframe(df_stats_vis.style.format("{:.3f}"), use_container_width=True)

                    var_grafico = st.selectbox("Selecione a variável para inspecionar no gráfico didático:", cols_estat)
                    
                    if var_grafico:
                        s_plot = df_ml[var_grafico].dropna()
                        media_plot = s_plot.mean()
                        mediana_plot = s_plot.median()
                        assimetria_plot = s_plot.skew()
                        
                        fig_box = px.box(df_ml, x=var_grafico, title=f"Boxplot de {var_grafico} (Quartis e Outliers)", color_discrete_sequence=['#ff7f0e'])
                        fig_box.update_layout(height=300)
                        st.plotly_chart(fig_box, use_container_width=True)

                        fig_dist = px.histogram(df_ml, x=var_grafico, title=f"Histograma de {var_grafico} (Assimetria: {assimetria_plot:.2f})", color_discrete_sequence=['#1f77b4'], nbins=40)
                        fig_dist.add_vline(x=media_plot, line_dash="dash", line_color="red", annotation_text=f"Média: {media_plot:.2f}")
                        fig_dist.add_vline(x=mediana_plot, line_dash="dash", line_color="green", annotation_text=f"Mediana: {mediana_plot:.2f}")
                        fig_dist.update_layout(height=400)
                        st.plotly_chart(fig_dist, use_container_width=True)

                # ---------------------------------------------------------
                # 3. CORRELAÇÃO E SELEÇÃO DE FEATURES
                # ---------------------------------------------------------
                st.markdown("---")
                st.subheader("🔗 4. Matriz de Correlação Personalizada")

                c_rend = col_map.get('l_rendimento', 'Nenhuma')
                c_rec_au = col_map.get('l_rec_au', 'Nenhuma')
                c_au = col_map.get('l_grade', 'Nenhuma')

                if c_rend in [None, 'Nenhuma'] or c_rec_au in [None, 'Nenhuma'] or c_au in [None, 'Nenhuma']:
                    st.warning("⚠️ Configure o Rendimento, a Recuperação e o Teor Principal na barra lateral para prosseguir.")
                else:
                    df_ml['Au_Conc'] = np.where(
                        (df_ml[c_rend] > 0) & (df_ml[c_rend].notna()), 
                        (df_ml[c_rec_au] * df_ml[c_au]) / df_ml[c_rend], 
                        np.nan
                    )
                    
                    alvo_selecionado = st.radio("Selecione o Alvo (Target) para analisar correlações:", [c_rec_au, c_rend, 'Au_Conc'], horizontal=True)

                    vars_correlacao = st.multiselect(
                        "Selecione as variáveis (inclusive as suas razões) para montar a Matriz de Correlação:", 
                        colunas_numericas_disp,
                        default=[c for c in ['Au_gpt', 'S_pct', 'Fe_pct', 'Cu_pct'] if c in colunas_numericas_disp]
                    )

                    if vars_correlacao and alvo_selecionado in df_ml.columns:
                        col_corr = [c for c in vars_correlacao if c != alvo_selecionado] + [alvo_selecionado]
                        # Somente linhas com valor real no alvo e nas variáveis (sem imputar com a mediana)
                        df_ml_corr = (df_ml[col_corr]
                                      .replace([np.inf, -np.inf], np.nan)
                                      .dropna()
                                      .reset_index(drop=True))
                        st.caption(f"Amostras usadas na correlação: {len(df_ml_corr)}")
                        if len(df_ml_corr) < 10:
                            st.warning("Poucas amostras com valores reais em todas as variáveis selecionadas; a correlação não é confiável.")

                        matriz_corr = df_ml_corr.corr()
                        
                        mask = np.triu(np.ones_like(matriz_corr, dtype=bool), k=1)
                        matriz_tri = matriz_corr.mask(mask)
                        
                        styled_corr = matriz_tri.style.background_gradient(cmap='RdBu', vmin=-1, vmax=1, axis=None).format("{:.2f}", na_rep="")
                        
                        st.markdown("**Matriz de Correlação**")
                        st.dataframe(styled_corr, use_container_width=True, height=600)
                        
                        target_corr = matriz_corr[alvo_selecionado].drop(alvo_selecionado).dropna()
                        candidates = target_corr[abs(target_corr) >= 0.3].index.tolist()
                        if candidates:
                            st.success(f"💡 **Variáveis Candidatas sugeridas** (Correlação moderada/forte > |0.3| com {alvo_selecionado}): **{', '.join(candidates)}**")
                        else:
                            st.info(f"💡 Nenhuma variável apresentou correlação maior que |0.3| com {alvo_selecionado}.")

                    features = st.multiselect(
                        "Selecione as variáveis Explicativas finais para os Modelos Preditivos (Regressão / K-Means):", 
                        colunas_numericas_disp,
                        default=candidates if 'candidates' in locals() and candidates else []
                    )
                    
                    # ---------------------------------------------------------
                    # 4. REGRESSÃO E MACHINE LEARNING
                    # ---------------------------------------------------------
                    if st.button("🚀 Treinar Modelos (Regressão Excel, W1, W2, W3)") and features:
                        with st.spinner("Realizando regressão multivariada e clusterização..."):
                            
                            col_processamento = features + [alvo_selecionado]
                            # Usa SOMENTE linhas com ensaio real (sem imputar o alvo com a mediana)
                            df_ml_imp = (df_ml[col_processamento + [col_map['l_id']]]
                                         .replace([np.inf, -np.inf], np.nan)
                                         .dropna()
                                         .reset_index(drop=True))

                            X = df_ml_imp[features]
                            y = df_ml_imp[alvo_selecionado]
                            grupos = df_ml_imp[col_map['l_id']].astype(str).values

                            n_splits_cv = min(5, len(np.unique(grupos)))
                            st.caption(f"Amostras usadas no modelo: {len(df_ml_imp)} (linhas sem ensaio foram descartadas) | Furos: {len(np.unique(grupos))}")
                            if len(df_ml_imp) < 10 or n_splits_cv < 2:
                                st.error("Amostras ou furos insuficientes com valores reais nas variáveis e no alvo selecionados para treinar e validar o modelo.")
                                st.stop()
                            
                            # ========================================================
                            # REGRESSÃO MÚLTIPLA OLS (ESTILO EXCEL)
                            # ========================================================
                            st.markdown("---")
                            st.markdown("#### Regressão Linear Multivariada (OLS - Estilo Excel)")
                            
                            X_ols = sm.add_constant(X)
                            modelo_ols = sm.OLS(y, X_ols).fit()
                            
                            alvo_formatado = alvo_selecionado.replace('_', r'\_')
                            equacao = f"\\text{{{alvo_formatado}}} = {modelo_ols.params.iloc[0]:.4f}"
                            
                            for i, feat in enumerate(features):
                                coef = modelo_ols.params.iloc[i+1]
                                sinal = "+" if coef >= 0 else "-"
                                feat_formatada = feat.replace('_', r'\_')
                                equacao += f" {sinal} {abs(coef):.4f} \\cdot \\text{{{feat_formatada}}}"
                                
                            equacao += " + u_i"
                            
                            st.markdown("A equação preditiva baseada nos coeficientes da regressão multivariada ($\\beta$) calculados:")
                            st.latex(equacao)
                            
                            resumo_reg = pd.DataFrame({
                                'Estatística de regressão': ['R múltiplo (R)', 'R-Quadrado (R²)', 'R-quadrado ajustado', 'Erro padrão', 'Observações'],
                                'Valor': [
                                    np.sqrt(modelo_ols.rsquared) if modelo_ols.rsquared > 0 else 0,
                                    modelo_ols.rsquared,
                                    modelo_ols.rsquared_adj,
                                    np.sqrt(modelo_ols.mse_resid),
                                    modelo_ols.nobs
                                ]
                            })
                            
                            tabela_coef = pd.DataFrame({
                                'Coeficientes': modelo_ols.params,
                                'Erro padrão': modelo_ols.bse,
                                'Stat t': modelo_ols.tvalues,
                                'valor-P': modelo_ols.pvalues,
                                'Inferior 95%': modelo_ols.conf_int()[0],
                                'Superior 95%': modelo_ols.conf_int()[1]
                            })
                            tabela_coef.index.name = ''
                            
                            col_t1, col_t2 = st.columns([1, 2])
                            with col_t1:
                                st.markdown("**Resumo da Regressão**")
                                st.dataframe(resumo_reg.style.format({'Valor': '{:.4f}'}))
                            with col_t2:
                                st.markdown("**Tabela de Coeficientes (ANOVA)**")
                                st.dataframe(tabela_coef.style.format("{:.4f}").apply(lambda x: ['background: lightgreen' if v < 0.05 else 'background: lightcoral' for v in x], subset=['valor-P']))
                            
                            st.caption("Nota: Células verdes na coluna 'valor-P' indicam que a variável é estatisticamente significante (p < 0.05).")

                            # ========================================================
                            # ML - WORKFLOWS
                            # ========================================================
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(X)
                            
                            lasso_model = Lasso(alpha=0.1, max_iter=10000)
                            
                            resultados_comparativos = []
                            
                            # W1: Regressão Direta
                            # Validação cruzada AGRUPADA POR FURO: amostras do mesmo furo nunca ficam
                            # simultaneamente no treino e no teste (evita resultado otimista).
                            cv = GroupKFold(n_splits=n_splits_cv)
                            cv_w1 = cross_validate(lasso_model, X_scaled, y, cv=cv, groups=grupos, scoring=('neg_root_mean_squared_error', 'neg_mean_absolute_error', 'r2'))
                            resultados_comparativos.append({
                                'Workflow': 'W1: Regressão Direta (Lasso)',
                                'RMSE': -cv_w1['test_neg_root_mean_squared_error'].mean(),
                                'MAE': -cv_w1['test_neg_mean_absolute_error'].mean(),
                                'R²': cv_w1['test_r2'].mean()
                            })
                            
                            # W2: Hierárquico + Lasso
                            hc = AgglomerativeClustering(n_clusters=3, linkage='ward', metric='euclidean')
                            clusters_hc = hc.fit_predict(X_scaled)
                            
                            rmse_w2, mae_w2, r2_w2, pesos_w2 = [], [], [], []
                            for c in np.unique(clusters_hc):
                                mask = (clusters_hc == c)
                                n_grp_c = len(np.unique(grupos[mask]))
                                if mask.sum() > 30 and n_grp_c >= 2:
                                    cv_c = cross_validate(lasso_model, X_scaled[mask], y[mask], cv=GroupKFold(n_splits=min(5, n_grp_c)), groups=grupos[mask], scoring=('neg_root_mean_squared_error', 'neg_mean_absolute_error', 'r2'))
                                    rmse_w2.append(-cv_c['test_neg_root_mean_squared_error'].mean())
                                    mae_w2.append(-cv_c['test_neg_mean_absolute_error'].mean())
                                    r2_w2.append(cv_c['test_r2'].mean())
                                    pesos_w2.append(mask.sum())
                                    
                            if pesos_w2:
                                resultados_comparativos.append({
                                    'Workflow': 'W2: Cluster Hierárquico',
                                    'RMSE': np.average(rmse_w2, weights=pesos_w2),
                                    'MAE': np.average(mae_w2, weights=pesos_w2),
                                    'R²': np.average(r2_w2, weights=pesos_w2)
                                })
                            
                            # W3: PCA + K-Means + Lasso
                            pca = PCA(n_components=min(len(features), 3))
                            X_pca = pca.fit_transform(X_scaled)
                            
                            kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
                            clusters_km = kmeans.fit_predict(X_pca)
                            
                            rmse_w3, mae_w3, r2_w3, pesos_w3 = [], [], [], []
                            for c in np.unique(clusters_km):
                                mask = (clusters_km == c)
                                n_grp_k = len(np.unique(grupos[mask]))
                                if mask.sum() > 30 and n_grp_k >= 2:
                                    cv_k = cross_validate(lasso_model, X_scaled[mask], y[mask], cv=GroupKFold(n_splits=min(5, n_grp_k)), groups=grupos[mask], scoring=('neg_root_mean_squared_error', 'neg_mean_absolute_error', 'r2'))
                                    rmse_w3.append(-cv_k['test_neg_root_mean_squared_error'].mean())
                                    mae_w3.append(-cv_k['test_neg_mean_absolute_error'].mean())
                                    r2_w3.append(cv_k['test_r2'].mean())
                                    pesos_w3.append(mask.sum())
                                    
                            if pesos_w3:
                                resultados_comparativos.append({
                                    'Workflow': 'W3: PCA + K-Means',
                                    'RMSE': np.average(rmse_w3, weights=pesos_w3),
                                    'MAE': np.average(mae_w3, weights=pesos_w3),
                                    'R²': np.average(r2_w3, weights=pesos_w3)
                                })
                                
                            # ========================================================
                            # COMPARATIVO FINAL - GRÁFICOS SEPARADOS
                            # ========================================================
                            st.markdown("---")
                            st.subheader("📊 Comparação dos Modelos (Resultados Individuais)")
                            df_resultados = pd.DataFrame(resultados_comparativos)
                            
                            st.dataframe(df_resultados.set_index('Workflow').style.format("{:.4f}"), use_container_width=True)
                            
                            st.markdown("#### Gráficos Individuais por Workflow")
                            st.info("Como os modelos de cluster (W2 e W3) podem apresentar erros gigantescos (R² negativo) devido à falta de dados, os gráficos foram separados para que a escala de um não esmague o outro.")
                            
                            col_charts = st.columns(len(df_resultados))
                            
                            for i, row in df_resultados.iterrows():
                                with col_charts[i]:
                                    df_plot = pd.DataFrame({
                                        'Métrica': ['RMSE', 'MAE', 'R²'],
                                        'Valor': [row['RMSE'], row['MAE'], row['R²']]
                                    })
                                    
                                    fig = px.bar(df_plot, x='Métrica', y='Valor', color='Métrica', 
                                                 title=f"<b>{row['Workflow']}</b>", text_auto='.3s')
                                    fig.update_layout(showlegend=False, height=400)
                                    fig.add_hline(y=0, line_width=2, line_color="black")
                                    
                                    st.plotly_chart(fig, use_container_width=True)