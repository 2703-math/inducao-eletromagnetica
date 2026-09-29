"""
🧲 Eletromagnetismo Visual — Indução Eletromagnética em simulações interativas
==============================================================================
Executar com:  streamlit run inducao_eletromagnetica.py

Estrutura didática (Prever → Observar → Explicar):
  1. Oersted   – corrente gera campo magnético (bússola, regra da mão direita, B ~ 1/r)
  2. Faraday   – ímã + bobina com fluxo, FEM, voltímetro e Lei de Lenz calculados de verdade
  3. NFC       – a mesma física aplicada: distância × tensão induzida × chip liga/desliga
  4. Ondas     – antena, campos E e B, e o comprimento de onda de serviços reais

Todos os números vêm de modelos físicos (dipolo magnético, Biot-Savart simplificado);
nada de "curvas desenhadas à mão".
"""
import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ============================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================
st.set_page_config(page_title="Física Visual: Eletromagnetismo", page_icon="🧲", layout="wide")

st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 800; color: #0f172a; text-align: center; margin-bottom: 0.3rem; }
    .subtitle   { font-size: 1.1rem; color: #64748b; text-align: center; margin-bottom: 1.5rem; }
    .concept-card { background: #f8fafc; border-radius: 12px; padding: 1.2rem; border-left: 4px solid #3b82f6;
                    margin-bottom: 1rem; color: #334155; }
    .alert-card   { background: #fffbeb; border-radius: 12px; padding: 1.2rem; border-left: 4px solid #f59e0b;
                    margin-bottom: 1rem; color: #334155; }
    .highlight { color: #ef4444; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ============================================
# CONSTANTES E MODELOS FÍSICOS (funções puras, fáceis de testar)
# ============================================
MU0 = 4e-7 * np.pi          # permeabilidade do vácuo (T·m/A)
B_TERRA = 50e-6             # campo magnético terrestre típico (T)
C_LUZ = 299_792_458.0       # velocidade da luz (m/s)
COR_N, COR_S = "#ef4444", "#3b82f6"   # convenção: polo Norte vermelho, Sul azul

R_BOBINA_CM = 2.5           # raio da bobina da Aba 2 (cm)
N_FRAMES = 60


def campo_fio(corrente_A, r_m):
    """Campo de um fio longo: B = μ0·I / (2π·r)."""
    return MU0 * corrente_A / (2 * np.pi * r_m)


def fluxo_dipolo(d, R, m):
    """Fluxo de um dipolo magnético (momento m) sobre o eixo, a distância d, por uma espira de raio R.
    Φ = μ0·m·R² / (2·(R²+d²)^(3/2))  — resultado exato para dipolo pontual."""
    return MU0 * m * R**2 / (2 * (R**2 + d**2) ** 1.5)


def fem_dipolo(d, vx, R, m, N):
    """FEM de Faraday: ε = -N·dΦ/dt = -N·(dΦ/dd)·vx, com dΦ/dd derivado analiticamente."""
    return 3 * MU0 * m * N * R**2 * d * vx / (2 * (R**2 + d**2) ** 2.5)


def trajetoria(modo, v_ms, n):
    """Posição do ímã (cm) e velocidade (m/s) ao longo do tempo (s). Bobina fica em x = 0."""
    v_cm = 100.0 * v_ms
    if modo == "atravessa":
        T = 24.0 / v_cm
        t = np.linspace(0, T, n)
        return t, -12.0 + v_cm * t, np.full(n, v_ms)
    if modo == "oscila":
        A, xc = 5.5, -7.5
        w = v_cm / A                         # velocidade máxima = A·ω
        t = np.linspace(0, 4 * np.pi / w, n)  # dois vai-e-vens
        return t, xc - A * np.cos(w * t), v_ms * np.sin(w * t)
    t = np.linspace(0, 24.0 / v_cm, n)        # "parado"
    return t, np.full(n, -3.0), np.zeros(n)


def calcular_faraday(modo, N, v_ms, m, n):
    t, x_cm, vx = trajetoria(modo, v_ms, n)
    d = x_cm / 100.0
    R = R_BOBINA_CM / 100.0
    return t * 1e3, x_cm, vx, fluxo_dipolo(d, R, m) * 1e6, fem_dipolo(d, vx, R, m, N) * 1e3  # ms, cm, m/s, µWb, mV


ESCALAS_MV = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000]


def escolher_escala(pico):
    """Como num multímetro: escolhe o menor fundo de escala que comporta a medida."""
    return next((e for e in ESCALAS_MV if pico <= e), ESCALAS_MV[-1])


def bussola(corrente, r_cm):
    """Oersted: agulha sob campo da Terra (→ x) + campo do fio (→ ±y). Retorna B_fio (µT) e desvio (graus)."""
    b_fio = float(campo_fio(corrente, r_cm / 100.0))
    return abs(b_fio) * 1e6, math.degrees(math.atan2(b_fio, B_TERRA))


R_NFC, V_MIN_NFC = 2.5, 1.5     # raio da antena (cm) e tensão mínima p/ ligar o chip (V) — valores ilustrativos


def tensao_nfc(d_cm, n_espiras):
    v0 = 1.2 * n_espiras        # tensão de contato (ilustrativa), proporcional a N (Faraday)
    return v0 * (R_NFC**2 / (R_NFC**2 + d_cm**2)) ** 1.5   # mesma queda do acoplamento de dipolo


def alcance_nfc(n_espiras):
    v0 = 1.2 * n_espiras
    return 0.0 if v0 <= V_MIN_NFC else R_NFC * math.sqrt((v0 / V_MIN_NFC) ** (2 / 3) - 1)


SERVICOS = {
    "📻 Rádio AM (1 MHz)": 1e6,
    "📻 Rádio FM (100 MHz)": 100e6,
    "💳 NFC / cartão de aproximação (13,56 MHz)": 13.56e6,
    "📶 Wi-Fi 2,4 GHz": 2.4e9,
    "🍲 Forno de micro-ondas (2,45 GHz)": 2.45e9,
    "📱 5G (3,5 GHz)": 3.5e9,
}


def fmt_comprimento(m):
    return f"{m:,.0f} m".replace(",", ".") if m >= 10 else (f"{m:.2f} m" if m >= 1 else f"{m * 100:.1f} cm")


# ============================================
# COMPONENTES DE INTERFACE REUTILIZÁVEIS
# ============================================
def mostrar(fig):
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def botoes_animacao(duracao_ms, rotulo):
    return [{
        "type": "buttons", "showactive": False, "direction": "left",
        "x": 0.0, "y": 1.13, "xanchor": "left", "yanchor": "bottom",
        "buttons": [
            {"label": rotulo, "method": "animate",
             "args": [None, {"frame": {"duration": duracao_ms, "redraw": True}, "fromcurrent": True,
                             "transition": {"duration": 0}, "mode": "immediate"}]},
            {"label": "❚❚ Pausar", "method": "animate",
             "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate",
                               "transition": {"duration": 0}}]},
        ],
    }]


def pergunta(chave, enunciado, opcoes, correta, explicacao):
    """Mini-quiz 'Teste sua intuição': o aluno pode tentar de novo até acertar."""
    with st.expander("🎯 Teste sua intuição"):
        resp = st.radio(enunciado, opcoes, index=None, key=f"q_{chave}")
        if resp is not None:
            if opcoes.index(resp) == correta:
                st.success(f"✅ Isso mesmo! {explicacao}")
            else:
                st.warning("🤔 Ainda não. Volte à simulação, mexa nos controles, observe de novo e tente outra vez.")


# ============================================
# FIGURA 1 — OERSTED: CORRENTE GERA CAMPO (com bússola)
# ============================================
def gerar_grafico_oersted(corrente, r_cm):
    fig = go.Figure()
    sinal = 1 if corrente > 0 else -1
    cor = COR_N if corrente > 0 else COR_S

    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-5, 5], mode="lines",
                               line=dict(color="#94a3b8", width=14), hoverinfo="skip"))

    if corrente != 0:
        fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-4, 4], mode="lines",
                                   line=dict(color=cor, width=6), hoverinfo="skip"))
        fig.add_trace(go.Cone(x=[0], y=[0], z=[4.6 * sinal], u=[0], v=[0], w=[sinal], anchor="tip",
                              sizemode="absolute", sizeref=1.0, showscale=False,
                              colorscale=[[0, cor], [1, cor]], hoverinfo="skip"))

        # Anéis de campo: círculos + setas tangentes. Tamanho da seta ∝ I/r (as internas são mais fortes!)
        th = np.linspace(0, 2 * np.pi, 80)
        rx, ry, rz = [], [], []
        cx, cy, cz, cu, cv = [], [], [], [], []
        for z in (-2.5, 0.0, 2.5):
            for r in (1.5, 2.5, 4.0):
                rx += list(r * np.cos(th)) + [None]
                ry += list(r * np.sin(th)) + [None]
                rz += [z] * len(th) + [None]
                mag = 4.0 * abs(corrente) / (10 * r)
                for a in np.linspace(0, 2 * np.pi, 4, endpoint=False):
                    x, y = r * math.cos(a), r * math.sin(a)
                    cx.append(x); cy.append(y); cz.append(z)
                    cu.append(-sinal * math.sin(a) * mag)      # regra da mão direita: tangente ao círculo
                    cv.append(sinal * math.cos(a) * mag)
        fig.add_trace(go.Scatter3d(x=rx, y=ry, z=rz, mode="lines", hoverinfo="skip",
                                   line=dict(color="rgba(100,116,139,0.35)", width=2)))
        fig.add_trace(go.Cone(x=cx, y=cy, z=cz, u=cu, v=cv, w=[0] * len(cx), sizemode="absolute", sizeref=0.7,
                              colorscale="Reds" if corrente > 0 else "Blues", showscale=False, hoverinfo="skip"))

    # Bússola em (r, 0, 0): campo da Terra aponta para +x; o campo do fio é perpendicular (±y)
    _, ang = bussola(corrente, r_cm)
    dx, dy = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    fig.add_trace(go.Scatter3d(x=[r_cm, r_cm + 0.8 * dx], y=[0, 0.8 * dy], z=[0, 0], mode="lines",
                               line=dict(color=COR_N, width=12), hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(x=[r_cm, r_cm - 0.8 * dx], y=[0, -0.8 * dy], z=[0, 0], mode="lines",
                               line=dict(color="#e2e8f0", width=12), hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(x=[r_cm], y=[0], z=[0.9], mode="text", text=["🧭 bússola"],
                               textfont=dict(size=13, color="#0f172a"), hoverinfo="skip"))

    fig.update_layout(
        height=420, margin=dict(l=0, r=0, t=0, b=0), showlegend=False, paper_bgcolor="white",
        scene=dict(xaxis=dict(range=[-5.5, 5.5], visible=False), yaxis=dict(range=[-5.5, 5.5], visible=False),
                   zaxis=dict(range=[-5.5, 5.5], visible=False), camera=dict(eye=dict(x=1.2, y=1.2, z=0.9))),
    )
    return fig


# ============================================
# FIGURA 2 — FARADAY-LENZ (ímã × bobina, voltímetro, Φ(t), ε(t))
# ============================================
def gerar_animacao_faraday(modo, N, v_ms, m_mag, sinal, dur_ms):
    m = sinal * m_mag
    R_CM = R_BOBINA_CM

    # Curvas suaves (alta resolução) e quadros da animação
    t_c, _, _, phi_c, fem_c = calcular_faraday(modo, N, v_ms, m, 300)
    t_f, x_f, vx_f, phi_f, fem_f = calcular_faraday(modo, N, v_ms, m, N_FRAMES)
    pico = float(np.max(np.abs(fem_c)))
    FS = escolher_escala(pico)
    limiar = 0.04 * pico if pico > 0 else np.inf

    def faixa(y, pad=0.15):
        lo, hi = min(float(y.min()), 0.0), max(float(y.max()), 0.0)
        if hi - lo < 1e-12:
            lo, hi = -1.0, 1.0
        return lo - pad * (hi - lo), hi + pad * (hi - lo)

    phi_lo, phi_hi = faixa(phi_c)
    fem_lo, fem_hi = faixa(fem_c)

    # Linhas de campo de um dipolo (r = L·sen²θ), deslocadas junto com o ímã
    campo_x, campo_y = [], []
    for L in (1.8, 3.2, 4.8):
        th = np.linspace(0.04, np.pi - 0.04, 60)
        for lado in (1, -1):
            campo_x += list(L * np.sin(th) ** 2 * np.cos(th)) + [np.nan]
            campo_y += list(lado * L * np.sin(th) ** 3) + [np.nan]
    campo_x = np.array(campo_x)

    esq_cor, dir_cor = (COR_S, COR_N) if sinal > 0 else (COR_N, COR_S)
    lab_esq, lab_dir = ("S", "N") if sinal > 0 else ("N", "S")
    yi = [-0.8, -0.8, 0.8, 0.8, -0.8]
    cor_polo = {"N": COR_N, "S": COR_S}

    fig = make_subplots(
        rows=3, cols=2, column_widths=[0.68, 0.32], row_heights=[0.44, 0.28, 0.28],
        specs=[[{}, {}], [{"colspan": 2}, None], [{"colspan": 2}, None]],
        horizontal_spacing=0.06, vertical_spacing=0.09,
        subplot_titles=("Ímã e bobina (vista lateral)", f"Voltímetro (fundo de escala ±{FS:g} mV)",
                        "Fluxo magnético por espira  Φ(t)", "FEM induzida  ε(t)"),
    )
    idx = {}

    def add(nome, trace, row, col):
        fig.add_trace(trace, row=row, col=col)
        idx[nome] = len(fig.data) - 1

    # ---- Elementos estáticos ----
    add("bobina", go.Scatter(x=[-0.6, 0.6, 0.6, -0.6, -0.6], y=[-R_CM, -R_CM, R_CM, R_CM, -R_CM], mode="lines",
                             fill="toself", fillcolor="rgba(148,163,184,0.35)", line=dict(color="#64748b", width=3),
                             hoverinfo="skip"), 1, 1)
    add("rot_bobina", go.Scatter(x=[0], y=[-3.5], mode="text", text=[f"Bobina ({N} espiras)"],
                                 textfont=dict(size=12, color="#475569"), hoverinfo="skip"), 1, 1)
    add("chave", go.Scatter(x=[8.5], y=[-4.6], mode="text", text=["•  corrente saindo da tela<br>×  corrente entrando"],
                            textfont=dict(size=11, color="#64748b"), hoverinfo="skip"), 1, 1)

    arco = np.deg2rad(np.linspace(10, 170, 60))
    add("arco", go.Scatter(x=np.cos(arco), y=np.sin(arco), mode="lines", line=dict(color="#cbd5e1", width=4),
                           hoverinfo="skip"), 1, 2)
    tx, ty, lx, ly, lt = [], [], [], [], []
    for f in (-1, -0.5, 0, 0.5, 1):
        a = math.radians(90 - 80 * f)
        tx += [0.88 * math.cos(a), math.cos(a), None]
        ty += [0.88 * math.sin(a), math.sin(a), None]
        lx.append(1.17 * math.cos(a)); ly.append(1.17 * math.sin(a)); lt.append(f"{f * FS:g}")
    add("ticks", go.Scatter(x=tx, y=ty, mode="lines", line=dict(color="#94a3b8", width=2), hoverinfo="skip"), 1, 2)
    add("ticks_txt", go.Scatter(x=lx, y=ly, mode="text", text=lt, textfont=dict(size=11, color="#64748b"),
                                hoverinfo="skip"), 1, 2)
    add("pivo", go.Scatter(x=[0], y=[0], mode="markers", marker=dict(size=12, color="#0f172a"), hoverinfo="skip"), 1, 2)

    add("curva_phi", go.Scatter(x=t_c, y=phi_c, mode="lines", line=dict(color="#2563eb", width=3),
                                hovertemplate="t=%{x:.1f} ms<br>Φ=%{y:.2f} µWb<extra></extra>"), 2, 1)
    add("curva_fem", go.Scatter(x=t_c, y=fem_c, mode="lines", fill="tozeroy", fillcolor="rgba(239,68,68,0.15)",
                                line=dict(color="#dc2626", width=3),
                                hovertemplate="t=%{x:.1f} ms<br>ε=%{y:.1f} mV<extra></extra>"), 3, 1)

    # ---- Elementos dinâmicos: (classe, linha, coluna, propriedades fixas) ----
    FIXOS = {
        "campo": (go.Scatter, 1, 1, dict(mode="lines", y=campo_y, line=dict(color="#93c5fd", width=1.5), hoverinfo="skip")),
        "ima_esq": (go.Scatter, 1, 1, dict(mode="lines", y=yi, fill="toself", fillcolor=esq_cor,
                                           line=dict(color="#0f172a", width=2), hoverinfo="skip")),
        "ima_dir": (go.Scatter, 1, 1, dict(mode="lines", y=yi, fill="toself", fillcolor=dir_cor,
                                           line=dict(color="#0f172a", width=2), hoverinfo="skip")),
        "rot_polos": (go.Scatter, 1, 1, dict(mode="text", y=[0, 0], text=[lab_esq, lab_dir],
                                             textfont=dict(size=20, color="white"), hoverinfo="skip")),
        "fio_sup": (go.Scatter, 1, 1, dict(mode="markers+text", x=[0], y=[R_CM], textposition="middle center",
                                           hoverinfo="skip")),
        "fio_inf": (go.Scatter, 1, 1, dict(mode="markers+text", x=[0], y=[-R_CM], textposition="middle center",
                                           hoverinfo="skip")),
        "polos_ind": (go.Scatter, 1, 1, dict(mode="text", y=[1.9, 1.9], x=[-1.3, 1.3], hoverinfo="skip")),
        "legenda": (go.Scatter, 1, 1, dict(mode="text", x=[0], y=[6.9], textfont=dict(size=13, color="#0f172a"),
                                           hoverinfo="skip")),
        "agulha": (go.Scatter, 1, 2, dict(mode="lines", hoverinfo="skip")),
        "leitura": (go.Scatter, 1, 2, dict(mode="text", x=[0], y=[-0.22], textfont=dict(size=18, color="#0f172a"),
                                           hoverinfo="skip")),
        "cur_phi": (go.Scatter, 2, 1, dict(mode="lines", y=[phi_lo, phi_hi], hoverinfo="skip",
                                           line=dict(color="#64748b", width=2, dash="dot"))),
        "dot_phi": (go.Scatter, 2, 1, dict(mode="markers", hoverinfo="skip",
                                           marker=dict(size=13, color="#1d4ed8", line=dict(color="white", width=2)))),
        "cur_fem": (go.Scatter, 3, 1, dict(mode="lines", y=[fem_lo, fem_hi], hoverinfo="skip",
                                           line=dict(color="#64748b", width=2, dash="dot"))),
        "dot_fem": (go.Scatter, 3, 1, dict(mode="markers", hoverinfo="skip",
                                           marker=dict(size=13, color="#b91c1c", line=dict(color="white", width=2)))),
    }

    def dinamicos(i):
        xm, eps, ph, tms = float(x_f[i]), float(fem_f[i]), float(phi_f[i]), float(t_f[i])
        ativo = abs(eps) > limiar
        cor_e = "#dc2626" if eps > 0 else "#2563eb"
        # ε > 0  ⇒ corrente "horária" vista da esquerda; topo da espira sai da tela (•), base entra (×)
        sup, inf = ("•", "×") if eps > 0 else ("×", "•")
        marc = dict(symbol="circle-open", size=34, color=cor_e, opacity=1.0 if ativo else 0.0,
                    line=dict(color=cor_e, width=3))
        face_esq = "S" if eps > 0 else "N"          # a bobina vira um ímã: onde fica o N induzido?
        face_dir = "N" if eps > 0 else "S"

        if not ativo:
            l1, l2, l3 = "Fluxo praticamente constante", "→ nenhuma FEM, nenhuma corrente induzida", " "
        else:
            aumentando = np.sign(ph) * (-eps / N) > 0            # d|Φ|/dt > 0 ?
            l1 = "Fluxo AUMENTANDO ↑ — a bobina reage contra" if aumentando else "Fluxo DIMINUINDO ↓ — a bobina tenta manter"
            l2 = f"Corrente induzida: {'horária' if eps > 0 else 'anti-horária'} (vista da esquerda)"
            if abs(xm) < 2.2:
                l3 = "Ímã na bobina: o campo induzido segue freando o ímã"
            else:
                p_ima = ("N" if sinal > 0 else "S") if xm < 0 else ("S" if sinal > 0 else "N")
                p_bob = face_esq if xm < 0 else face_dir
                rel = "REPULSÃO" if p_ima == p_bob else "ATRAÇÃO"
                l3 = f"Polos frente a frente: {p_ima}–{p_bob} ⇒ {rel} (opõe-se ao movimento)"
        ang = math.radians(90 - 80 * float(np.clip(eps / FS, -1, 1)))

        return {
            "campo": dict(x=campo_x + xm),
            "ima_esq": dict(x=[xm - 1.5, xm, xm, xm - 1.5, xm - 1.5]),
            "ima_dir": dict(x=[xm, xm + 1.5, xm + 1.5, xm, xm]),
            "rot_polos": dict(x=[xm - 0.75, xm + 0.75]),
            "fio_sup": dict(text=[sup if ativo else ""], marker=marc, textfont=dict(size=30, color=cor_e)),
            "fio_inf": dict(text=[inf if ativo else ""], marker=marc, textfont=dict(size=30, color=cor_e)),
            "polos_ind": dict(text=[face_esq, face_dir] if ativo else ["", ""],
                              textfont=dict(size=20, color=[cor_polo[face_esq], cor_polo[face_dir]])),
            "legenda": dict(text=[f"<b>{l1}</b><br>{l2}<br>{l3}"]),
            "agulha": dict(x=[0, 0.92 * math.cos(ang)], y=[0, 0.92 * math.sin(ang)],
                           line=dict(color=cor_e if ativo else "#0f172a", width=5)),
            "leitura": dict(text=[f"{eps:+.1f} mV"]),
            "cur_phi": dict(x=[tms, tms]), "dot_phi": dict(x=[tms], y=[ph]),
            "cur_fem": dict(x=[tms, tms]), "dot_fem": dict(x=[tms], y=[eps]),
        }

    d0 = dinamicos(0)
    for nome, (cls, r, c, fixo) in FIXOS.items():
        add(nome, cls(**fixo, **d0[nome]), r, c)

    frames = []
    for i in range(N_FRAMES):
        d = dinamicos(i)
        frames.append(go.Frame(name=str(i), traces=[idx[n] for n in FIXOS],
                               data=[FIXOS[n][0](**d[n]) for n in FIXOS]))
    fig.frames = frames

    # ---- Eixos e layout ----
    fig.update_xaxes(range=[-14, 14], visible=False, constrain="domain", row=1, col=1)
    fig.update_yaxes(range=[-5.5, 8.4], visible=False, scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_xaxes(range=[-1.3, 1.3], visible=False, constrain="domain", row=1, col=2)
    fig.update_yaxes(range=[-0.4, 1.35], visible=False, scaleanchor="x2", scaleratio=1, row=1, col=2)
    for r_, ttl, rng in ((2, "Φ (µWb)", (phi_lo, phi_hi)), (3, "ε (mV)", (fem_lo, fem_hi))):
        fig.update_xaxes(range=[float(t_c[0]), float(t_c[-1])], gridcolor="#e2e8f0", zeroline=False,
                         title_text="tempo (ms)" if r_ == 3 else None, row=r_, col=1)
        fig.update_yaxes(range=list(rng), gridcolor="#e2e8f0", zerolinecolor="#94a3b8", title_text=ttl, row=r_, col=1)

    passos = [dict(method="animate", label=f"{t:.1f}",
                   args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))])
              for i, t in enumerate(t_f)]
    fig.update_layout(
        height=880, showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=10, r=10, t=90, b=10),
        updatemenus=botoes_animacao(dur_ms, "▶ Iniciar experimento"),
        sliders=[dict(active=0, x=0.0, len=1.0, y=-0.01, pad=dict(t=30, b=0), steps=passos,
                      currentvalue=dict(prefix="⏱ t = ", suffix=" ms  (arraste para ver quadro a quadro)",
                                        font=dict(size=13, color="#334155")),
                      font=dict(color="rgba(0,0,0,0)"), tickcolor="rgba(0,0,0,0)", ticklen=0, minorticklen=0)],
    )
    return fig


# ============================================
# FIGURA 3 — NFC: distância × tensão induzida × chip
# ============================================
def gerar_diagrama_nfc(d_cm, n_espiras):
    v = tensao_nfc(d_cm, n_espiras)
    ligado = v >= V_MIN_NFC
    cor_estado = "#10b981" if ligado else "#ef4444"

    fig = make_subplots(rows=1, cols=2, column_widths=[0.58, 0.42], horizontal_spacing=0.08,
                        subplot_titles=("Celular (ativo)  →  Cartão (passivo)", "Tensão induzida no cartão × distância"))

    # Celular: retângulo + bobina; linhas de campo de dipolo saindo da bobina
    fig.add_shape(type="rect", x0=-1.0, y0=-6, x1=-0.2, y1=6, line=dict(color="#1e293b", width=3),
                  fillcolor="#f1f5f9", row=1, col=1)
    fig.add_trace(go.Scatter(x=[-0.6, -0.6], y=[-R_NFC, R_NFC], mode="lines", line=dict(color=COR_N, width=8),
                             hoverinfo="skip"), row=1, col=1)
    th = np.linspace(0.04, np.pi - 0.04, 60)
    lx, ly = [], []
    for L in (3, 5, 7.5):
        for lado in (1, -1):
            lx += list(-0.6 + L * np.sin(th) ** 2 * np.cos(th)) + [None]
            ly += list(lado * L * np.sin(th) ** 3) + [None]
    fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines", line=dict(color="#10b981", width=1.5, dash="dash"),
                             hoverinfo="skip"), row=1, col=1)

    # Cartão na distância escolhida
    fig.add_shape(type="rect", x0=d_cm, y0=-2.7, x1=d_cm + 0.25, y1=2.7, line=dict(color="#1e293b", width=3),
                  fillcolor="#e2e8f0", row=1, col=1)
    fig.add_trace(go.Scatter(x=[d_cm + 0.12] * 2, y=[-R_NFC, R_NFC], mode="lines",
                             line=dict(color=COR_S, width=5, dash="dot"), hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[d_cm + 0.12], y=[1.6], mode="markers", marker=dict(symbol="square", size=14, color="#334155"),
                             hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[-0.6, d_cm + 0.12, d_cm + 0.12], y=[-6.2, -3.5, 3.6], mode="text",
                             text=["Celular", "Cartão", ("Chip LIGADO ✓" if ligado else "Chip desligado ✗")],
                             textfont=dict(size=13, color=["#0f172a", "#0f172a", cor_estado]),
                             hoverinfo="skip"), row=1, col=1)

    # Curva V(d), zona de operação e ponto atual
    dd = np.linspace(0, 8, 200)
    fig.add_trace(go.Scatter(x=dd, y=tensao_nfc(dd, n_espiras), mode="lines", line=dict(color="#3b82f6", width=3),
                             hovertemplate="d=%{x:.1f} cm<br>V=%{y:.2f} V<extra></extra>"), row=1, col=2)
    fig.add_hline(y=V_MIN_NFC, line=dict(color="#f59e0b", dash="dash", width=2),
                  annotation_text="tensão mínima do chip", annotation_position="top right", row=1, col=2)
    alc = alcance_nfc(n_espiras)
    if alc > 0:
        fig.add_vrect(x0=0, x1=alc, fillcolor="rgba(16,185,129,0.12)", line_width=0, row=1, col=2)
    fig.add_trace(go.Scatter(x=[d_cm], y=[v], mode="markers", marker=dict(size=14, color=cor_estado,
                             line=dict(color="white", width=2)), hoverinfo="skip"), row=1, col=2)

    fig.update_xaxes(range=[-3, 12], visible=False, constrain="domain", row=1, col=1)
    fig.update_yaxes(range=[-6.6, 6.6], visible=False, scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_xaxes(title_text="distância (cm)", range=[0, 8], gridcolor="#e2e8f0", row=1, col=2)
    fig.update_yaxes(title_text="tensão induzida (V, ilustrativo)", rangemode="tozero", gridcolor="#e2e8f0", row=1, col=2)
    fig.update_layout(height=430, showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


# ============================================
# FIGURA 4 — ANTENA E ONDA EM 3D (E × B aponta na propagação)
# ============================================
def gerar_animacao_antena(duracao_ms):
    """Onda para +x com E ao longo de y (paralelo à antena) e B ao longo de z:  E × B = +x."""
    fig = make_subplots(rows=1, cols=2, column_widths=[0.25, 0.75], horizontal_spacing=0.05,
                        specs=[[{"type": "xy"}, {"type": "scene"}]],
                        subplot_titles=("Micro: elétrons na antena", "Macro: onda eletromagnética (3D)"))
    k, n_q = 2 * np.pi / 4.0, 96
    t_vals = np.linspace(0, 4 * np.pi, n_q, endpoint=False)        # 2 ciclos completos
    x_onda = np.linspace(0, 10, 120)
    x_st = np.arange(0.5, 10.01, 0.5)                                 # posições das "hastes" de campo
    x_h = [v for xs in x_st for v in (xs, xs, None)]
    z_h = [v for _ in x_st for v in (0.0, 0.0, None)]

    def hastes(val):
        return [w for a in val for w in (0.0, float(a), None)]

    idx = {}

    def add(nome, trace, row, col):
        fig.add_trace(trace, row=row, col=col)
        idx[nome] = len(fig.data) - 1

    add("antena2d", go.Scatter(x=[0, 0], y=[-2, 2], mode="lines", line=dict(color="#94a3b8", width=8), hoverinfo="skip"), 1, 1)
    add("antena3d", go.Scatter3d(x=[0, 0], y=[-1.3, 1.3], z=[0, 0], mode="lines", line=dict(color="#64748b", width=10),
                                 hoverinfo="skip"), 1, 2)
    add("eixo", go.Scatter3d(x=[0, 10], y=[0, 0], z=[0, 0], mode="lines", line=dict(color="black", width=2),
                             hoverinfo="skip"), 1, 2)
    add("poynting", go.Cone(x=[10.2], y=[0], z=[0], u=[1], v=[0], w=[0], anchor="tail", sizemode="absolute", sizeref=1.0,
                            showscale=False, colorscale=[[0, "#0f172a"], [1, "#0f172a"]], hoverinfo="skip"), 1, 2)

    def dinamicos(t):
        s = np.sin(k * x_onda - t)                    # E_y = sen(kx − ωt);  B_z = mesma fase
        s_st = np.sin(k * x_st - t)
        return {
            "eletron": dict(y=[1.5 * math.sin(t)]),
            "E": dict(y=s), "B": dict(z=s),
            "hE": dict(y=hastes(s_st)), "hB": dict(z=hastes(s_st)),
        }

    FIXOS = {
        "eletron": (go.Scatter, 1, 1, dict(mode="markers", x=[0], marker=dict(color="red", size=16), hoverinfo="skip")),
        "E": (go.Scatter3d, 1, 2, dict(mode="lines", x=x_onda, z=np.zeros_like(x_onda),
                                       line=dict(color="#3b82f6", width=5), hoverinfo="skip")),
        "B": (go.Scatter3d, 1, 2, dict(mode="lines", x=x_onda, y=np.zeros_like(x_onda),
                                       line=dict(color="#ef4444", width=5), hoverinfo="skip")),
        "hE": (go.Scatter3d, 1, 2, dict(mode="lines", x=x_h, z=z_h, line=dict(color="rgba(59,130,246,0.55)", width=3),
                                        hoverinfo="skip")),
        "hB": (go.Scatter3d, 1, 2, dict(mode="lines", x=x_h, y=z_h, line=dict(color="rgba(239,68,68,0.55)", width=3),
                                        hoverinfo="skip")),
    }
    d0 = dinamicos(t_vals[0])
    for nome, (cls, r, c, fixo) in FIXOS.items():
        add(nome, cls(**fixo, **d0[nome]), r, c)

    fig.frames = [go.Frame(name=str(i), traces=[idx[n] for n in FIXOS],
                           data=[FIXOS[n][0](**dinamicos(t)[n]) for n in FIXOS]) for i, t in enumerate(t_vals)]

    fig.update_layout(
        height=500, showlegend=False, paper_bgcolor="white", plot_bgcolor="white", margin=dict(l=10, r=10, t=90, b=10),
        scene=dict(xaxis_title="Propagação (x)", yaxis_title="Campo E (y)", zaxis_title="Campo B (z)",
                   xaxis=dict(range=[0, 10.8], showgrid=False, zeroline=False, showticklabels=False),
                   yaxis=dict(range=[-1.6, 1.6], showgrid=False, zeroline=False, showticklabels=False),
                   zaxis=dict(range=[-1.6, 1.6], showgrid=False, zeroline=False, showticklabels=False),
                   aspectmode="manual", aspectratio=dict(x=3, y=1, z=1), camera=dict(eye=dict(x=1.5, y=-1.7, z=0.8))),
        updatemenus=botoes_animacao(duracao_ms, "▶ Transmitir sinal"),
    )
    fig.update_xaxes(range=[-1, 1], visible=False, row=1, col=1)
    fig.update_yaxes(range=[-2.5, 2.5], visible=False, row=1, col=1)
    return fig


# ============================================
# BARRA LATERAL (guia do professor)
# ============================================
with st.sidebar:
    st.header("⚙️ Configurações")
    avancado = st.checkbox("Mostrar equações e valores numéricos", value=False,
                           help="Desligado: foco na intuição (ensino médio). Ligado: mostra as fórmulas usadas na simulação.")
    st.markdown("---")
    st.subheader("🧑‍🏫 Roteiro sugerido")
    st.markdown(
        "**1. Prever** – peça que os alunos apostem o que vai acontecer (use as caixas *Teste sua intuição*).\n\n"
        "**2. Observar** – rodem a animação e arrastem a barra de tempo para pausar em qualquer instante.\n\n"
        "**3. Explicar** – relacionem com Faraday e Lenz: *o que variou? qual foi a reação da bobina?*"
    )

# ============================================
# TÍTULO E ABAS
# ============================================
st.markdown('<div class="main-title">🧲 Eletromagnetismo Visual</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">A simetria da natureza: eletricidade gerando magnetismo e vice-versa</div>',
            unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["1. Fundamentos (Oersted)", "2. Indução e Lei de Lenz",
                                  "3. Tecnologia NFC", "4. Antenas e Ondas"])

# ---------------- ABA 1 ----------------
with tab1:
    st.markdown("""
    <div class="concept-card">
        <b>O início de tudo:</b> até 1820, eletricidade e magnetismo eram ciências separadas. Oersted percebeu que a
        agulha de uma bússola se movia quando havia corrente num fio por perto. Anos depois, Faraday perguntou:
        <i>"Se a eletricidade gera magnetismo, será que o magnetismo pode gerar eletricidade?"</i>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 2.2])
    with col_a:
        with st.container(border=True):
            st.markdown("### 1. Corrente gera magnetismo")
            st.markdown("Mude o **sentido e a intensidade** da corrente e observe a **regra da mão direita**: "
                        "polegar no sentido da corrente, dedos curvados no sentido do campo.")
            corrente = st.slider("Corrente elétrica I (A)", -10.0, 10.0, 5.0, 1.0)
            r_cm = st.slider("Distância da bússola ao fio (cm)", 1.0, 4.0, 2.0, 0.5)
            b_uT, desvio = bussola(corrente, r_cm)
            m1, m2 = st.columns(2)
            m1.metric("Campo do fio", f"{b_uT:.0f} µT")
            m2.metric("Desvio da agulha", f"{abs(desvio):.0f}°")
            st.caption("Referência: o campo da Terra vale ≈ 50 µT. É por isso que a bússola de Oersted se moveu — "
                       "o fio ganhou da Terra!")
            if avancado:
                st.latex(r"B = \frac{\mu_0 \, I}{2\pi \, r}")
                st.caption("Repare: as setas internas (r pequeno) são maiores — o campo cai com 1/r.")
    with col_b:
        mostrar(gerar_grafico_oersted(corrente, r_cm))

    pergunta("oersted", "Se você inverter o sentido da corrente, o que acontece com a agulha da bússola?",
             ["Nada, ela continua igual", "Ela desvia para o lado oposto", "Ela gira sem parar"], 1,
             "O campo magnético inverte junto com a corrente, então o desvio também inverte.")

    st.markdown("---")
    st.markdown("""
    <div class="alert-card">
        <h3>2. O campo magnético gera corrente? (a busca de Faraday)</h3>
        Sabendo disso, Faraday colocou um ímã forte e <b>parado</b> ao lado de um fio, esperando uma corrente.
        <b>Resultado: zero.</b> O segredo veio quase por acaso: o ímã não pode ficar parado!
        É preciso a <b>VARIAÇÃO</b> do campo magnético. Vá à <b>Aba 2</b> e comprove.
    </div>
    """, unsafe_allow_html=True)

# ---------------- ABA 2 ----------------
MODOS = {"🧲 Ímã atravessa a bobina": "atravessa", "↔️ Ímã aproxima e afasta": "oscila",
         "⏸️ Ímã parado (o 'fracasso' de Faraday)": "parado"}

with tab2:
    st.markdown("""
    <div class="concept-card">
        <b>Lei de Faraday:</b> a <i>variação</i> do fluxo magnético através de uma espira gera uma tensão (FEM) induzida.<br>
        <b>Lei de Lenz (o sinal de menos):</b> a corrente induzida surge num sentido tal que seu próprio campo
        <span class="highlight">se opõe</span> à variação que a causou. A natureza "resiste" à mudança.
    </div>
    """, unsafe_allow_html=True)
    st.latex(r"\mathcal{E} = -N\,\frac{\Delta \Phi_B}{\Delta t}")

    col1, col2 = st.columns([1, 2.6])
    with col1:
        with st.container(border=True):
            st.markdown("**Controles do experimento**")
            modo_lbl = st.radio("O que o ímã faz?", list(MODOS), key="modo_f")
            N_esp = st.slider("Espiras da bobina (N)", 1, 300, 100, key="N_f")
            v_ms = st.slider("Velocidade do ímã (m/s)", 0.2, 2.0, 1.0, 0.1, key="v_f")
            m_mag = st.slider("Força do ímã (A·m²)", 0.5, 2.0, 1.0, 0.1, key="m_f")
            inverter = st.checkbox("Inverter os polos do ímã", key="inv_f")
            dur = st.select_slider("Velocidade da animação", options=[120, 60, 30], value=60, key="dur_f",
                                   format_func=lambda x: {120: "Lenta", 60: "Normal", 30: "Rápida"}[x])
            st.caption("A animação roda em câmera lenta; os gráficos mostram o tempo real em milissegundos.")

        with st.expander("🔬 Desafios para explorar"):
            st.markdown(
                "1. **Dobre a velocidade.** O que acontece com a altura e a largura do pulso de ε?\n"
                "2. **Dobre N.** O pico de ε muda? (E o fluxo Φ por espira?)\n"
                "3. **Inverta os polos.** Qual gráfico muda de sinal, Φ, ε ou ambos?\n"
                "4. **Deixe o ímã parado.** Há fluxo? Há FEM? Por quê?\n"
                "5. **Ímã atravessando:** por que ε muda de sinal quando ele passa pelo centro?"
            )

    modo = MODOS[modo_lbl]
    sinal = -1 if inverter else 1
    _, _, _, phi_ref, fem_ref = calcular_faraday(modo, N_esp, v_ms, sinal * m_mag, 300)
    t_tot = calcular_faraday(modo, N_esp, v_ms, sinal * m_mag, 2)[0][-1]

    with col2:
        c1, c2, c3 = st.columns(3)
        c1.metric("Pico da FEM", f"{np.max(np.abs(fem_ref)):.1f} mV")
        c2.metric("Fluxo máximo (por espira)", f"{np.max(np.abs(phi_ref)):.1f} µWb")
        c3.metric("Duração real do experimento", f"{t_tot:.0f} ms")
        mostrar(gerar_animacao_faraday(modo, N_esp, v_ms, m_mag, sinal, dur))

    if avancado:
        st.markdown("**Modelo usado na simulação** — ímã como dipolo magnético m sobre o eixo da bobina (raio R), a distância d:")
        st.latex(r"\Phi(d)=\frac{\mu_0\,m\,R^2}{2\,(R^2+d^2)^{3/2}}\qquad "
                 r"\mathcal{E}=-N\frac{d\Phi}{dt}=\frac{3\,\mu_0\,m\,N\,R^2\,d\,v}{2\,(R^2+d^2)^{5/2}}")
        st.caption(f"R = {R_BOBINA_CM} cm. Note que ε ∝ N · v · m: mais espiras, mais velocidade ou ímã mais forte "
                   "= mais tensão. E ε = 0 quando v = 0 ou quando o ímã está exatamente no centro (dΦ/dd = 0).")

    pergunta("lenz", "Você segura o ímã parado dentro da bobina. Qual é a corrente induzida?",
             ["Máxima, pois o fluxo é grande", "Zero, pois o fluxo não está variando",
              "Pequena, mas constante"], 1,
             "Faraday depende da VARIAÇÃO do fluxo. Fluxo grande porém constante ⇒ ε = 0.")

# ---------------- ABA 3 ----------------
with tab3:
    st.markdown("""
    <div class="concept-card" style="border-left-color: #10b981;">
        <b>NFC (Near Field Communication):</b> tecnologia do pagamento por aproximação, crachás de acesso e bilhetes
        de transporte. Opera em <b>13,56 MHz</b> e funciona graças à indução eletromagnética de Faraday.
    </div>
    """, unsafe_allow_html=True)

    cc1, cc2 = st.columns([1, 2.4])
    with cc1:
        with st.container(border=True):
            st.markdown("**Aproxime o cartão do celular**")
            dist = st.slider("Distância celular–cartão (cm)", 0.0, 8.0, 5.0, 0.5, key="d_nfc")
            n_cartao = st.slider("Espiras na antena do cartão", 1, 10, 5, key="n_nfc")
            v_nfc = tensao_nfc(dist, n_cartao)
            st.metric("Tensão induzida", f"{v_nfc:.2f} V", help="Valores ilustrativos, mas com a física correta.")
            if v_nfc >= V_MIN_NFC:
                st.success("Chip LIGADO: energia suficiente!")
            else:
                st.error("Chip desligado: pouca energia induzida.")
            alc = alcance_nfc(n_cartao)
            st.caption(f"Alcance com {n_cartao} espira(s): {alc:.1f} cm." if alc > 0
                       else "Com tão poucas espiras o chip nunca liga.")
    with cc2:
        mostrar(gerar_diagrama_nfc(dist, n_cartao))

    with st.container(border=True):
        st.markdown("### Como acontece a mágica? (passo a passo)")
        st.markdown("""
**1. O campo ativo:** o celular (ou a maquininha) tem uma bobina alimentada por bateria que gera um campo magnético alternado, milhões de vezes por segundo.

**2. Indução no cartão:** o cartão **não tem bateria**. Ele possui uma antena de cobre enrolada. O campo variável do celular atravessa essa antena.

**3. Energia "do nada":** a variação do fluxo (Lei de Faraday) induz uma corrente que alimenta o microchip. *Afaste o cartão e veja a tensão despencar!*

**4. A resposta:** o chip altera a resistência da própria antena e isso perturba o campo do celular (*modulação de carga*). O celular "sente" essa perturbação e decodifica os dados.
        """)
    if avancado:
        st.latex(r"V_{ind} \;\propto\; N \cdot f \cdot \frac{R^3}{(R^2+d^2)^{3/2}}")
        st.caption("Mais espiras (N) ou mais frequência (f) ⇒ mais tensão; a distância d derruba o acoplamento rapidamente.")

    pergunta("nfc", "Por que o cartão NFC precisa ficar a poucos centímetros do celular?",
             ["Porque o fluxo magnético que atravessa a antena do cartão cai rapidamente com a distância",
              "Porque o cartão tem uma bateria fraca", "Porque a luz não passa por plástico"], 0,
             "Sem bateria, toda a energia vem do campo induzido, e ele enfraquece muito rápido com a distância.")

# ---------------- ABA 4 ----------------
with tab4:
    st.markdown("""
    <div class="alert-card">
        <b>O segredo das telecomunicações (Wi-Fi, 5G, rádio):</b> como a informação atravessa paredes de forma
        invisível? A resposta está nas cargas elétricas aceleradas.
    </div>
    """, unsafe_allow_html=True)

    ca1, ca2 = st.columns([1, 2.5])
    with ca1:
        with st.container(border=True):
            st.markdown("### A visão micro e macro")
            st.markdown(
                "• **Microscópico:** um circuito oscilador faz os elétrons subirem e descerem na antena. "
                "Cargas **aceleradas** irradiam.\n\n"
                "• **Macroscópico:** a perturbação se solta como **onda eletromagnética**, viajando à velocidade da luz "
                "(c ≈ 300 000 km/s).\n\n"
                "• **Regra da onda:** o campo elétrico **E** (azul) e o magnético **B** (vermelho) são perpendiculares "
                "entre si e à direção de propagação (E × B aponta para onde a onda vai)."
            )
            dur_ant = st.select_slider("Velocidade da animação", options=[120, 70, 40], value=70, key="dur_ant",
                                       format_func=lambda x: {120: "Lenta", 70: "Normal", 40: "Rápida"}[x])
    with ca2:
        mostrar(gerar_animacao_antena(dur_ant))

    st.markdown("#### 📡 Quão grande é uma onda de verdade?")
    st.caption("A animação está fora de escala. Escolha um serviço e veja o comprimento de onda real: λ = c / f.")
    serv = st.selectbox("Serviço", list(SERVICOS), index=2)
    f_hz = SERVICOS[serv]
    lam = C_LUZ / f_hz
    s1, s2, s3 = st.columns(3)
    s1.metric("Frequência", f"{f_hz / 1e6:,.2f} MHz".replace(",", "."))
    s2.metric("Comprimento de onda λ", fmt_comprimento(lam))
    s3.metric("Antena de meia-onda ≈ λ/2", fmt_comprimento(lam / 2))
    if "NFC" in serv:
        st.info("💡 Com λ ≈ 22 m e o cartão a poucos centímetros, o cartão está no **campo próximo**: ele não "
                "'recebe uma onda', ele participa de uma **indução entre bobinas** — exatamente a Aba 3.")
    else:
        st.info("💡 Antenas eficientes têm tamanho comparável ao comprimento de onda. Por isso a antena de um "
                "celular (Wi-Fi, 5G) cabe no bolso, mas a de uma rádio AM ocupa torres.")

    pergunta("ondas", "Numa onda eletromagnética, E e B são…",
             ["Paralelos entre si e à propagação", "Perpendiculares entre si e à direção de propagação",
              "Iguais em direção, mas com sentidos opostos"], 1,
             "Os campos são transversais e perpendiculares, e E × B aponta na direção em que a onda viaja.")

# Rodapé
st.markdown("---")
st.markdown('<div style="text-align:center;color:#94a3b8;font-size:0.85rem;padding:1rem;">'
            '🧲 <b>Física Visual: Eletromagnetismo</b> — construído para ensino interativo via simulação matemática.'
            '</div>', unsafe_allow_html=True)
