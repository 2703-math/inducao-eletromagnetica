import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

# ============================================================
# FÍSICA VISUAL — ELETROMAGNETISMO 2.0
# ============================================================
# Dependências:
#   pip install streamlit numpy plotly
#
# Executar:
#   streamlit run eletromagnetismo_visual_2.py
# ============================================================

st.set_page_config(
    page_title="Física Visual: Eletromagnetismo 2.0",
    page_icon="🧲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CONSTANTES
# ============================================================

MU0 = 4 * np.pi * 1e-7
C = 299_792_458.0


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.35rem;
        font-weight: 800;
        color: #0f172a;
        text-align: center;
        margin-bottom: 0.15rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 1.5rem;
    }

    .concept-card {
        background: #f8fafc;
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        border-left: 4px solid #3b82f6;
        margin-bottom: 1rem;
        color: #334155;
    }

    .success-card {
        background: #f0fdf4;
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        border-left: 4px solid #16a34a;
        margin-bottom: 1rem;
        color: #334155;
    }

    .alert-card {
        background: #fffbeb;
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        border-left: 4px solid #f59e0b;
        margin-bottom: 1rem;
        color: #334155;
    }

    .formula-card {
        background: #eff6ff;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #bfdbfe;
        margin: 0.7rem 0;
        color: #1e3a8a;
        text-align: center;
    }

    .param-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }

    .highlight {
        color: #ef4444;
        font-weight: 700;
    }

    .small-note {
        color: #64748b;
        font-size: 0.88rem;
    }

    .metric-box {
        text-align: center;
        background: #f8fafc;
        border-radius: 10px;
        padding: 0.8rem;
        border: 1px solid #e2e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def magnitude(v):
    return float(np.linalg.norm(v))


def normalize_vector(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros_like(v, dtype=float)
    return np.asarray(v, dtype=float) / n


def metric_row(items):
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        with col:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="small-note">{label}</div>
                    <strong style="font-size:1.25rem">{value}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# 1. CAMPO ELÉTRICO — VISUALIZAÇÃO 2D
# ============================================================

@st.cache_data
def gerar_campo_eletrico(q, distancia, grade):
    k = 8.9875517923e9
    lim = 4.0
    xs = np.linspace(-lim, lim, grade)
    ys = np.linspace(-lim, lim, grade)
    X, Y = np.meshgrid(xs, ys)

    dx = X
    dy = Y
    R2 = dx**2 + dy**2
    R = np.sqrt(R2)
    R_safe = np.maximum(R, 0.35)

    Ex = k * q * dx / (R_safe**3)
    Ey = k * q * dy / (R_safe**3)

    mask = R < 0.35
    Ex[mask] = np.nan
    Ey[mask] = np.nan

    # Normalização apenas para visualização das setas.
    E_mod = np.sqrt(np.nan_to_num(Ex)**2 + np.nan_to_num(Ey)**2)
    E_scale = np.percentile(E_mod[np.isfinite(E_mod)], 80)
    E_scale = max(E_scale, 1e-12)

    U = Ex / E_scale
    V = Ey / E_scale

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode="markers",
            marker=dict(
                size=22,
                color="red" if q > 0 else "blue",
                symbol="circle",
            ),
            name="Carga",
        )
    )

    step = 2
    fig.add_trace(
        go.Cone(
            x=X[::step, ::step].flatten(),
            y=Y[::step, ::step].flatten(),
            z=np.zeros_like(X[::step, ::step].flatten()),
            u=U[::step, ::step].flatten(),
            v=V[::step, ::step].flatten(),
            w=np.zeros_like(U[::step, ::step].flatten()),
            sizemode="absolute",
            sizeref=0.35,
            showscale=False,
            anchor="tail",
            name="Campo E",
        )
    )

    fig.update_layout(
        height=500,
        template="plotly_white",
        title="Linhas de campo elétrico — representação vetorial",
        margin=dict(l=10, r=10, t=45, b=10),
        xaxis=dict(title="x", range=[-lim, lim], scaleanchor="y"),
        yaxis=dict(title="y", range=[-lim, lim]),
    )

    return fig


# ============================================================
# 2. OERSTED — CORRENTE GERA CAMPO MAGNÉTICO
# ============================================================

@st.cache_data
def gerar_grafico_oersted(corrente, raio_max=5.0):
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.55, 0.45],
        subplot_titles=(
            "Campo magnético ao redor do fio",
            "B × distância",
        ),
    )

    # Vista superior do fio.
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode="markers",
            marker=dict(
                size=18,
                color="red" if corrente > 0 else "blue",
            ),
            name="Fio",
            hovertemplate="Condutor<extra></extra>",
        ),
        row=1,
        col=1,
    )

    if abs(corrente) > 0:
        sentido = 1 if corrente > 0 else -1

        for raio in [1.0, 2.0, 3.0, 4.0]:
            theta = np.linspace(0, 2 * np.pi, 120)

            x = raio * np.cos(theta)
            y = raio * np.sin(theta)

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(width=2),
                    name=f"r = {raio:.1f} m",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

            # Seta tangencial.
            theta_a = 0.55
            theta_b = theta_a + sentido * 0.28

            x0 = raio * np.cos(theta_a)
            y0 = raio * np.sin(theta_a)
            x1 = raio * np.cos(theta_b)
            y1 = raio * np.sin(theta_b)

            fig.add_annotation(
                x=x1,
                y=y1,
                ax=x0,
                ay=y0,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1.3,
                row=1,
                col=1,
            )

    r = np.linspace(0.25, raio_max, 300)
    B = MU0 * abs(corrente) / (2 * np.pi * r)

    fig.add_trace(
        go.Scatter(
            x=r,
            y=B,
            mode="lines",
            name="|B(r)|",
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(
        title_text="x (m)",
        range=[-5.2, 5.2],
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="y (m)",
        range=[-5.2, 5.2],
        scaleanchor="x",
        scaleratio=1,
        row=1,
        col=1,
    )

    fig.update_xaxes(
        title_text="Distância r (m)",
        row=1,
        col=2,
    )
    fig.update_yaxes(
        title_text="B (T)",
        row=1,
        col=2,
    )

    fig.update_layout(
        height=520,
        template="plotly_white",
        margin=dict(l=10, r=10, t=55, b=10),
        showlegend=False,
    )

    return fig


# ============================================================
# 3. FORÇA DE LORENTZ — RK4
# ============================================================

def derivadas_lorentz(state, q, m, E, B):
    v = state[3:6]
    F = q * (E + np.cross(v, B))
    a = F / m
    return np.concatenate((v, a))


@st.cache_data
def simular_lorentz(
    q,
    m,
    v0,
    E,
    B,
    t_max,
    dt,
):
    B_mod = np.linalg.norm(B)

    # Critério conservador de estabilidade.
    if B_mod > 0:
        omega = abs(q) * B_mod / m
        if omega * dt > 0.5:
            return None, (
                f"Passo de tempo insuficiente para estes parâmetros "
                f"(ω·dt = {omega * dt:.2f}). "
                f"Use um dt menor que {0.5 / omega:.5f} s."
            )

    passos = int(np.ceil(t_max / dt)) + 1

    if passos > 120_000:
        return None, (
            "A simulação teria mais de 120.000 passos. "
            "Aumente dt ou reduza o tempo total."
        )

    state = np.zeros((passos, 6), dtype=float)
    state[0, :3] = 0.0
    state[0, 3:] = v0

    for i in range(1, passos):
        y = state[i - 1]

        k1 = dt * derivadas_lorentz(y, q, m, E, B)
        k2 = dt * derivadas_lorentz(
            y + 0.5 * k1, q, m, E, B
        )
        k3 = dt * derivadas_lorentz(
            y + 0.5 * k2, q, m, E, B
        )
        k4 = dt * derivadas_lorentz(
            y + k3, q, m, E, B
        )

        state[i] = y + (
            k1 + 2 * k2 + 2 * k3 + k4
        ) / 6.0

        if not np.all(np.isfinite(state[i])):
            return None, (
                "A trajetória ficou numericamente instável. "
                "Reduza dt."
            )

        if np.linalg.norm(state[i, :3]) > 1e8:
            return None, (
                "A posição cresceu além do limite de segurança. "
                "Reduza dt ou os campos aplicados."
            )

    tempo = np.arange(passos) * dt
    return (tempo, state), None


# ============================================================
# 4. FARADAY / LENZ — MODELO QUANTITATIVO
# ============================================================

@st.cache_data
def dados_inducao(
    velocidade,
    intensidade,
    area,
    n_espiras,
):
    # Modelo didático:
    # B(x) = B0 exp[-(x/sigma)^2]
    # O fluxo varia com a posição do ímã.
    #
    # A posição é modelada como movimento uniforme.
    t = np.linspace(-2.0, 2.0, 400)

    x = velocidade * t
    sigma = 0.75
    B = intensidade * np.exp(-(x / sigma) ** 2)

    fluxo = n_espiras * area * B

    fem = -np.gradient(fluxo, t)

    # Corrente aproximada com resistência fixa de 10 ohms.
    resistencia = 10.0
    corrente = fem / resistencia

    return t, x, B, fluxo, fem, corrente


@st.cache_data
def gerar_animacao_lenz(
    velocidade,
    intensidade,
    area,
    n_espiras,
    duracao_ms,
):
    t, x, B, fluxo, fem, corrente = dados_inducao(
        velocidade,
        intensidade,
        area,
        n_espiras,
    )

    # Reduzimos o número de frames para manter a aplicação leve.
    indices = np.linspace(0, len(t) - 1, 70).astype(int)

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.6, 0.4],
        subplot_titles=(
            "Ímã → espira",
            "Força eletromotriz induzida",
        ),
    )

    # Espira.
    fig.add_trace(
        go.Scatter(
            x=[0, 0],
            y=[-1.6, 1.6],
            mode="lines",
            line=dict(width=8),
            name="Espira",
        ),
        row=1,
        col=1,
    )

    # Ímã inicial.
    xi = x[indices[0]]
    fig.add_trace(
        go.Scatter(
            x=[xi - 1.0, xi, xi, xi - 1.0, xi - 1.0],
            y=[-0.7, -0.7, 0.7, 0.7, -0.7],
            fill="toself",
            mode="lines",
            line=dict(color="black"),
            name="Ímã",
        ),
        row=1,
        col=1,
    )

    # Gráfico da fem.
    fig.add_trace(
        go.Scatter(
            x=t,
            y=fem,
            mode="lines",
            line=dict(width=3),
            name="ε",
        ),
        row=1,
        col=2,
    )

    # Marcador da posição temporal.
    fig.add_trace(
        go.Scatter(
            x=[t[0]],
            y=[fem[0]],
            mode="markers",
            marker=dict(size=10),
            name="Instante",
        ),
        row=1,
        col=2,
    )

    frames = []

    for idx in indices:
        xi = x[idx]
        cor = "#ef4444" if fem[idx] >= 0 else "#3b82f6"

        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=[xi - 1.0, xi, xi, xi - 1.0, xi - 1.0],
                        y=[-0.7, -0.7, 0.7, 0.7, -0.7],
                        fill="toself",
                        mode="lines",
                        line=dict(color="black"),
                    ),
                    go.Scatter(
                        x=t,
                        y=fem,
                        mode="lines",
                        line=dict(width=3),
                    ),
                    go.Scatter(
                        x=[t[idx]],
                        y=[fem[idx]],
                        mode="markers",
                        marker=dict(size=11, color=cor),
                    ),
                ],
                traces=[1, 2, 3],
                name=f"f{idx}",
            )
        )

    fig.frames = frames

    fig.update_layout(
        height=500,
        template="plotly_white",
        margin=dict(l=10, r=10, t=55, b=10),
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.0,
                "y": 1.13,
                "buttons": [
                    {
                        "label": "▶ Iniciar",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {
                                    "duration": duracao_ms,
                                    "redraw": True,
                                },
                                "fromcurrent": True,
                                "mode": "immediate",
                            },
                        ],
                    },
                    {
                        "label": "❚❚ Pausar",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {
                                    "duration": 0,
                                    "redraw": False,
                                },
                                "mode": "immediate",
                            },
                        ],
                    },
                ],
            }
        ],
    )

    fig.update_xaxes(
        range=[-3.5, 3.5],
        showgrid=False,
        zeroline=False,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        range=[-2.2, 2.2],
        showgrid=False,
        zeroline=False,
        row=1,
        col=1,
    )

    fig.update_xaxes(
        title_text="Tempo (s)",
        row=1,
        col=2,
    )
    fig.update_yaxes(
        title_text="ε (V)",
        row=1,
        col=2,
    )

    return fig


# ============================================================
# 5. GERADOR
# ============================================================

@st.cache_data
def dados_gerador(B, area, n, frequencia):
    f = np.linspace(0, 2 * frequencia, 500)
    omega = 2 * np.pi * frequencia
    tempo = np.linspace(0, 2 / frequencia, 500)

    fem_max = n * B * area * omega
    fem = fem_max * np.sin(omega * tempo)

    return tempo, fem, fem_max


# ============================================================
# 6. TRANSFORMADOR
# ============================================================

def calcular_transformador(v1, n1, n2):
    if n1 <= 0:
        return 0.0
    return v1 * n2 / n1


# ============================================================
# 7. ONDA ELETROMAGNÉTICA
# ============================================================

@st.cache_data
def gerar_onda_eletromagnetica(
    frequencia,
    amplitude_e,
    fase,
    comprimento,
):
    x = np.linspace(0, 2 * comprimento, 300)

    k = 2 * np.pi / comprimento
    omega = 2 * np.pi * frequencia

    # Para visualização estática em um instante t.
    E = amplitude_e * np.sin(k * x - omega * 0 + fase)

    # Em unidades normalizadas, B = E/c.
    B = (amplitude_e / C) * np.sin(
        k * x - omega * 0 + fase
    )

    return x, E, B


@st.cache_data
def gerar_animacao_onda(
    frequencia,
    amplitude_e,
    comprimento,
    duracao_ms,
):
    x = np.linspace(0, 2 * comprimento, 180)
    k = 2 * np.pi / comprimento

    frames = []

    fases = np.linspace(0, 2 * np.pi, 50)

    for fase in fases:
        E = amplitude_e * np.sin(k * x - fase)
        B = np.sin(k * x - fase)

        frames.append(
            go.Frame(
                data=[
                    go.Scatter3d(
                        x=x,
                        y=np.zeros_like(x),
                        z=E,
                        mode="lines",
                        line=dict(width=5),
                    ),
                    go.Scatter3d(
                        x=x,
                        y=B,
                        z=np.zeros_like(x),
                        mode="lines",
                        line=dict(width=5),
                    ),
                ],
                traces=[0, 1],
            )
        )

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=np.zeros_like(x),
                z=np.zeros_like(x),
                mode="lines",
                line=dict(width=5),
                name="E",
            ),
            go.Scatter3d(
                x=x,
                y=np.zeros_like(x),
                z=np.zeros_like(x),
                mode="lines",
                line=dict(width=5),
                name="B",
            ),
        ],
        frames=frames,
    )

    fig.update_layout(
        height=560,
        template="plotly_white",
        margin=dict(l=0, r=0, t=45, b=0),
        scene=dict(
            xaxis_title="Propagação k",
            yaxis_title="Campo B",
            zaxis_title="Campo E",
            xaxis=dict(range=[0, 2 * comprimento]),
            yaxis=dict(range=[-1.2, 1.2]),
            zaxis=dict(range=[-amplitude_e * 1.3, amplitude_e * 1.3]),
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=0.7)
            ),
        ),
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.0,
                "y": 1.1,
                "buttons": [
                    {
                        "label": "▶ Transmitir",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {
                                    "duration": duracao_ms,
                                    "redraw": True,
                                },
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "❚❚ Pausar",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {
                                    "duration": 0,
                                    "redraw": False,
                                }
                            },
                        ],
                    },
                ],
            }
        ],
    )

    return fig


# ============================================================
# 8. DESAFIO
# ============================================================

DESAFIOS = [
    {
        "pergunta": "Se a corrente em um fio retilíneo aumentar, o módulo de B próximo ao fio:",
        "opcoes": [
            "Diminui",
            "Aumenta",
            "Permanece igual",
            "Muda apenas de direção",
        ],
        "resposta": "Aumenta",
        "explicacao": "Para um fio retilíneo ideal, B = μ₀I/(2πr). Portanto, B é diretamente proporcional à corrente.",
    },
    {
        "pergunta": "Uma carga positiva entra perpendicularmente em um campo magnético uniforme. A força magnética é:",
        "opcoes": [
            "Paralela à velocidade",
            "Perpendicular à velocidade",
            "Sempre nula",
            "Paralela ao campo magnético",
        ],
        "resposta": "Perpendicular à velocidade",
        "explicacao": "A força magnética é q(v × B), portanto é perpendicular a v e a B quando v é perpendicular a B.",
    },
    {
        "pergunta": "Na indução eletromagnética, uma corrente induzida aparece quando:",
        "opcoes": [
            "Existe apenas um campo magnético estático",
            "O fluxo magnético através do circuito varia",
            "A resistência é infinita",
            "A carga do fio é zero",
        ],
        "resposta": "O fluxo magnético através do circuito varia",
        "explicacao": "A Lei de Faraday relaciona a fem induzida à variação temporal do fluxo magnético.",
    },
    {
        "pergunta": "Em uma onda eletromagnética plana no vácuo, E, B e a direção de propagação são:",
        "opcoes": [
            "Paralelos",
            "Todos na mesma direção",
            "Mutuamente perpendiculares",
            "Sempre opostos",
        ],
        "resposta": "Mutuamente perpendiculares",
        "explicacao": "Para uma onda eletromagnética plana, E é perpendicular a B e ambos são perpendiculares à direção de propagação.",
    },
]


# ============================================================
# CABEÇALHO
# ============================================================

st.markdown(
    '<div class="main-title">🧲 Eletromagnetismo Visual 2.0</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Laboratório virtual: observe, altere parâmetros, faça previsões e explique os fenômenos."
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# ABAS
# ============================================================

tabs = st.tabs(
    [
        "1. Campos",
        "2. Oersted",
        "3. Força de Lorentz",
        "4. Faraday & Lenz",
        "5. Geradores & Transformadores",
        "6. NFC",
        "7. Ondas & Antenas",
        "🎯 Desafio",
    ]
)


# ============================================================
# ABA 1 — CAMPOS
# ============================================================

with tabs[0]:

    st.markdown(
        """
        <div class="concept-card">
        <b>Campo elétrico:</b> uma carga elétrica cria uma região do espaço
        na qual outra carga pode sofrer força.
        <br><br>
        Para uma carga puntiforme:
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.latex(r"E = \frac{k|q|}{r^2}")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<div class="param-box">', unsafe_allow_html=True)

        q_campo = st.slider(
            "Carga fonte q",
            -10.0,
            10.0,
            1.0,
            0.5,
            key="q_campo",
        )

        st.markdown(
            """
            **Observe:**
            - carga positiva → campo aponta para fora;
            - carga negativa → campo aponta para dentro.
            """
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        if q_campo == 0:
            st.info("Escolha uma carga diferente de zero.")
        else:
            st.plotly_chart(
                gerar_campo_eletrico(q_campo, 1.0, 19),
                use_container_width=True,
                config={"displayModeBar": False},
            )

    st.markdown("---")

    st.markdown(
        """
        <div class="formula-card">
        <b>Ideia-chave:</b> o campo não é a força. Ele representa a força
        que seria exercida por unidade de carga de teste.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# ABA 2 — OERSTED
# ============================================================

with tabs[1]:

    st.markdown(
        """
        <div class="concept-card">
        <b>Experimento de Oersted:</b> uma corrente elétrica produz um
        campo magnético ao redor do condutor. O sentido pode ser obtido
        pela <b>regra da mão direita</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 2.3])

    with col1:
        st.markdown('<div class="param-box">', unsafe_allow_html=True)

        corrente = st.slider(
            "Corrente I (A)",
            -10.0,
            10.0,
            5.0,
            0.5,
            key="corrente_oersted",
        )

        raio = st.slider(
            "Distância ao fio r (m)",
            0.25,
            5.0,
            1.0,
            0.25,
            key="raio_oersted",
        )

        B_ponto = (
            MU0 * abs(corrente) / (2 * np.pi * raio)
            if abs(corrente) > 0
            else 0
        )

        st.markdown(
            f"""
            **Campo no ponto escolhido:**

            B = {B_ponto:.3e} T
            """
        )

        st.markdown(
            """
            A relação usada é:

            B = μ₀I / (2πr)
            """
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.plotly_chart(
            gerar_grafico_oersted(corrente),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    st.markdown(
        """
        <div class="success-card">
        <b>Experimento mental:</b> mantenha I constante e dobre r.
        O campo magnético deve cair pela metade.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# ABA 3 — LORENTZ
# ============================================================

with tabs[2]:

    st.markdown(
        """
        <div class="concept-card">
        <b>Força de Lorentz:</b> uma partícula carregada submetida a campos
        elétrico e magnético sofre a força
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.latex(
        r"\vec F = q(\vec E + \vec v \times \vec B)"
    )

    col1, col2 = st.columns([1, 2.2])

    with col1:
        st.markdown('<div class="param-box">', unsafe_allow_html=True)

        q_l = st.number_input(
            "Carga q (C)",
            -5.0,
            5.0,
            1.0,
            0.1,
            key="lorentz_q",
        )

        m_l = st.number_input(
            "Massa m (kg)",
            0.01,
            10.0,
            1.0,
            0.01,
            key="lorentz_m",
        )

        st.markdown("**Velocidade inicial (m/s)**")

        vx = st.slider(-20.0, 20.0, 10.0, 1.0, key="vx_l")
        vy = st.slider(-20.0, 20.0, 0.0, 1.0, key="vy_l")
        vz = st.slider(-20.0, 20.0, 0.0, 1.0, key="vz_l")

        st.markdown("**Campo elétrico (V/m)**")

        ex = st.slider(-10.0, 10.0, 0.0, 0.5, key="ex_l")
        ey = st.slider(-10.0, 10.0, 0.0, 0.5, key="ey_l")
        ez = st.slider(-10.0, 10.0, 0.0, 0.5, key="ez_l")

        st.markdown("**Campo magnético (T)**")

        bx = st.slider(-2.0, 2.0, 0.0, 0.1, key="bx_l")
        by = st.slider(-2.0, 2.0, 0.0, 0.1, key="by_l")
        bz = st.slider(-2.0, 2.0, 1.0, 0.1, key="bz_l")

        tempo_l = st.slider(
            "Tempo (s)",
            0.5,
            20.0,
            5.0,
            0.5,
            key="tempo_l",
        )

        dt_l = st.select_slider(
            "Passo dt",
            options=[0.001, 0.002, 0.005, 0.01, 0.02],
            value=0.01,
            key="dt_l",
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:

        v0_l = np.array([vx, vy, vz], dtype=float)
        E_l = np.array([ex, ey, ez], dtype=float)
        B_l = np.array([bx, by, bz], dtype=float)

        resultado, erro = simular_lorentz(
            q_l,
            m_l,
            v0_l,
            E_l,
            B_l,
            tempo_l,
            dt_l,
        )

        if erro:
            st.error(erro)
        else:
            tempo, estado = resultado

            x = estado[:, 0]
            y = estado[:, 1]
            z = estado[:, 2]

            velocidade = np.linalg.norm(
                estado[:, 3:6],
                axis=1,
            )

            fig = go.Figure()

            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="lines",
                    line=dict(width=5),
                    name="Trajetória",
                )
            )

            fig.add_trace(
                go.Scatter3d(
                    x=[x[0]],
                    y=[y[0]],
                    z=[z[0]],
                    mode="markers",
                    marker=dict(size=7),
                    name="Inicial",
                )
            )

            fig.add_trace(
                go.Scatter3d(
                    x=[x[-1]],
                    y=[y[-1]],
                    z=[z[-1]],
                    mode="markers",
                    marker=dict(size=7),
                    name="Final",
                )
            )

            fig.update_layout(
                height=620,
                template="plotly_white",
                title="Trajetória da partícula",
                scene=dict(
                    xaxis_title="x (m)",
                    yaxis_title="y (m)",
                    zaxis_title="z (m)",
                ),
                margin=dict(l=0, r=0, t=50, b=0),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )

            metric_row(
                [
                    ("Velocidade inicial", f"{np.linalg.norm(v0_l):.3f} m/s"),
                    ("Velocidade final", f"{velocidade[-1]:.3f} m/s"),
                    ("|B|", f"{np.linalg.norm(B_l):.3f} T"),
                ]
            )

            st.markdown(
                """
                <div class="alert-card">
                <b>Teste:</b> coloque E = 0, v perpendicular a B e q > 0.
                A trajetória tende a ser circular. Inverta q e observe
                a inversão do sentido da curvatura.
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# ABA 4 — FARADAY & LENZ
# ============================================================

with tabs[3]:

    st.markdown(
        """
        <div class="concept-card">
        <b>Lei de Faraday:</b> a fem induzida depende da taxa de variação
        do fluxo magnético.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.latex(
        r"\mathcal{E}=-N\frac{d\Phi_B}{dt}"
    )

    col1, col2 = st.columns([1, 2.2])

    with col1:
        st.markdown('<div class="param-box">', unsafe_allow_html=True)

        vel_f = st.slider(
            "Velocidade do ímã",
            -2.0,
            2.0,
            1.0,
            0.1,
            key="vel_f",
        )

        intensidade_f = st.slider(
            "Campo máximo (T)",
            0.1,
            2.0,
            1.0,
            0.1,
            key="int_f",
        )

        area_f = st.slider(
            "Área da espira (m²)",
            0.01,
            2.0,
            0.25,
            0.01,
            key="area_f",
        )

        n_f = st.slider(
            "Número de espiras",
            1,
            100,
            20,
            1,
            key="n_f",
        )

        velocidade_anim = st.select_slider(
            "Velocidade da animação",
            options=[20, 50, 90],
            value=50,
            key="anim_f",
            format_func=lambda x: (
                "Rápida" if x == 20
                else "Média" if x == 50
                else "Lenta"
            ),
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:

        fig_f = gerar_animacao_lenz(
            vel_f,
            intensidade_f,
            area_f,
            n_f,
            velocidade_anim,
        )

        st.plotly_chart(
            fig_f,
            use_container_width=True,
            config={"displayModeBar": False},
        )

        t, x, B, fluxo, fem, corrente = dados_inducao(
            vel_f,
            intensidade_f,
            area_f,
            n_f,
        )

        metric_row(
            [
                ("Fluxo máximo", f"{np.max(np.abs(fluxo)):.3e} Wb"),
                ("Fem máxima", f"{np.max(np.abs(fem)):.3e} V"),
                ("Corrente máx. aprox.", f"{np.max(np.abs(corrente)):.3e} A"),
            ]
        )


# ============================================================
# ABA 5 — GERADORES & TRANSFORMADORES
# ============================================================

with tabs[4]:

    sub1, sub2 = st.tabs(
        ["⚙️ Gerador", "🔌 Transformador"]
    )

    # --------------------------------------------------------
    # GERADOR
    # --------------------------------------------------------

    with sub1:

        st.markdown(
            """
            <div class="concept-card">
            Um gerador converte energia mecânica em energia elétrica
            por indução eletromagnética.
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns([1, 2])

        with col1:

            B_g = st.slider(
                "Campo B (T)",
                0.1,
                2.0,
                0.5,
                0.1,
                key="B_g",
            )

            A_g = st.slider(
                "Área da espira (m²)",
                0.01,
                1.0,
                0.1,
                0.01,
                key="A_g",
            )

            N_g = st.slider(
                "Número de espiras",
                1,
                500,
                100,
                1,
                key="N_g",
            )

            f_g = st.slider(
                "Frequência de rotação (Hz)",
                0.1,
                20.0,
                5.0,
                0.1,
                key="f_g",
            )

        with col2:

            tempo_g, fem_g, fem_max = dados_gerador(
                B_g,
                A_g,
                N_g,
                f_g,
            )

            fig_g = go.Figure()

            fig_g.add_trace(
                go.Scatter(
                    x=tempo_g,
                    y=fem_g,
                    mode="lines",
                    line=dict(width=3),
                    name="ε(t)",
                )
            )

            fig_g.update_layout(
                title="Tensão induzida no gerador",
                xaxis_title="Tempo (s)",
                yaxis_title="Fem (V)",
                template="plotly_white",
                height=470,
            )

            st.plotly_chart(
                fig_g,
                use_container_width=True,
                config={"displayModeBar": False},
            )

            st.metric(
                "Fem máxima",
                f"{fem_max:.2f} V",
            )

        st.latex(
            r"\mathcal{E}_{max}=NBA\omega"
        )

    # --------------------------------------------------------
    # TRANSFORMADOR
    # --------------------------------------------------------

    with sub2:

        st.markdown(
            """
            <div class="concept-card">
            Em um transformador ideal, a razão entre as tensões é igual
            à razão entre o número de espiras.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.latex(
            r"\frac{V_2}{V_1}=\frac{N_2}{N_1}"
        )

        col1, col2 = st.columns([1, 1.5])

        with col1:

            V1 = st.number_input(
                "Tensão primária V₁ (V)",
                1.0,
                10000.0,
                127.0,
                key="V1_t",
            )

            N1 = st.number_input(
                "Espiras primárias N₁",
                1,
                10000,
                500,
                key="N1_t",
            )

            N2 = st.number_input(
                "Espiras secundárias N₂",
                1,
                10000,
                100,
                key="N2_t",
            )

        with col2:

            V2 = calcular_transformador(
                V1,
                N1,
                N2,
            )

            tipo = (
                "Elevador"
                if V2 > V1
                else "Abaixador"
                if V2 < V1
                else "1:1"
            )

            st.metric(
                "Tensão secundária",
                f"{V2:.2f} V",
            )

            st.metric(
                "Tipo",
                tipo,
            )

            fig_t = go.Figure()

            fig_t.add_trace(
                go.Bar(
                    x=["Primário", "Secundário"],
                    y=[V1, V2],
                )
            )

            fig_t.update_layout(
                title="Comparação das tensões",
                yaxis_title="Tensão (V)",
                template="plotly_white",
            )

            st.plotly_chart(
                fig_t,
                use_container_width=True,
                config={"displayModeBar": False},
            )


# ============================================================
# ABA 6 — NFC
# ============================================================

with tabs[5]:

    st.markdown(
        """
        <div class="concept-card">
        <b>NFC:</b> a comunicação de curto alcance utiliza acoplamento
        eletromagnético entre bobinas. O dispositivo ativo cria um campo
        magnético alternado e o dispositivo passivo pode obter energia
        por indução.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.4, 1])

    with col1:

        distancia = st.slider(
            "Distância entre leitor e cartão (cm)",
            0.5,
            10.0,
            2.0,
            0.5,
            key="dist_nfc",
        )

        frequencia_nfc = 13.56e6

        # Modelo didático de decaimento visual.
        acoplamento = 1 / (
            1 + (distancia / 2.0) ** 3
        )

        fig_nfc = go.Figure()

        # Smartphone.
        fig_nfc.add_shape(
            type="rect",
            x0=0,
            y0=0,
            x1=2.5,
            y1=5,
            line=dict(width=3),
            fillcolor="#f1f5f9",
        )

        fig_nfc.add_annotation(
            x=1.25,
            y=2.5,
            text="<b>Leitor</b><br>Bobina ativa",
            showarrow=False,
        )

        # Cartão.
        pos = 4 + distancia * 0.35

        fig_nfc.add_shape(
            type="rect",
            x0=pos,
            y0=1,
            x1=pos + 2.4,
            y1=4,
            line=dict(width=3),
            fillcolor="#f1f5f9",
        )

        fig_nfc.add_annotation(
            x=pos + 1.2,
            y=2.5,
            text="<b>Cartão</b><br>Bobina passiva",
            showarrow=False,
        )

        # Linhas de campo.
        for i in range(5):
            x0 = 2.7 + i * 0.12
            x1 = pos - 0.2 - i * 0.05

            fig_nfc.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[2.5 + i * 0.25, 2.5 + i * 0.25],
                    mode="lines",
                    line=dict(
                        width=2,
                        dash="dash",
                    ),
                    showlegend=False,
                )
            )

        fig_nfc.update_layout(
            height=500,
            template="plotly_white",
            showlegend=False,
            xaxis=dict(
                range=[-0.5, 10],
                visible=False,
            ),
            yaxis=dict(
                range=[-0.5, 5.5],
                visible=False,
            ),
            margin=dict(l=0, r=0, t=20, b=0),
        )

        st.plotly_chart(
            fig_nfc,
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with col2:

        st.markdown('<div class="param-box">', unsafe_allow_html=True)

        st.metric(
            "Frequência NFC",
            "13,56 MHz",
        )

        st.metric(
            "Acoplamento — modelo didático",
            f"{acoplamento * 100:.1f}%",
        )

        st.markdown(
            """
            **Sequência física simplificada**

            1. O leitor alimenta sua bobina.
            2. A corrente alternada produz campo magnético variável.
            3. O campo atravessa a bobina do cartão.
            4. O fluxo variável induz uma tensão.
            5. O circuito do cartão utiliza a energia disponível.
            6. O cartão pode modificar sua resposta eletromagnética.
            """
        )

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="alert-card">
            O percentual mostrado é apenas um <b>índice visual de acoplamento</b>,
            não uma eficiência real de um sistema NFC.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# ABA 7 — ONDAS E ANTENAS
# ============================================================

with tabs[6]:

    st.markdown(
        """
        <div class="concept-card">
        Uma onda eletromagnética pode ser representada como campos
        elétrico e magnético oscilantes, perpendiculares entre si e
        à direção de propagação.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 2.2])

    with col1:

        f_onda = st.select_slider(
            "Frequência",
            options=[
                1e6,
                10e6,
                100e6,
                2.4e9,
                5e9,
                10e9,
            ],
            value=2.4e9,
            format_func=lambda x: (
                f"{x/1e9:.1f} GHz"
                if x >= 1e9
                else f"{x/1e6:.0f} MHz"
            ),
            key="f_onda",
        )

        amp_onda = st.slider(
            "Amplitude visual de E",
            0.2,
            2.0,
            1.0,
            0.1,
            key="amp_onda",
        )

        lambda_onda = C / f_onda

        st.metric(
            "Comprimento de onda",
            f"{lambda_onda:.4g} m",
        )

        st.markdown(
            f"""
            **Relação fundamental**

            λ = c / f

            Para esta frequência:

            **λ ≈ {lambda_onda:.4g} m**
            """
        )

        velocidade_onda = st.select_slider(
            "Velocidade da animação",
            options=[30, 60, 100],
            value=60,
            key="vel_onda",
            format_func=lambda x: (
                "Rápida" if x == 30
                else "Média" if x == 60
                else "Lenta"
            ),
        )

    with col2:

        fig_onda = gerar_animacao_onda(
            f_onda,
            amp_onda,
            lambda_onda,
            velocidade_onda,
        )

        st.plotly_chart(
            fig_onda,
            use_container_width=True,
            config={"displayModeBar": False},
        )

    st.markdown(
        """
        <div class="formula-card">
        <b>Relação vetorial:</b>
        &nbsp;&nbsp; E ⟂ B ⟂ direção de propagação
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# ABA DESAFIO
# ============================================================

with tabs[7]:

    st.markdown(
        """
        <div class="concept-card">
        <b>Modo Desafio:</b> primeiro faça uma previsão. Depois confira
        a resposta e tente explicar o fenômeno usando uma lei física.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "desafio_indice" not in st.session_state:
        st.session_state.desafio_indice = 0

    if "desafio_verificado" not in st.session_state:
        st.session_state.desafio_verificado = False

    desafio = DESAFIOS[
        st.session_state.desafio_indice
    ]

    st.markdown(
        f"### Desafio {st.session_state.desafio_indice + 1} de {len(DESAFIOS)}"
    )

    st.write(desafio["pergunta"])

    resposta = st.radio(
        "Escolha uma alternativa:",
        desafio["opcoes"],
        key=f"resposta_{st.session_state.desafio_indice}",
    )

    col_a, col_b = st.columns(2)

    with col_a:

        if st.button(
            "🔎 Verificar resposta",
            use_container_width=True,
        ):
            st.session_state.desafio_verificado = True

    with col_b:

        if st.button(
            "➡️ Próximo desafio",
            use_container_width=True,
        ):
            st.session_state.desafio_indice = (
                (st.session_state.desafio_indice + 1)
                % len(DESAFIOS)
            )
            st.session_state.desafio_verificado = False
            st.rerun()

    if st.session_state.desafio_verificado:

        if resposta == desafio["resposta"]:

            st.success("✅ Resposta correta!")

        else:

            st.error(
                f"❌ A resposta correta é: {desafio['resposta']}"
            )

        st.info(
            f"**Explicação:** {desafio['explicacao']}"
        )


# ============================================================
# RODAPÉ
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        color:#94a3b8;
        font-size:0.85rem;
        padding:1rem;
    ">
        🧲 <b>Física Visual — Eletromagnetismo 2.0</b><br>
        Laboratório virtual para exploração, experimentação e aprendizagem.
    </div>
    """,
    unsafe_allow_html=True,
)
