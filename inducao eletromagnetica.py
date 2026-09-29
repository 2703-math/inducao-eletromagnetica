import streamlit as st
import numpy as np
import plotly.graph_objects as go

# ============================================================
# CONFIGURAÇÃO
# ============================================================

st.set_page_config(
    page_title="Simulação: Força de Lorentz",
    layout="wide"
)

st.title("⚡ Simulação da Força de Lorentz")

st.write(
    "Trajetória de uma partícula carregada sob influência "
    "de campos elétricos e magnéticos utilizando o método RK4."
)

# ============================================================
# PARÂMETROS
# ============================================================

st.sidebar.header("⚛️ Partícula")

q = st.sidebar.number_input(
    "Carga q (C)",
    min_value=-10.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)

m = st.sidebar.number_input(
    "Massa m (kg)",
    min_value=0.01,
    max_value=10.0,
    value=1.0,
    step=0.01
)

# ------------------------------------------------------------

st.sidebar.header("🚀 Velocidade Inicial (m/s)")

v0x = st.sidebar.slider(
    "v₀x",
    -20.0, 20.0, 10.0
)

v0y = st.sidebar.slider(
    "v₀y",
    -20.0, 20.0, 0.0
)

v0z = st.sidebar.slider(
    "v₀z",
    -20.0, 20.0, 0.0
)

# ------------------------------------------------------------

st.sidebar.header("🔵 Campo Elétrico E (V/m)")

Ex = st.sidebar.slider(
    "Ex",
    -10.0, 10.0, 0.0
)

Ey = st.sidebar.slider(
    "Ey",
    -10.0, 10.0, 0.0
)

Ez = st.sidebar.slider(
    "Ez",
    -10.0, 10.0, 0.0
)

# ------------------------------------------------------------

st.sidebar.header("🧲 Campo Magnético B (T)")

Bx = st.sidebar.slider(
    "Bx",
    -5.0, 5.0, 0.0
)

By = st.sidebar.slider(
    "By",
    -5.0, 5.0, 0.0
)

Bz = st.sidebar.slider(
    "Bz",
    -5.0, 5.0, 1.0
)

# ------------------------------------------------------------

st.sidebar.header("⏱️ Simulação")

t_max = st.sidebar.slider(
    "Tempo de simulação (s)",
    1.0,
    50.0,
    10.0
)

dt = st.sidebar.number_input(
    "Passo de tempo (dt)",
    min_value=0.0001,
    max_value=0.1,
    value=0.01,
    format="%.4f"
)

# ============================================================
# FUNÇÃO DA EQUAÇÃO DE LORENTZ
# ============================================================

def derivadas(state, q, m, E, B):

    # state = [x, y, z, vx, vy, vz]

    v = state[3:6]

    # Força magnética:
    # Fm = q(v x B)

    forca_magnetica = q * np.cross(v, B)

    # Força elétrica:
    # Fe = qE

    forca_eletrica = q * E

    # Força total

    F = forca_eletrica + forca_magnetica

    # Aceleração

    aceleracao = F / m

    # dr/dt = v
    # dv/dt = a

    return np.concatenate((v, aceleracao))


# ============================================================
# SIMULAÇÃO RK4
# ============================================================

@st.cache_data
def simular(
    q, m,
    v0x, v0y, v0z,
    Ex, Ey, Ez,
    Bx, By, Bz,
    t_max, dt
):

    E = np.array([Ex, Ey, Ez], dtype=float)
    B = np.array([Bx, By, Bz], dtype=float)

    r0 = np.array([0.0, 0.0, 0.0])

    v0 = np.array([
        v0x,
        v0y,
        v0z
    ])

    # --------------------------------------------------------
    # Verificação da estabilidade
    # --------------------------------------------------------

    B_modulo = np.linalg.norm(B)

    if B_modulo > 0:

        omega = abs(q) * B_modulo / m

        # Parâmetro adimensional aproximado
        estabilidade = omega * dt

        if estabilidade > 1.0:

            return None, (
                f"O passo de tempo é muito grande para esses parâmetros. "
                f"ω·dt = {estabilidade:.2f}. "
                f"Reduza o dt ou aumente a massa."
            )

    # --------------------------------------------------------
    # Número de passos
    # --------------------------------------------------------

    passos = int(np.ceil(t_max / dt)) + 1

    # Proteção contra simulações exageradamente grandes

    if passos > 100000:

        return None, (
            "A simulação possui mais de 100.000 passos. "
            "Aumente o valor de dt."
        )

    # --------------------------------------------------------
    # Pré-alocação
    # --------------------------------------------------------

    state = np.zeros((passos, 6), dtype=float)

    state[0] = np.concatenate((r0, v0))

    # --------------------------------------------------------
    # RK4
    # --------------------------------------------------------

    for i in range(1, passos):

        y = state[i - 1]

        k1 = dt * derivadas(
            y, q, m, E, B
        )

        k2 = dt * derivadas(
            y + 0.5 * k1,
            q, m, E, B
        )

        k3 = dt * derivadas(
            y + 0.5 * k2,
            q, m, E, B
        )

        k4 = dt * derivadas(
            y + k3,
            q, m, E, B
        )

        state[i] = (
            y
            + (k1 + 2*k2 + 2*k3 + k4) / 6
        )

        # Segurança contra explosão numérica

        if not np.all(np.isfinite(state[i])):

            return None, (
                "A simulação ficou numericamente instável. "
                "Tente diminuir o dt."
            )

    return state, None


# ============================================================
# BOTÃO
# ============================================================

executar = st.sidebar.button(
    "▶️ Executar simulação",
    type="primary"
)

# ============================================================
# EXECUÇÃO
# ============================================================

if executar:

    with st.spinner("Calculando trajetória..."):

        state, erro = simular(
            q, m,
            v0x, v0y, v0z,
            Ex, Ey, Ez,
            Bx, By, Bz,
            t_max, dt
        )

    # --------------------------------------------------------
    # Erro
    # --------------------------------------------------------

    if erro is not None:

        st.error(erro)

    else:

        # ----------------------------------------------------
        # Dados
        # ----------------------------------------------------

        x = state[:, 0]
        y = state[:, 1]
        z = state[:, 2]

        vx = state[:, 3]
        vy = state[:, 4]
        vz = state[:, 5]

        velocidade = np.sqrt(
            vx**2 + vy**2 + vz**2
        )

        # ----------------------------------------------------
        # Informações
        # ----------------------------------------------------

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Passos",
                len(state)
            )

        with col2:
            st.metric(
                "|B|",
                f"{np.linalg.norm([Bx, By, Bz]):.3f} T"
            )

        with col3:
            st.metric(
                "Velocidade final",
                f"{velocidade[-1]:.3f} m/s"
            )

        # ----------------------------------------------------
        # GRÁFICO 3D
        # ----------------------------------------------------

        fig = go.Figure()

        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line=dict(
                    width=4
                ),
                name="Trajetória"
            )
        )

        # Posição inicial

        fig.add_trace(
            go.Scatter3d(
                x=[x[0]],
                y=[y[0]],
                z=[z[0]],
                mode="markers",
                marker=dict(
                    size=7
                ),
                name="Início"
            )
        )

        # Posição final

        fig.add_trace(
            go.Scatter3d(
                x=[x[-1]],
                y=[y[-1]],
                z=[z[-1]],
                mode="markers",
                marker=dict(
                    size=7
                ),
                name="Final"
            )
        )

        fig.update_layout(

            title="Movimento sob Força de Lorentz",

            scene=dict(
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                zaxis_title="Z (m)"
            ),

            margin=dict(
                l=0,
                r=0,
                b=0,
                t=40
            ),

            template="plotly_dark",

            height=700
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ----------------------------------------------------
        # GRÁFICO DA VELOCIDADE
        # ----------------------------------------------------

        tempo = np.arange(len(state)) * dt

        fig_vel = go.Figure()

        fig_vel.add_trace(
            go.Scatter(
                x=tempo,
                y=velocidade,
                mode="lines",
                name="|v|"
            )
        )

        fig_vel.update_layout(
            title="Módulo da velocidade",
            xaxis_title="Tempo (s)",
            yaxis_title="Velocidade (m/s)",
            template="plotly_dark"
        )

        st.plotly_chart(
            fig_vel,
            use_container_width=True
        )

else:

    st.info(
        "Configure os parâmetros na barra lateral "
        "e clique em **▶️ Executar simulação**."
    )
