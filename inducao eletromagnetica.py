import streamlit as st
import numpy as np
import plotly.graph_objects as go

# 1. Configuração e Otimização do Streamlit
st.set_page_config(page_title="Simulação: Força de Lorentz", layout="wide")
st.title("Simulação da Força de Lorentz")
st.write("Trajetória de uma partícula sob influência de campos elétricos e magnéticos usando RK4.")

# 2. Parâmetros na Barra Lateral
st.sidebar.header("Parâmetros da Partícula")
q = st.sidebar.number_input("Carga q (C)", value=1.0)
m = st.sidebar.number_input("Massa m (kg)", value=1.0, min_value=0.01)

st.sidebar.header("Velocidade Inicial (m/s)")
v0x = st.sidebar.slider("v0x", -20.0, 20.0, 10.0)
v0y = st.sidebar.slider("v0y", -20.0, 20.0, 0.0)
v0z = st.sidebar.slider("v0z", -20.0, 20.0, 0.0)

st.sidebar.header("Campo Elétrico E (V/m)")
Ex = st.sidebar.slider("Ex", -10.0, 10.0, 0.0)
Ey = st.sidebar.slider("Ey", -10.0, 10.0, 0.0)
Ez = st.sidebar.slider("Ez", -10.0, 10.0, 0.0)

st.sidebar.header("Campo Magnético B (T)")
Bx = st.sidebar.slider("Bx", -5.0, 5.0, 0.0)
By = st.sidebar.slider("By", -5.0, 5.0, 0.0)
Bz = st.sidebar.slider("Bz", -5.0, 5.0, 1.0)

st.sidebar.header("Controle da Simulação")
t_max = st.sidebar.slider("Tempo de Simulação (s)", 1.0, 50.0, 10.0)
dt = st.sidebar.number_input("Passo de tempo (dt)", value=0.01, min_value=0.001)

# Montando os vetores iniciais
E = np.array([Ex, Ey, Ez])
B = np.array([Bx, By, Bz])
r0 = np.array([0.0, 0.0, 0.0])
v0 = np.array([v0x, v0y, v0z])

# 3. Equações Diferenciais (Força de Lorentz)
def derivadas(state, q, m, E, B):
    # state = [x, y, z, vx, vy, vz]
    v = state[3:6]
    
    # Produto vetorial v x B usando numpy para máxima eficiência
    F_mag = np.cross(v, B)
    F = q * (E + F_mag)
    
    dv_dt = F / m
    dr_dt = v
    return np.concatenate((dr_dt, dv_dt))

# 4. LOOP SEGURO: Cálculo de passos exatos evita travamento (Hanging)
passos = int(t_max / dt)

# Pré-alocação de memória (muito mais rápido que usar listas e append)
state = np.zeros((passos, 6))
state[0] = np.concatenate((r0, v0))

# RK4 Vectorizado
for i in range(1, passos):
    y = state[i-1]
    
    k1 = dt * derivadas(y, q, m, E, B)
    k2 = dt * derivadas(y + 0.5*k1, q, m, E, B)
    k3 = dt * derivadas(y + 0.5*k2, q, m, E, B)
    k4 = dt * derivadas(y + k3, q, m, E, B)
    
    state[i] = y + (k1 + 2*k2 + 2*k3 + k4) / 6.0

# 5. Extração dos dados e Plotagem 3D
x = state[:, 0]
y = state[:, 1]
z = state[:, 2]

fig = go.Figure()
fig.add_trace(go.Scatter3d(
    x=x, y=y, z=z, 
    mode='lines',
    line=dict(color='cyan', width=4),
    name='Trajetória'
))

fig.update_layout(
    title="Movimento sob Força de Lorentz",
    scene=dict(
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        zaxis_title='Z (m)'
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)
