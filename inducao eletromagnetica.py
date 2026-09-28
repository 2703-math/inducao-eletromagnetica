import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

st.set_page_config(page_title="Física Visual: Eletromagnetismo", page_icon="🧲", layout="wide")

st.markdown("""
<style>
.main-title { font-size:2.2rem; font-weight:800; color:#0f172a; text-align:center; margin-bottom:0.3rem; }
.subtitle { font-size:1.1rem; color:#64748b; text-align:center; margin-bottom:2rem; }
.concept-card { background:#f8fafc; border-radius:12px; padding:1.2rem; border-left:4px solid #3b82f6; margin-bottom:1rem; color:#334155; }
.alert-card { background:#fffbeb; border-radius:12px; padding:1.2rem; border-left:4px solid #f59e0b; margin-bottom:1rem; color:#334155; }
.param-box { background:#fff; border:1px solid #e2e8f0; border-radius:10px; padding:1rem; margin-bottom:1rem; }
.highlight { color:#ef4444; font-weight:bold; }
.section-title { font-size:1.25rem; font-weight:700; color:#0f172a; margin-bottom:0.5rem; }
</style>
""", unsafe_allow_html=True)

# ==========================
# UTIL
# ==========================
def _check_plotly():
    v = go.__version__
    st.caption(f"Plotly v{v} instalado")
    major = int(v.split(".")[0])
    if major < 5:
        st.warning("Plotly < 5 pode não renderizar animações corretamente no Streamlit. Instale: `pip install 'plotly>=5'`")

# ==========================
# 1. OERSTED (estático)
# ==========================
@st.cache_data
def build_oersted(corrente):
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=[0,0], y=[0,0], z=[-5,5], mode='lines',
                               line=dict(color='#94a3b8', width=12), name='Fio', hoverinfo='skip'))
    if corrente == 0:
        fig.update_layout(height=350, margin=dict(l=0,r=0,t=0,b=0), showlegend=False,
                          scene=dict(xaxis=dict(range=[-5,5], visible=False),
                                     yaxis=dict(range=[-5,5], visible=False),
                                     zaxis=dict(range=[-5,5], visible=False),
                                     camera=dict(eye=dict(x=1.2,y=1.2,z=0.8))),
                          paper_bgcolor='white', plot_bgcolor='white')
        return fig

    sentido = 1 if corrente > 0 else -1
    cor = '#ef4444' if corrente > 0 else '#3b82f6'
    fig.add_trace(go.Scatter3d(x=[0,0], y=[0,0],
                               z=[-4 if corrente>0 else 4, 4 if corrente>0 else -4],
                               mode='lines', line=dict(color=cor, width=5), name='Corrente (I)'))
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[4 if corrente>0 else -4],
                               mode='markers', marker=dict(symbol='diamond', size=8, color=cor), name='Sentido'))

    xv, yv, zv, uv, vv, wv = [], [], [], [], [], []
    for z in [-2.5, 0, 2.5]:
        for r in [2, 4]:
            for theta in np.linspace(0, 2*np.pi, 12, endpoint=False):
                x, y = r*math.cos(theta), r*math.sin(theta)
                mag = abs(corrente) / 4.0
                xv.append(x); yv.append(y); zv.append(z)
                uv.append(-sentido * y / r * mag)
                vv.append(sentido * x / r * mag)
                wv.append(0)
                
    fig.add_trace(go.Cone(x=xv, y=yv, z=zv, u=uv, v=vv, w=wv,
                          colorscale='Reds' if corrente>0 else 'Blues',
                          sizemode='absolute', sizeref=0.6, showscale=False, name='Campo B'))
    fig.update_layout(height=350, margin=dict(l=0,r=0,t=0,b=0), showlegend=False,
                      scene=dict(xaxis=dict(range=[-5,5], visible=False),
                                 yaxis=dict(range=[-5,5], visible=False),
                                 zaxis=dict(range=[-5,5], visible=False),
                                 camera=dict(eye=dict(x=1.2,y=1.2,z=0.8))),
                      paper_bgcolor='white', plot_bgcolor='white')
    return fig

# ==========================
# 2. FARADAY & LENZ (animação)
# ==========================
@st.cache_data
def build_lenz_figure(speed_ms):
    n_frames = 40
    t_vals = np.linspace(0, 2*np.pi, n_frames)

    fig = make_subplots(rows=1, cols=2, column_widths=[0.7, 0.3], horizontal_spacing=0.05,
                        subplot_titles=("Ímã e Espira (Indução)", "Galvanômetro"))

    # Traços BASE
    fig.add_trace(go.Scatter(x=[0,0], y=[0,0], mode='lines', line=dict(color='red', width=5), name='S'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0,0], y=[0,0], mode='lines', line=dict(color='blue', width=5), name='N'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='lines+markers', marker=dict(symbol='arrow-right', size=15),
                             line=dict(color='black', width=4), name='Corrente'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0,0], y=[0,0], mode='lines', line=dict(color='black', width=4), name='Ponteiro'), row=1, col=2)

    x_espira = [0, 1, 1, 0, 0]; y_espira = [-1.5, -1.5, 1.5, 1.5, -1.5]
    fig.add_trace(go.Scatter(x=x_espira, y=y_espira, mode='lines', line=dict(color='#94a3b8', width=5), name='Espira', hoverinfo='skip'), row=1, col=1)
    
    th = np.linspace(0, np.pi, 50)
    fig.add_trace(go.Scatter(x=np.cos(th), y=np.sin(th), mode='lines', line=dict(color='#cbd5e1', width=3), hoverinfo='skip'), row=1, col=2)
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers', marker=dict(color='black', size=10), hoverinfo='skip'), row=1, col=2)

    frames = []
    for t in t_vals:
        x_c = -3.5 + 2*math.cos(t)
        v = -2*math.sin(t)
        I = -v
        if I > 0.15:
            ix, iy = [0.2, 0.8], [1.8, 1.8]; cor = '#ef4444'; txt = 'Corrente ↻ (opõe aumento de Φ)'
        elif I < -0.15:
            ix, iy = [0.8, 0.2], [1.8, 1.8]; cor = '#3b82f6'; txt = 'Corrente ↺ (opõe diminuição de Φ)'
        else:
            ix = iy = [0.5]; cor = 'rgba(0,0,0,0)'; txt = 'Sem corrente (v ≈ 0)'
            
        ang = np.pi/2 - I*0.4
        px, py = 0.9*math.cos(ang), 0.9*math.sin(ang)

        x_sul = [x_c, x_c+1.5, x_c+1.5, x_c, x_c]
        x_norte = [x_c-1.5, x_c, x_c, x_c-1.5, x_c-1.5]
        
        frames.append(go.Frame(
            data=[
                go.Scatter(x=x_sul, y=[-0.8,-0.8,0.8,0.8,-0.8], fill='toself', fillcolor='#ef4444', line=dict(color='black')),
                go.Scatter(x=x_norte, y=[-0.8,-0.8,0.8,0.8,-0.8], fill='toself', fillcolor='#3b82f6', line=dict(color='black')),
                go.Scatter(x=ix, y=iy, mode='lines+markers', marker=dict(symbol='arrow-right', size=15), line=dict(color=cor, width=4), text=txt, hoverinfo='text'),
                go.Scatter(x=[0, px], y=[0, py], mode='lines', line=dict(color=cor if cor!='rgba(0,0,0,0)' else 'black', width=4))
            ],
            traces=[0,1,2,3]
        ))

    fig.frames = frames
    fig.update_layout(
        height=400, showlegend=False, plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=10,r=10,t=40,b=10),
        updatemenus=[{
            "type":"buttons","showactive":False,"x":0.0,"y":1.15,
            "buttons":[
                {"label":"▶ Iniciar Experimento","method":"animate",
                 "args":[None, {"frame":{"duration":speed_ms,"redraw":True},"fromcurrent":True,"mode":"immediate"}]},
                {"label":"❚❚ Pausar","method":"animate",
                 "args":[[None], {"frame":{"duration":0,"redraw":False},"mode":"immediate"}]}
            ]
        }]
    )
    fig.update_xaxes(range=[-6,2], visible=False, row=1, col=1)
    fig.update_yaxes(range=[-2.5,2.5], visible=False, row=1, col=1)
    fig.update_xaxes(range=[-1.2,1.2], visible=False, row=1, col=2)
    fig.update_yaxes(range=[-0.2,1.2], visible=False, row=1, col=2)
    fig.add_annotation(x=-2, y=0, text='<b>N</b>', showarrow=False, font=dict(color='white', size=20), row=1, col=1)
    fig.add_annotation(x=-4, y=0, text='<b>S</b>', showarrow=False, font=dict(color='white', size=20), row=1, col=1)
    return fig

# ==========================
# 3. NFC (estático)
# ==========================
@st.cache_data
def build_nfc():
    fig = go.Figure()
    fig.add_shape(type='rect', x0=0, y0=0, x1=2, y1=4, line=dict(color='#1e293b', width=3), fillcolor='#f1f5f9')
    fig.add_shape(type='circle', x0=0.8, y0=0.2, x1=1.2, y1=0.6, line=dict(color='#94a3b8', width=2))
    fig.add_annotation(x=1, y=2, text='<b>Smartphone<br>(Ativo)</b><br>Gera Campo<br>Magnético AC', showarrow=False, font=dict(size=14, color='#0f172a'))
    
    theta = np.linspace(0, 10*np.pi, 200); r = np.linspace(0.5, 0.9, 200)
    fig.add_trace(go.Scatter(x=1+r*np.cos(theta), y=2+r*np.sin(theta), mode='lines', line=dict(color='#ef4444', width=2), hoverinfo='skip'))
    
    fig.add_shape(type='rect', x0=7, y0=0.5, x1=10, y1=3.5, line=dict(color='#1e293b', width=3), fillcolor='#f1f5f9')
    fig.add_annotation(x=8.5, y=2, text='<b>Cartão NFC<br>(Passivo)</b><br>Sofre Indução<br>e Responde', showarrow=False, font=dict(size=14, color='#0f172a'))
    
    for i in (1,2):
        fig.add_shape(type='rect', x0=7+i*0.2, y0=0.7+i*0.1, x1=10-i*0.2, y1=3.3-i*0.1, line=dict(color='#3b82f6', width=2, dash='dot'))
    fig.add_shape(type='rect', x0=8.3, y0=2.6, x1=8.7, y1=3.0, line=dict(color='black', width=2), fillcolor='#334155')
    fig.add_annotation(x=8.5, y=3.2, text='Microchip', showarrow=False, font=dict(size=10, color='#0f172a'))
    
    for rad in [2,3,4,5]:
        arc = np.linspace(-np.pi/4, np.pi/4, 50)
        fig.add_trace(go.Scatter(x=2+rad*np.cos(arc), y=2+rad*np.sin(arc), mode='lines', line=dict(color='#10b981', width=2, dash='dash'), hoverinfo='skip'))
        
    fig.update_layout(height=400, showlegend=False, plot_bgcolor='white', paper_bgcolor='white',
                      xaxis=dict(range=[-1,11], visible=False), yaxis=dict(range=[-0.5,4.5], visible=False), margin=dict(l=0,r=0,t=10,b=10))
    return fig

# ==========================
# 4. ANTENA (animação)
# ==========================
@st.cache_data
def build_antena_figure(speed_ms):
    n_frames = 40
    nx = 80 
    t_vals = np.linspace(0, 4*np.pi, n_frames)
    x_onda = np.linspace(0, 10, nx)

    fig = make_subplots(rows=1, cols=2, column_widths=[0.25, 0.75], horizontal_spacing=0.05,
                        specs=[[{"type":"xy"}, {"type":"scene"}]],
                        subplot_titles=("Micro: Elétron na Antena", "Macro: Onda Eletromagnética (3D)"))

    fig.add_trace(go.Scatter(x=[0,0], y=[-2,2], mode='lines', line=dict(color='#94a3b8', width=8), hoverinfo='skip'), row=1, col=1) 
    fig.add_trace(go.Scatter3d(x=[0,10], y=[0,0], z=[0,0], mode='lines', line=dict(color='black', width=2), hoverinfo='skip'), row=1, col=2)
    fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='lines', line=dict(color='#3b82f6', width=4), name='E'), row=1, col=2)
    fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode='lines', line=dict(color='#ef4444', width=4), name='B'), row=1, col=2)

    frames = []
    for t in t_vals:
        y_e = 1.5*math.sin(t)
        Ez = np.sin(x_onda - t)
        By = np.sin(x_onda - t)
        
        frames.append(go.Frame(
            data=[
                go.Scatter(x=[0], y=[y_e], mode='markers', marker=dict(color='red', size=15)),
                go.Scatter3d(x=x_onda, y=np.zeros_like(x_onda), z=Ez, mode='lines', line=dict(color='#3b82f6', width=4)),
                go.Scatter3d(x=x_onda, y=By, z=np.zeros_like(x_onda), mode='lines', line=dict(color='#ef4444', width=4)),
            ],
            traces=[0, 1, 2, 3]
        ))

    fig.frames = frames
    fig.update_layout(
        height=500, showlegend=False, paper_bgcolor='white', plot_bgcolor='white',
        margin=dict(l=10,r=10,t=50,b=10),
        scene=dict(xaxis=dict(range=[0,10], showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(range=[-1.5,1.5], showgrid=False, zeroline=False, showticklabels=False),
                    zaxis=dict(range=[-1.5,1.5], showgrid=False, zeroline=False, showticklabels=False),
                    camera=dict(eye=dict(x=1.5,y=-1.5,z=0.5))),
        updatemenus=[{
            "type":"buttons","showactive":False,"x":0.0,"y":1.15,
            "buttons":[
                {"label":"▶ Transmitir Sinal","method":"animate",
                 "args":[None, {"frame":{"duration":speed_ms,"redraw":True},"fromcurrent":True}]},
                {"label":"❚❚ Pausar","method":"animate",
                 "args":[[None], {"frame":{"duration":0,"redraw":False}}]}
            ]
        }]
    )
    fig.update_xaxes(range=[-1,1], visible=False, row=1, col=1)
    fig.update_yaxes(range=[-2.5,2.5], visible=False, row=1, col=1)
    return fig

# ==========================
# RESUMO
# ==========================
def render_resumo():
    st.markdown('<div class="section-title">📚 Resumo dos Conceitos</div>', unsafe_allow_html=True)
    items = [
        ("Corrente → Campo", "Oersted: toda corrente gera campo magnético circular. Regra da mão direita."),
        ("Campo → Corrente", "Faraday: variação do fluxo Φ_B através de uma espira gera FEM: $\\mathcal{E} = -d\\Phi_B/dt$."),
        ("Lei de Lenz", "O sinal '−' indica que a corrente induzida se opõe à causa que a gerou. Conservação de energia."),
        ("NFC", "Campo AC da maquininha induz corrente no cartão passivo. Chip responde por modulação de carga."),
        ("Antena", "Elétrons acelerados geram ondas EM: E ⟂ B ⟂ direção de propagação. $c = f\\lambda$."),
    ]
    for titulo, desc in items:
        with st.expander(f"🔬 {titulo}"):
            st.write(desc)

# ==========================
# APP
# ==========================
_check_plotly()

st.markdown('<div class="main-title">🧲 Eletromagnetismo Visual</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">A Simetria da Natureza: Eletricidade gera Magnetismo e vice-versa</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["1. Fundamentos (Oersted)", "2. Indução & Lenz", "3. NFC (Aplicação)", "4. Antenas & Ondas", "📖 Resumo"])

with tab1:
    st.markdown('<div class="concept-card"><b>O Início de Tudo:</b> Até 1820, Eletricidade e Magnetismo eram separadas. Oersted descobriu que a eletricidade gera magnetismo. Faraday perguntou: <i>"Se eletricidade gera magnetismo, o magnetismo pode gerar eletricidade?"</i></div>', unsafe_allow_html=True)
    col_f1, col_f2 = st.columns([1, 2.2])
    with col_f1:
        st.markdown('<div class="param-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">1. Corrente gera Magnetismo</div>', unsafe_allow_html=True)
        st.markdown("**(Experimento de Oersted)**<br>Passar corrente por um fio cria campo magnético em anéis. Use a <b>Regra da Mão Direita</b>: polegar no sentido da corrente, dedos curlam no sentido de B.", unsafe_allow_html=True)
        corrente_val = st.slider("Corrente Elétrica (I)", -10.0, 10.0, 5.0, 1.0, key="oersted_I")
        st.markdown("</div>", unsafe_allow_html=True)
    with col_f2:
        try:
            st.plotly_chart(build_oersted(corrente_val), use_container_width=True, config={"displayModeBar": False})
        except Exception as e:
            st.error(f"Erro ao renderizar Oersted: {e}")
    st.markdown("---")
    st.markdown('<div class="alert-card"><b>E o inverso?</b> Faraday colocou um ímã parado ao lado de um fio — zero corrente. A chave é a <b>VARIAÇÃO do fluxo</b>. Vá para a <b>Aba 2</b> para ver isso em ação!</div>', unsafe_allow_html=True)

with tab2:
    st.markdown("""
    <div class="concept-card"><b>Lei da Indução de Faraday:</b> A variação do fluxo magnético através de uma espira gera uma FEM induzida.<br>
    <b>A Regra de Lenz (o sinal de menos):</b> A natureza "odeia" mudanças. A corrente induzida sempre surge num sentido que cria um campo magnético que se <b>opõe</b> à mudança que a originou.</div>""", unsafe_allow_html=True)
    st.latex(r"\mathcal{E} = - \frac{\Delta \Phi_B}{\Delta t}")
    col1, col2 = st.columns([1, 2.5])
    with col1:
        st.markdown('<div class="param-box">', unsafe_allow_html=True)
        st.markdown("<b>Controle do Experimento</b>")
        st.markdown("Quando o ímã <span class='highlight'>aproxima</span> (v > 0), o fluxo aumenta e a corrente gira num sentido para <b>opor</b> o aumento. Quando <span class='highlight'>afasta</span>, inverte para tentar 'puxá-lo' de volta — é a Lei de Lenz!", unsafe_allow_html=True)
        speed = st.select_slider("Velocidade do Ímã", [20, 50, 100], 50, key='lenz_speed',
                                 format_func=lambda x: "Rápido" if x==20 else ("Médio" if x==50 else "Lento"))
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        try:
            st.plotly_chart(build_lenz_figure(speed), use_container_width=True, config={"displayModeBar": False})
        except Exception as e:
            st.error(f"Erro ao renderizar Lenz: {e}")

with tab3:
    st.markdown('<div class="concept-card" style="border-left-color:#10b981"><b>NFC (Near Field Communication):</b> Pagamento por aproximação, crachás, bilhetes. Opera a <b>13.56 MHz</b> — pura indução eletromagnética de Faraday.</div>', unsafe_allow_html=True)
    col_n1, col_n2 = st.columns([1.5, 1.2])
    with col_n1:
        try:
            st.plotly_chart(build_nfc(), use_container_width=True, config={"displayModeBar": False})
        except Exception as e:
            st.error(f"Erro ao renderizar NFC: {e}")
    with col_n2:
        st.markdown('<div class="param-box">', unsafe_allow_html=True)
        st.markdown("<b>Como acontece:</b>")
        st.markdown("""
        1. <b>Campo Ativo:</b> Bobina na maquininha gera campo AC.
        2. <b>Indução:</b> Fluxo variável atravessa a antena do cartão (sem bateria!).
        3. <b>Energia:</b> Lei de Faraday induz corrente — liga o chip.
        4. <b>Resposta:</b> Chip altera resistência da antena (Carga Modulada). Celular detecta perturbação e decodifica dados.
        """)
        st.markdown("</div>", unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="alert-card"><b>O Segredo das Telecomunicações (Wi-Fi, 5G, Rádio):</b> Como a informação atravessa paredes invisivelmente? A resposta está nas cargas elétricas aceleradas.</div>', unsafe_allow_html=True)
    col_a1, col_a2 = st.columns([1, 2.5])
    with col_a1:
        st.markdown('<div class="param-box">', unsafe_allow_html=True)
        st.markdown("<b>Visão Micro e Macro</b>")
        st.markdown("""
        • <b>Micro:</b> Oscilador força elétrons a subirem/descerem na haste metálica.
        • <b>Macro:</b> Perturbação se solta como Onda EM a $c \approx 3\\times10^8$ m/s.
        """)
        st.markdown(r"**A Onda Perfeita:** O Campo Elétrico ($E$, azul) e o Campo Magnético ($B$, vermelho) são gerados ortogonalmente entre si, e ambos ortogonais à direção de propagação.")
        freq = st.select_slider("Frequência de Oscilação", [30, 80, 150], 80, key="ant_freq",
                                format_func=lambda x: "Alta" if x==30 else ("Média" if x==80 else "Baixa"))
        st.markdown("</div>", unsafe_allow_html=True)
    with col_a2:
        try:
            st.plotly_chart(build_antena_figure(freq), use_container_width=True, config={"displayModeBar": False})
        except Exception as e:
            st.error(f"Erro ao renderizar Antena: {e}")

with tab5:
    render_resumo()

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#94a3b8; font-size:0.85rem; padding:1rem;">
🧲 <b>Física Visual: Eletromagnetismo Avançado</b> — Ensino interativo via simulação matemática.
</div>""", unsafe_allow_html=True)
