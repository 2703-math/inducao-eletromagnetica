import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

# ============================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================
st.set_page_config(
    page_title="Física Visual: Eletromagnetismo",
    page_icon="🧲",
    layout="wide"
)

# ============================================
# CSS PERSONALIZADO (PADRÃO SAAS)
# ============================================
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0f172a;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
    }
    .concept-card {
        background: #f8fafc;
        border-radius: 12px;
        padding: 1.2rem;
        border-left: 4px solid #3b82f6;
        margin-bottom: 1rem;
        color: #334155;
    }
    .alert-card {
        background: #fffbeb;
        border-radius: 12px;
        padding: 1.2rem;
        border-left: 4px solid #f59e0b;
        margin-bottom: 1rem;
        color: #334155;
    }
    .param-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# FUNÇÃO 1: ANIMAÇÃO DE FARADAY-LENZ
# ============================================
def gerar_animacao_lenz(duracao_ms):
    fig = make_subplots(rows=1, cols=2, column_widths=[0.7, 0.3], horizontal_spacing=0.05, 
                        subplot_titles=("Ímã e Espira (Indução)", "Galvanômetro (Corrente Induzida)"))

    # Parâmetros de tempo para um ciclo completo de ida e volta do ímã
    t_vals = np.linspace(0, 2 * np.pi, 60)
    
    # Posições estáticas da espira (bobina)
    x_espira = [0, 1, 1, 0, 0]
    y_espira = [-1.5, -1.5, 1.5, 1.5, -1.5]
    
    fig.add_trace(go.Scatter(x=x_espira, y=y_espira, mode='lines', line=dict(color='#94a3b8', width=5), name='Espira', hoverinfo='skip'), row=1, col=1)
    
    # Galvanômetro (Fundo)
    theta_arc = np.linspace(0, np.pi, 50)
    fig.add_trace(go.Scatter(x=np.cos(theta_arc), y=np.sin(theta_arc), mode='lines', line=dict(color='#cbd5e1', width=3), hoverinfo='skip'), row=1, col=2)
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers', marker=dict(color='black', size=10), hoverinfo='skip'), row=1, col=2)

    frames = []
    for t in t_vals:
        # Movimento do Ímã: Oscila entre -5 e -1.5
        x_ima_centro = -3.5 + 2 * math.cos(t)
        v_ima = -2 * math.sin(t) # Velocidade (derivada da posição)
        
        # O fluxo magnético diminui com a distância quadrada, mas simplificaremos para o visual:
        # A corrente induzida (I) é proporcional a -Velocidade (Lei de Lenz)
        corrente_induzida = -v_ima 
        
        # Desenho do Ímã
        x_ima = [x_ima_centro-1.5, x_ima_centro, x_ima_centro, x_ima_centro-1.5, x_ima_centro-1.5]
        y_ima = [-0.8, -0.8, 0.8, 0.8, -0.8]
        
        x_ima_sul = [x_ima_centro, x_ima_centro+1.5, x_ima_centro+1.5, x_ima_centro, x_ima_centro]
        
        # Vetor de corrente na espira
        if corrente_induzida > 0.1:
            seta_i_y = [1.8, 1.8]
            seta_i_x = [0.2, 0.8]
            cor_seta = "#ef4444"
            texto_i = "Corrente (Sentido Horário)"
        elif corrente_induzida < -0.1:
            seta_i_y = [1.8, 1.8]
            seta_i_x = [0.8, 0.2]
            cor_seta = "#3b82f6"
            texto_i = "Corrente (Sentido Anti-horário)"
        else:
            seta_i_x = [0.5, 0.5]
            seta_i_y = [1.8, 1.8]
            cor_seta = "rgba(0,0,0,0)"
            texto_i = "Sem Corrente (v=0)"

        # Ponteiro do Galvanômetro
        ang_ponteiro = (np.pi/2) - (corrente_induzida * 0.4) # 90 graus (meio) +/- desvio
        px = 0.9 * math.cos(ang_ponteiro)
        py = 0.9 * math.sin(ang_ponteiro)

        frames.append(go.Frame(
            data=[
                # Ímã (Norte - Vermelho)
                go.Scatter(x=x_ima_sul, y=y_ima, fill='toself', fillcolor='#ef4444', line=dict(color='black')),
                # Ímã (Sul - Azul)
                go.Scatter(x=x_ima, y=y_ima, fill='toself', fillcolor='#3b82f6', line=dict(color='black')),
                # Indicador de Corrente
                go.Scatter(x=seta_i_x, y=seta_i_y, mode='lines+markers', marker=dict(symbol='arrow-right', size=15), line=dict(color=cor_seta, width=4), text=[texto_i], hoverinfo='text'),
                # Ponteiro do Galvanômetro
                go.Scatter(x=[0, px], y=[0, py], mode='lines', line=dict(color=cor_seta if cor_seta != "rgba(0,0,0,0)" else "black", width=4))
            ],
            traces=[2, 3, 4, 5],
            name=f"f_{t}"
        ))

    # Traces iniciais invisiveis/vazios para serem substituidos pelos frames
    fig.add_trace(go.Scatter(x=[], y=[]), row=1, col=1) # trace 2
    fig.add_trace(go.Scatter(x=[], y=[]), row=1, col=1) # trace 3
    fig.add_trace(go.Scatter(x=[], y=[]), row=1, col=1) # trace 4
    fig.add_trace(go.Scatter(x=[], y=[]), row=1, col=2) # trace 5

    fig.frames = frames

    fig.update_layout(
        height=400, showlegend=False, plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=10, r=10, t=40, b=10),
        updatemenus=[{
            "type": "buttons", "showactive": False, "x": 0.0, "y": 1.15,
            "buttons": [
                {"label": "▶ Iniciar Experimento", "method": "animate", "args": [None, {"frame": {"duration": duracao_ms, "redraw": True}, "fromcurrent": True, "mode": "immediate"}]},
                {"label": "❚❚ Pausar", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]}
            ]
        }]
    )
    fig.update_xaxes(range=[-6, 2], showgrid=False, zeroline=False, visible=False, row=1, col=1)
    fig.update_yaxes(range=[-2.5, 2.5], showgrid=False, zeroline=False, visible=False, row=1, col=1)
    fig.update_xaxes(range=[-1.2, 1.2], showgrid=False, zeroline=False, visible=False, row=1, col=2)
    fig.update_yaxes(range=[-0.2, 1.2], showgrid=False, zeroline=False, visible=False, row=1, col=2)
    
    # Anotações Fixas
    fig.add_annotation(x=-2, y=0, text="<b>N</b>", showarrow=False, font=dict(color="white", size=20), row=1, col=1)
    fig.add_annotation(x=-4, y=0, text="<b>S</b>", showarrow=False, font=dict(color="white", size=20), row=1, col=1)
    
    return fig

# ============================================
# FUNÇÃO 2: DIAGRAMA NFC (ESTÁTICO INTERATIVO)
# ============================================
def gerar_diagrama_nfc():
    fig = go.Figure()

    # Desenho do Smartphone (Dispositivo Ativo)
    fig.add_shape(type="rect", x0=0, y0=0, x1=2, y2=4, line=dict(color="#1e293b", width=3), fillcolor="#f1f5f9", rx=0.2, ry=0.2)
    fig.add_shape(type="circle", x0=0.8, y0=0.2, x1=1.2, y2=0.6, line=dict(color="#94a3b8", width=2))
    fig.add_annotation(x=1, y=2, text="<b>Smartphone<br>(Ativo)</b><br>Gera Campo<br>Magnético AC", showarrow=False, font=dict(size=14, color="#0f172a"))
    
    # Bobina Interna do Smartphone
    theta = np.linspace(0, 10*np.pi, 200)
    r = np.linspace(0.5, 0.9, 200)
    fig.add_trace(go.Scatter(x=1 + r*np.cos(theta), y=2 + r*np.sin(theta), mode='lines', line=dict(color="#ef4444", width=2), hoverinfo='skip'))

    # Desenho do Cartão / Tag NFC (Dispositivo Passivo)
    fig.add_shape(type="rect", x0=7, y0=0.5, x1=10, y2=3.5, line=dict(color="#1e293b", width=3), fillcolor="#f1f5f9", rx=0.2, ry=0.2)
    fig.add_annotation(x=8.5, y=2, text="<b>Cartão NFC<br>(Passivo)</b><br>Sofre Indução<br>e Responde", showarrow=False, font=dict(size=14, color="#0f172a"))
    
    # Bobina e Chip do Cartão
    fig.add_shape(type="rect", x0=7.2, y0=0.7, x1=9.8, y2=3.3, line=dict(color="#3b82f6", width=2, dash="dot"))
    fig.add_shape(type="rect", x0=7.3, y0=0.8, x1=9.7, y2=3.2, line=dict(color="#3b82f6", width=2, dash="dot"))
    fig.add_shape(type="rect", x0=8.3, y0=2.6, x1=8.7, y2=3.0, line=dict(color="black", width=2), fillcolor="#334155") # Microchip
    fig.add_annotation(x=8.5, y=3.2, text="Microchip", showarrow=False, font=dict(size=10, color="#0f172a"))

    # Linhas de Campo Magnético (Acoplamento Indutivo)
    for rad in [2, 3, 4, 5]:
        arc_th = np.linspace(-np.pi/4, np.pi/4, 50)
        fig.add_trace(go.Scatter(x=2 + rad*np.cos(arc_th), y=2 + rad*np.sin(arc_th), mode='lines', line=dict(color="#10b981", width=2, dash='dash'), hoverinfo='skip'))

    fig.update_layout(
        height=400, showlegend=False, plot_bgcolor='white', paper_bgcolor='white',
        xaxis=dict(range=[-1, 11], visible=False),
        yaxis=dict(range=[-0.5, 4.5], visible=False),
        margin=dict(l=0, r=0, t=10, b=10)
    )
    return fig

# ============================================
# FUNÇÃO 3: ANIMAÇÃO DA ANTENA (MICRO E MACRO - 3D)
# ============================================
def gerar_animacao_antena(duracao_ms):
    fig = make_subplots(
        rows=1, cols=2, 
        column_widths=[0.25, 0.75], 
        horizontal_spacing=0.05,
        specs=[[{"type": "xy"}, {"type": "scene"}]],
        subplot_titles=("Micro: Elétron na Antena", "Macro: Onda Eletromagnética (3D)")
    )

    t_vals = np.linspace(0, 4*np.pi, 40)
    x_onda = np.linspace(0, 10, 100)

    # Subplot 1: Haste da Antena
    fig.add_trace(go.Scatter(x=[0, 0], y=[-2, 2], mode='lines', line=dict(color='#94a3b8', width=8), hoverinfo='skip'), row=1, col=1)
    
    # Subplot 2: Eixo de propagação (Vazio inicial)
    fig.add_trace(go.Scatter3d(x=[0, 10], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='black', width=2), hoverinfo='skip'), row=1, col=2)

    frames = []
    for t in t_vals:
        # Posição do elétron oscilante
        y_eletron = 1.5 * math.sin(t)
        
        # Campos Eletromagnéticos se propagando
        # Campo Elétrico (E) oscila no eixo Z
        E_z = np.sin(x_onda - t)
        E_y = np.zeros_like(x_onda)
        
        # Campo Magnético (B) oscila ortogonalmente no eixo Y
        B_y = np.sin(x_onda - t)
        B_z = np.zeros_like(x_onda)

        frames.append(go.Frame(
            data=[
                # Elétron
                go.Scatter(x=[0], y=[y_eletron], mode='markers', marker=dict(color='red', size=15)),
                # Campo Elétrico (Azul)
                go.Scatter3d(x=x_onda, y=E_y, z=E_z, mode='lines', line=dict(color='#3b82f6', width=4)),
                # Campo Magnético (Vermelho)
                go.Scatter3d(x=x_onda, y=B_y, z=B_z, mode='lines', line=dict(color='#ef4444', width=4))
            ],
            traces=[1, 2, 3],
            name=f"f_{t}"
        ))

    fig.add_trace(go.Scatter(x=[], y=[]), row=1, col=1) # Trace 1
    fig.add_trace(go.Scatter3d(x=[], y=[], z=[]), row=1, col=2) # Trace 2
    fig.add_trace(go.Scatter3d(x=[], y=[], z=[]), row=1, col=2) # Trace 3

    fig.frames = frames

    fig.update_layout(
        height=500, showlegend=False, paper_bgcolor='white', plot_bgcolor='white',
        margin=dict(l=10, r=10, t=50, b=10),
        scene=dict(
            xaxis_title='Propagação (X)', yaxis_title='Campo B (Y)', zaxis_title='Campo E (Z)',
            xaxis=dict(range=[0, 10], showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(range=[-1.5, 1.5], showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(range=[-1.5, 1.5], showgrid=False, zeroline=False, showticklabels=False),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.5)) # Visão isométrica fixa
        ),
        updatemenus=[{
            "type": "buttons", "showactive": False, "x": 0.0, "y": 1.15,
            "buttons": [
                {"label": "▶ Transmitir Sinal", "method": "animate", "args": [None, {"frame": {"duration": duracao_ms, "redraw": True}, "fromcurrent": True}]},
                {"label": "❚❚ Pausar", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}}]}
            ]
        }]
    )
    fig.update_xaxes(range=[-1, 1], visible=False, row=1, col=1)
    fig.update_yaxes(range=[-2.5, 2.5], visible=False, row=1, col=1)
    return fig


# ============================================
# TÍTULO E ABAS DO APP
# ============================================
st.markdown('<div class="main-title">🧲 Eletromagnetismo Visual</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Da Lei de Faraday-Lenz à tecnologia NFC e Ondas de Rádio</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "1. Indução e Lei de Lenz", 
    "2. Tecnologia NFC (Aplicações)", 
    "3. Antenas e Ondas Eletromagnéticas"
])

# ============================================
# ABA 1: LEI DE FARADAY E LENZ
# ============================================
with tab1:
    st.markdown("""
    <div class="concept-card">
        <b>Lei da Indução de Faraday:</b> A variação do fluxo magnético através de uma espira gera uma força eletromotriz (tensão) induzida.<br>
        <b>A Regra de Lenz (O sinal de menos):</b> A natureza "odeia" mudanças. A corrente induzida sempre surge num sentido que cria um campo magnético que se <b>opõe</b> à mudança que a originou.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(r"$$ \mathcal{E} = - \frac{\Delta \Phi_B}{\Delta t} $$")
    
    col1, col2 = st.columns([1, 2.5])
    with col1:
        st.markdown("<div class='param-box'>", unsafe_allow_html=True)
        st.markdown("**Controle do Experimento:**")
        st.markdown("Observe que, quando o ímã se **aproxima** (v > 0), o fluxo aumenta e a corrente gira num sentido. Quando ele se **afasta**, o fluxo diminui e a corrente inverte para tentar 'puxar' o ímã de volta!")
        duracao_ms_1 = st.select_slider("Velocidade do Ímã", options=[20, 50, 100], value=50, key='dur_1', format_func=lambda x: "Rápido" if x==20 else ("Médio" if x==50 else "Lento"))
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        fig1 = gerar_animacao_lenz(duracao_ms_1)
        st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False})

# ============================================
# ABA 2: TECNOLOGIA NFC
# ============================================
with tab2:
    st.markdown("""
    <div class="concept-card" style="border-left-color: #10b981;">
        <b>NFC (Near Field Communication):</b> É a tecnologia por trás do pagamento por aproximação (Apple Pay, cartões de crédito), crachás de acesso e bilhetes de transporte. Ela opera na frequência de <b>13.56 MHz</b> e funciona inteiramente graças à Indução Eletromagnética de Faraday.
    </div>
    """, unsafe_allow_html=True)
    
    col_nfc1, col_nfc2 = st.columns([1.5, 1.2])
    with col_nfc1:
        st.plotly_chart(gerar_diagrama_nfc(), use_container_width=True, config={'displayModeBar': False})
        
    with col_nfc2:
        st.markdown("<div class='param-box'>", unsafe_allow_html=True)
        st.markdown("### Como acontece a mágica? (Passo a Passo)")
        st.markdown("""
        **1. O Campo Ativo:** A maquininha de cartão ou smartphone possui uma bobina energizada por bateria que gera um campo magnético que alterna milhões de vezes por segundo (AC).
        
        **2. Indução no Cartão:** O cartão de crédito **não tem bateria**. Ele possui uma antena de cobre enrolada internamente. Quando aproximado do celular (geralmente < 4 cm), o campo magnético variável atravessa essa antena.
        
        **3. Energia "Do Nada":** A variação do fluxo magnético (Lei de Faraday) induz uma corrente elétrica na antena do cartão, fornecendo energia suficiente para ligar o microchip interno.
        
        **4. A Resposta:** O chip acorda e altera a resistência da sua própria antena. Isso perturba o campo magnético do celular (Carga Modulada). O celular "sente" essa perturbação e decodifica os dados bancários com segurança!
        """)
        st.markdown("</div>", unsafe_allow_html=True)

# ============================================
# ABA 3: ANTENAS E ONDAS ELETROMAGNÉTICAS
# ============================================
with tab3:
    st.markdown("""
    <div class="alert-card">
        <b>O Segredo das Telecomunicações (Wi-Fi, 5G, Rádio):</b> Como a informação atravessa paredes invisivelmente? A resposta está nas cargas elétricas aceleradas.
    </div>
    """, unsafe_allow_html=True)
    
    col_ant1, col_ant2 = st.columns([1, 2.5])
    
    with col_ant1:
        st.markdown("<div class='param-box'>", unsafe_allow_html=True)
        st.markdown("### A Visão Micro e Macro")
        st.markdown("""
        • **Microscópico:** Um circuito oscilador força os elétrons a subirem e descerem rapidamente na haste metálica da antena. Elétrons acelerados perturbam o tecido do espaço.\n
        • **Macroscópico:** Essa perturbação se solta da antena em forma de uma Onda Eletromagnética, propagando-se na velocidade da luz ($c \approx 300.000$ km/s).\n
        """)
        st.markdown(r"**A Onda Perfeita:**")
        st.markdown("O Campo Elétrico ($E$, azul) e o Campo Magnético ($B$, vermelho) são gerados ortogonalmente entre si, e ambos ortogonais à direção de propagação.")
        duracao_ms_3 = st.select_slider("Frequência de Oscilação", options=[30, 80, 150], value=80, key='dur_3', format_func=lambda x: "Alta" if x==30 else ("Média" if x==80 else "Baixa"))
        st.markdown("</div>", unsafe_allow_html=True)

    with col_ant2:
        fig3 = gerar_animacao_antena(duracao_ms_3)
        st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})

# Rodapé
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.85rem; padding: 1rem;">
    🧲 <b>Física Visual: Eletromagnetismo Avançado</b> — Construído para ensino interativo via simulação matemática.
</div>
""", unsafe_allow_html=True)