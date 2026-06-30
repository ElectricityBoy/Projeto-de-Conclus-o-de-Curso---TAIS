import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

# =======================================================
# CONFIGURAÇÃO DA PÁGINA
# =======================================================
st.set_page_config(page_title="Relatório de Anomalias de Rede", layout="wide")
st.title("📡 Análise de Anomalias de Equipamentos de Telecomunicação")
st.markdown("Sistema de monitoramento inteligente para redes de distribuição elétrica.")

# =======================================================
# FUNÇÕES DE CARREGAMENTO DE DADOS (COM CACHE)
# =======================================================
@st.cache_data
def load_data(anomalies_file, outages_file):
    # Carrega anomalias
    df_anom = pd.read_csv(anomalies_file)
    df_anom['timestamp'] = pd.to_datetime(df_anom['timestamp'])
    
    # Carrega outages (Novo Formato com duration_min)
    df_out_raw = pd.read_csv(outages_file)
    df_out_raw['duration_min'] = pd.to_numeric(df_out_raw['duration_min'], errors='coerce')
    
    # Agrupa pegando a CONTAGEM de quedas e a SOMA do tempo fora do ar (em minutos)
    df_out = df_out_raw.groupby('host_name').agg(
        outages=('duration_min', 'count'),
        downtime_minutos=('duration_min', 'sum'),
        mttr_minutos=('duration_min', 'mean'),
        max_outage_min=('duration_min', 'max')
    ).reset_index()
    
    # Converte minutos para horas para ficar mais amigável no painel
    df_out['downtime_horas'] = df_out['downtime_minutos'] / 60.0
    
    # Agrupa pegando o score máximo por equipamento
    df_anom_agg = df_anom.groupby('host_name').agg(
        max_score=('score', 'max'),
        anomaly_count=('score', 'count')
    ).reset_index()
    
    # Mescla as bases (Left join para manter quem tem 0 quedas)
    df_merged = pd.merge(df_anom_agg, df_out, on='host_name', how='left')
    
    # Preenche os nulos com 0 para quem não caiu nenhuma vez
    df_merged.fillna({'outages': 0, 'downtime_horas': 0, 'mttr_minutos': 0, 'max_outage_min': 0}, inplace=True)
    
    return df_anom, df_merged

# =======================================================
# BARRA LATERAL E FILTROS GLOBAIS
# =======================================================
st.sidebar.header("Carregar Dados do Zabbix")
anomalies_csv = st.sidebar.file_uploader("Arquivo de Anomalias (CSV)", type="csv")
outages_csv = st.sidebar.file_uploader("Arquivo de Quedas (CSV)", type="csv")

st.sidebar.markdown("---")
st.sidebar.header("Filtros de Análise")
limiar_critico = st.sidebar.slider("Limiar Crítico de Anomalia (Score)", min_value=0.50, max_value=0.90, value=0.80, step=0.01)

tipo_filtro = st.sidebar.radio(
    "Tipo de Instalação:",
    ["Todas as Instalações", "Subestações", "Clientes Livres"]
)

# =======================================================
# LÓGICA PRINCIPAL DO APLICATIVO
# =======================================================
if anomalies_csv and outages_csv:
    # 1. Carrega os dados brutos
    df_anom_raw, df_merged_raw = load_data(anomalies_csv, outages_csv)
    sns.set_theme(style="whitegrid")
    
    # 2. Aplica o Filtro de Equipamento (com proteção na=False para evitar erros em linhas vazias)
    if tipo_filtro == "Subestações":
        df_anom = df_anom_raw[df_anom_raw['host_name'].str.contains('SUB', na=False)]
        df_merged = df_merged_raw[df_merged_raw['host_name'].str.contains('SUB', na=False)]
    elif tipo_filtro == "Clientes Livres":
        df_anom = df_anom_raw[df_anom_raw['host_name'].str.contains('SMF|CLV', na=False)]
        df_merged = df_merged_raw[df_merged_raw['host_name'].str.contains('SMF|CLV', na=False)]
    else:
        df_anom = df_anom_raw
        df_merged = df_merged_raw

    # 3. Verifica se o filtro retornou dados
    if df_merged.empty:
        st.warning(f"Nenhum dado encontrado para o filtro: {tipo_filtro}")
    else:
        # Cria as 4 abas interativas
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Visão Executiva", 
            "🎯 Diagnóstico e Priorização", 
            "📈 Distribuições Analíticas", 
            "🔍 Investigação de Máquina"
        ])
        
        # ------------------------------------------
        # ABA 1: VISÃO EXECUTIVA CLB_SUB_BMB_VST_01
        # ------------------------------------------
        with tab1:
            st.header(f"Panorama Geral da Rede - {tipo_filtro}")

            # 1. KPIs Estratégicos (Agora com métricas de Tempo)
            total_equipamentos = len(df_merged)
            equipamentos_criticos = len(df_merged[df_merged['max_score'] >= limiar_critico])
            taxa_criticidade = (equipamentos_criticos / total_equipamentos) * 100 if total_equipamentos > 0 else 0
            
            # Descobre quem ficou mais tempo fora do ar (Downtime) e quem teve maior Score
            pior_host_downtime = df_merged.loc[df_merged['downtime_horas'].idxmax()]
            pior_host_score = df_merged.loc[df_merged['max_score'].idxmax()]
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total de Equipamentos", total_equipamentos)
            col2.metric("Equipamentos Críticos", f"{equipamentos_criticos}", f"{taxa_criticidade:.1f}% dos equipamentos", delta_color="inverse")
            
            # NOVA MÉTRICA DE TEMPO:
            col3.metric("Maior Indisponibilidade (Horas)", f"{pior_host_downtime['host_name']}", f"{pior_host_downtime['downtime_horas']:.1f} horas offline", delta_color="inverse")
            
            col4.metric("Maior Risco (Score)", f"{pior_host_score['host_name']}", f"Score: {pior_host_score['max_score']:.3f}", delta_color="inverse")
            
            st.markdown("---")
            
            # 2. Nova Linha de Gráficos (Saúde e Linha do Tempo)
            colC, colD = st.columns(2)
            
            with colC:
                st.subheader("Status de Saúde da Rede")
                st.markdown("Proporção de equipamentos em risco baseada no limiar escolhido.")
                
                # Classificação de Saúde
                saudavel = len(df_merged[df_merged['max_score'] < 0.80])
                atencao = len(df_merged[(df_merged['max_score'] >= 0.80) & (df_merged['max_score'] < limiar_critico)])
                critico = equipamentos_criticos
                
                # Prepara os dados para o Plotly
                labels = ['Saudável (< 0.80)', f'Atenção (0.80 - {limiar_critico})', f'Crítico (>= {limiar_critico})']
                sizes = [saudavel, atencao, critico]
                
                df_donut = pd.DataFrame({'Status': labels, 'Quantidade': sizes})
                
                if sum(sizes) > 0:
                    # Cria o Gráfico de Rosca Interativo com Plotly Express
                    fig_donut = px.pie(
                        df_donut, 
                        names='Status', 
                        values='Quantidade', 
                        hole=0.45, # Define a espessura da rosca
                        color='Status',
                        color_discrete_map={
                            'Saudável (< 0.80)': '#00a443', # Verde
                            f'Atenção (0.80 - {limiar_critico})': '#ff7f0e', # Laranja
                            f'Crítico (>= {limiar_critico})': '#d62728' # Vermelho
                        }
                    )
                    
                    # Formatações visuais (hover, posição do texto)
                    fig_donut.update_traces(
                        textposition='inside', 
                        textinfo='percent', 
                        hovertemplate='<b>%{label}</b><br>Equipamentos: %{value}<br>Proporção: %{percent}<extra></extra>'
                    )
                    fig_donut.update_layout(
                        margin=dict(t=20, b=20, l=0, r=0), 
                        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                    )
                    
                    # Exibe no Streamlit usando a renderização própria do Plotly
                    st.plotly_chart(fig_donut, use_container_width=True)
                else:
                    st.info("Sem dados suficientes.")
            
            with colD:
                st.subheader("Volume de Anomalias Diárias")
                st.markdown("Evolução temporal para identificar picos de instabilidade nos parâmetros de comunicação.")
                
                # Extrai a data para fazer o agrupamento diário
                df_anom['data'] = df_anom['timestamp'].dt.date
                anom_diarias = df_anom.groupby('data').size().reset_index(name='contagem')
                
                if not anom_diarias.empty:
                    # Cria o Gráfico de Linha Interativo com Plotly
                    fig_linha = px.line(
                        anom_diarias, 
                        x='data', 
                        y='contagem', 
                        markers=True # Adiciona as bolinhas em cada dia
                    )
                    
                    # Customizações visuais (cor vermelha escura, preenchimento e tooltips)
                    fig_linha.update_traces(
                        line=dict(color='darkred', width=2.5),
                        marker=dict(size=7, color='darkred'),
                        fill='tozeroy', # Preenche o espaço abaixo da linha
                        fillcolor='rgba(139, 0, 0, 0.15)', # Vermelho transparente
                        hovertemplate='<b>%{x}</b><br>Total de Anomalias: %{y}<extra></extra>'
                    )
                    
                    # Organização dos eixos e margens
                    fig_linha.update_layout(
                        xaxis_title="Data do Evento",
                        yaxis_title="Quantidade de Alertas",
                        margin=dict(t=20, b=20, l=0, r=0),
                        hovermode="x unified" # Cria uma linha guia interativa ao passar o mouse
                    )
                    
                    # Renderiza o gráfico
                    st.plotly_chart(fig_linha, use_container_width=True)
                else:
                    st.info("Sem dados temporais suficientes.")

            st.markdown("---")
            
            # 3. Segunda Linha de Gráficos (Sazonalidade e Quedas - Agora com Plotly)
            colA, colB = st.columns(2)
            
            with colA:
                st.subheader("Mapa de Calor: Sazonalidade das Falhas")
                st.markdown("Horários mais críticos de degradação da rede.")
                df_anom_heat = df_anom.copy()
                df_anom_heat['hour'] = df_anom_heat['timestamp'].dt.hour
                
                # Obtém o nome do dia em inglês e traduz para português
                df_anom_heat['day_of_week'] = df_anom_heat['timestamp'].dt.day_name()
                dias_traducao = {
                    'Monday': 'Segunda', 'Tuesday': 'Terça', 'Wednesday': 'Quarta',
                    'Thursday': 'Quinta', 'Friday': 'Sexta', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
                }
                df_anom_heat['day_of_week'] = df_anom_heat['day_of_week'].map(dias_traducao)
                
                # Ordem correta dos dias
                days_order = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
                pivot = df_anom_heat.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0).reindex(days_order)
                
                # Garante que todas as 24 horas existam no eixo X para o gráfico não ficar cortado
                for h in range(24):
                    if h not in pivot.columns:
                        pivot[h] = 0
                pivot = pivot[range(24)]
                
                # Cria o Mapa de Calor interativo com Plotly
                fig_heat = px.imshow(
                    pivot,
                    labels=dict(x="Hora do Dia (0-23h)", y="Dia da Semana", color="Alertas"),
                    x=[f"{h}h" for h in pivot.columns], # Adiciona o 'h' no eixo X
                    y=pivot.index,
                    color_continuous_scale="YlOrRd",
                    aspect="auto" # Faz o gráfico se ajustar à largura da tela
                )
                
                fig_heat.update_traces(hovertemplate='<b>%{y} às %{x}</b><br>Total de Anomalias: %{z}<extra></extra>')
                fig_heat.update_layout(margin=dict(t=20, b=20, l=0, r=0))
                
                st.plotly_chart(fig_heat, use_container_width=True)
                
            with colB:
                st.subheader("Top 10 Instalações com Mais Quedas")
                st.markdown("Equipamentos que mais desconectaram durante o período.")
                
                # Pega os 10 piores
                top_outages = df_merged.sort_values(by='outages', ascending=False).head(10)
                
                # Cria o Gráfico de Barras interativo com Plotly
                fig_bar = px.bar(
                    top_outages, 
                    x='outages', 
                    y='host_name', 
                    orientation='h',
                    color='outages',
                    color_continuous_scale='Reds', # Usa gradiente de vermelho
                    labels={'outages': 'Número Total de Quedas', 'host_name': 'Equipamento'}
                )
                
                # Configurações de layout (inverte o eixo Y para o pior ficar no topo)
                fig_bar.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    margin=dict(t=20, b=20, l=0, r=0),
                    coloraxis_showscale=False # Esconde a barra lateral de cor para ficar mais limpo
                )
                
                # Coloca o número de quedas na ponta da barra
                fig_bar.update_traces(
                    texttemplate='<b>%{x}</b>', 
                    textposition='outside',
                    hovertemplate='<b>%{y}</b><br>Total de Quedas: %{x}<extra></extra>'
                )
                
                st.plotly_chart(fig_bar, use_container_width=True)

       # ------------------------------------------
        # ABA 2: DIAGNÓSTICO E PRIORIZAÇÃO
        # ------------------------------------------
        with tab2:
            st.header("Matriz de Criticidade")
            st.markdown("Equipamentos no quadrante superior direito (acima da linha tracejada) requerem intervenção imediata.")
            
            # Gráfico de Bolhas Interativo com Plotly
            fig_scatter = px.scatter(
                df_merged,
                x='outages',
                y='max_score',
                size='anomaly_count',
                hover_name='host_name', # Nome do host no título da caixinha
                color='max_score', # Gradiente baseado na gravidade
                color_continuous_scale='Reds',
                labels={
                    'outages': 'Total de Quedas', 
                    'max_score': 'Score de Anomalia Máximo', 
                    'anomaly_count': 'Contagem de Anomalias'
                },
                size_max=40 # Tamanho máximo da bolha
            )
            
            # Adiciona a linha tracejada do limiar
            fig_scatter.add_hline(
                y=limiar_critico, 
                line_dash="dash", 
                line_color="red", 
                annotation_text=f"Limiar Crítico ({limiar_critico})", 
                annotation_position="top left"
            )
            
            fig_scatter.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')), opacity=0.8)
            fig_scatter.update_layout(margin=dict(t=20, b=20, l=0, r=0))
            
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            st.subheader("Tabela de Priorização de Chamados Técnicos")
            
            df_critico = df_merged[df_merged['max_score'] >= limiar_critico].sort_values(by='max_score', ascending=False)
            
            # Copia as colunas, incluindo o max_outage_min que já tínhamos no backend
            df_tabela = df_critico[['host_name', 'max_score', 'outages', 'downtime_horas', 'mttr_minutos', 'max_outage_min']].copy()
            
            # Calcula a Disponibilidade Mensal (SLA) assumindo um mês de 30 dias (720 horas)
            df_tabela['sla_percentual'] = ((720 - df_tabela['downtime_horas']) / 720) * 100
            # Garante que não passe de 100% ou fique negativo
            df_tabela['sla_percentual'] = df_tabela['sla_percentual'].clip(lower=0, upper=100)
            
            # Converte o pior apagão de minutos para horas
            df_tabela['pior_apagao_horas'] = df_tabela['max_outage_min'] / 60.0
            
            # Organiza as colunas finais
            df_tabela = df_tabela[['host_name', 'max_score', 'sla_percentual', 'outages', 'downtime_horas', 'pior_apagao_horas']]
            df_tabela.columns = ['Equipamento', 'Score de Anomalia', 'Disponibilidade (SLA %)', 'Quedas', 'Horas Offline', 'Maior Apagão (Horas)']
            
            st.dataframe(
                df_tabela, use_container_width=True, hide_index=True, 
                column_config={
                    "Equipamento": st.column_config.TextColumn("Equipamento (Host)"),
                    "Score de Anomalia": st.column_config.ProgressColumn("Score", format="%.3f", min_value=0.0, max_value=1.0),
                    "Disponibilidade (SLA %)": st.column_config.NumberColumn(
                        "Disponibilidade (SLA)", 
                        help="Meta ideal é acima de 99.5%. Valores menores indicam quebra de contrato operacional.",
                        format="%.2f %%" # Formata com o símbolo de porcentagem
                    ),
                    "Quedas": st.column_config.NumberColumn("Quedas Totais"),
                    "Horas Offline": st.column_config.NumberColumn("Tempo Offline (h)", format="%.1f"),
                    "Maior Apagão (Horas)": st.column_config.NumberColumn(
                        "Pior Queda Contínua", 
                        help="Duração em horas da maior queda individual que o equipamento sofreu.",
                        format="%.1f h"
                    )                }
            )

        # ------------------------------------------
        # ABA 3: DISTRIBUIÇÕES ANALÍTICAS
        # ------------------------------------------
        with tab3:
            st.header("Análise de Distribuição Estatística")
            
            col_dist1, col_dist2 = st.columns(2)
            
            with col_dist1:
                st.subheader("Distribuição de Scores de Anomalia")
                st.markdown("Frequência de equipamentos por anomalia identificada.")
                
                # Histograma interativo refinado (sem o marginal para não amassar o gráfico)
                fig_hist = px.histogram(
                    df_anom,
                    x='score',
                    nbins=40, # Reduzido levemente para agrupar melhor os dados
                    color_discrete_sequence=['#8B0000'], # Vermelho escuro
                    opacity=0.85,
                    labels={'score': 'Score de Anomalia'}
                )
                
                # Adiciona o contorno preto nas barras e a linha de corte
                fig_hist.update_traces(marker_line_width=1.2, marker_line_color="black")
                
                fig_hist.add_vline(
                    x=limiar_critico, 
                    line_dash="dash", 
                    line_color="yellow", 
                    annotation_text=f" Limiar ({limiar_critico})",
                    annotation_position="top right"
                )
                
                # Adiciona o gap (espaço) entre as barras para facilitar a leitura
                fig_hist.update_layout(
                    yaxis_title="Frequência de Registros", 
                    bargap=0.05, 
                    margin=dict(t=20, b=20, l=0, r=0),
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig_hist, use_container_width=True)
                
            with col_dist2:
                st.subheader("Dispersão de Latência (Top 10)")
                st.markdown("Caixas deslocadas à direita indicam lentidão extrema e prolongada.")
                
                top_10_hosts = list(df_merged.sort_values(by='max_score', ascending=False).head(10)['host_name'])
                
                if len(top_10_hosts) > 0:
                    df_top10 = df_anom[df_anom['host_name'].isin(top_10_hosts)]
                    
                    # Boxplot interativo
                    fig_box = px.box(
                        df_top10,
                        x='Response time',
                        y='host_name',
                        color='host_name',
                        category_orders={"host_name": top_10_hosts}, # Garante a ordem do pior para o melhor
                        labels={'Response time': 'Tempo de Resposta - Latência (ms)', 'host_name': 'Equipamento'}
                    )
                    
                    fig_box.update_layout(showlegend=False, margin=dict(t=20, b=20, l=0, r=0))
                    st.plotly_chart(fig_box, use_container_width=True)
                else:
                    st.info("Não há equipamentos suficientes para gerar o Boxplot neste filtro.")

        # ------------------------------------------
        # ABA 4: INVESTIGAÇÃO DE MÁQUINA
        # ------------------------------------------
        with tab4:
            st.header("Inspeção Individual de Equipamento")
            
            lista_hosts = list(df_merged['host_name'].sort_values())
            host_selecionado = st.selectbox("Selecione a instalação para investigação profunda:", lista_hosts)
            
            if host_selecionado:
                df_host_especifico = df_anom[df_anom['host_name'] == host_selecionado]
                
                if not df_host_especifico.empty:
                    st.markdown(f"**Analisando Assinatura de Tráfego:** `{host_selecionado}`")
                    
                    # Gráfico de Dispersão Individual interativo
                    fig_ind = px.scatter(
                        df_host_especifico,
                        x='Packet loss',
                        y='Response time',
                        color='score',
                        color_continuous_scale='RdBu_r', # Simula o coolwarm (Azul -> Vermelho)
                        hover_data=['timestamp'], # Mostra a data exata ao passar o mouse!
                        labels={
                            'Packet loss': 'Perda de Pacotes (%)', 
                            'Response time': 'Tempo de Resposta - Latência (ms)', 
                            'score': 'Score de Anomalia',
                            'timestamp': 'Data/Hora do Evento'
                        }
                    )
                    
                    fig_ind.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')), opacity=0.9)
                    fig_ind.update_layout(margin=dict(t=20, b=20, l=0, r=0))
                    
                    st.plotly_chart(fig_ind, use_container_width=True)
                else:
                    st.warning("Sem dados de anomalia detalhados para este equipamento.")

else:
    # Mensagem de boas-vindas quando os arquivos ainda não foram carregados
    st.info("👋 Bem-vindo! Por favor, carregue os arquivos `degradation_anomalies.csv` e `outages.csv` na barra lateral esquerda para iniciar a análise.")