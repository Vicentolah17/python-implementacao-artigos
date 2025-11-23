import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import plotly.graph_objects as go
import math
from typing import Dict, List

st.set_page_config(
    page_title="SSAP - Shelf Space Allocation Problem",
    page_icon="📦",
    layout="wide"
)

def solve_shelf_space_gcd(weights: List[int], values: List[int], capacity: int) -> Dict:
    """
    Implementação do algoritmo de Programação Dinâmica otimizado por GCD
    conforme proposto por Czerniachowska e Lutosławski (2021).
    
    Args:
        weights (list): Lista de pesos (widths) dos produtos.
        values (list): Lista de valores (lucro) dos produtos.
        capacity (int): Capacidade total da prateleira (shelf space).
        
    Returns:
        dict: Dicionário contendo o valor máximo, itens escolhidos e tamanho da tabela.
    """
    n = len(weights)
    
    # 1. Calcular o GCD (Máximo Divisor Comum) de todos os pesos
    common_divisor = math.gcd(*weights)
    
    # 2. Reduzir a dimensão do problema
    scaled_capacity = capacity // common_divisor
    scaled_weights = [w // common_divisor for w in weights]
    
    # 3. Inicializar a tabela de Programação Dinâmica (V)
    dp = [[0 for _ in range(scaled_capacity + 1)] for _ in range(n + 1)]
    
    # 4. Preencher a tabela (Lógica Iterativa)
    for i in range(1, n + 1):
        w_current = scaled_weights[i-1]
        v_current = values[i-1]
        
        for w in range(1, scaled_capacity + 1):
            if w_current <= w:
                dp[i][w] = max(dp[i-1][w], 
                               v_current + dp[i-1][w - w_current])
            else:
                dp[i][w] = dp[i-1][w]
                
    # 5. Reconstrução da solução (Backtracking para achar os itens)
    selected_items = []
    w_temp = scaled_capacity
    for i in range(n, 0, -1):
        if dp[i][w_temp] != dp[i-1][w_temp]:
            item_index = i - 1
            selected_items.append({
                "index": item_index,
                "weight": weights[item_index],
                "value": values[item_index]
            })
            w_temp -= scaled_weights[item_index]
            
    return {
        "max_value": dp[n][scaled_capacity],
        "selected_items": selected_items,
        "gcd": common_divisor,
        "table_size": (n + 1) * (scaled_capacity + 1),
        "original_table_size": (n + 1) * (capacity + 1)
    }


def draw_shelf_visualization(weights: List[int], values: List[int], capacity: int, 
                             selected_items: List[Dict], gcd: int):
    """
    Desenha a visualização gráfica da prateleira com os produtos alocados.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Cores vibrantes para os produtos
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', 
              '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B400', '#52B788']
    
    # ============ PRATELEIRA COM ITENS SELECIONADOS ============
    ax1.set_xlim(0, capacity + 10)
    ax1.set_ylim(0, 15)
    ax1.set_aspect('equal')
    
    # Desenhar a prateleira (fundo)
    shelf_rect = patches.Rectangle((0, 0), capacity, 10, 
                                   linewidth=3, edgecolor='#2C3E50', 
                                   facecolor='#ECF0F1', alpha=0.3)
    ax1.add_patch(shelf_rect)
    
    # Adicionar suporte da prateleira
    support_left = patches.Rectangle((-2, -2), 2, 12, 
                                     linewidth=2, edgecolor='#34495E', 
                                     facecolor='#7F8C8D')
    support_right = patches.Rectangle((capacity, -2), 2, 12, 
                                      linewidth=2, edgecolor='#34495E', 
                                      facecolor='#7F8C8D')
    ax1.add_patch(support_left)
    ax1.add_patch(support_right)
    
    # Desenhar produtos selecionados
    x_position = 0
    for idx, item in enumerate(reversed(selected_items)):
        item_idx = item['index']
        width = item['weight']
        value = item['value']
        
        # Altura proporcional ao valor
        height = min(8, 3 + (value / max(values)) * 5)
        
        color = colors[item_idx % len(colors)]
        
        # Desenhar o produto
        product = patches.Rectangle((x_position, 1), width, height,
                                    linewidth=2, edgecolor='#2C3E50',
                                    facecolor=color, alpha=0.8)
        ax1.add_patch(product)
        
        # Adicionar rótulo do produto
        ax1.text(x_position + width/2, height/2 + 1, 
                f'P{item_idx}\n${value}',
                ha='center', va='center', fontsize=10, 
                fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))
        
        # Adicionar largura do produto
        ax1.text(x_position + width/2, -0.5, 
                f'{width}',
                ha='center', va='top', fontsize=9, 
                fontweight='bold', color='#2C3E50')
        
        x_position += width
    
    # Mostrar espaço não utilizado
    if x_position < capacity:
        unused_space = capacity - x_position
        unused_rect = patches.Rectangle((x_position, 1), unused_space, 8,
                                       linewidth=2, edgecolor='#E74C3C',
                                       facecolor='white', alpha=0.3,
                                       linestyle='--')
        ax1.add_patch(unused_rect)
        ax1.text(x_position + unused_space/2, 5, 
                f'Espaço\nNão Usado\n({unused_space})',
                ha='center', va='center', fontsize=9, 
                color='#E74C3C', style='italic')
    
    # Configurações do eixo
    ax1.set_xlabel('Largura da Prateleira (unidades)', fontsize=12, fontweight='bold')
    ax1.set_title('🏪 Visualização da Prateleira com Produtos Otimizados (SSAP)', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks(range(0, capacity + 1, max(1, capacity // 20)))
    ax1.set_yticks([])
    
    # Adicionar legenda de capacidade
    ax1.text(capacity/2, 13, 
            f'Capacidade Total: {capacity} | Utilizado: {x_position} | GCD: {gcd}',
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#3498DB', 
                     edgecolor='#2C3E50', linewidth=2, alpha=0.9),
            color='white')
    
    # ============ TODOS OS PRODUTOS DISPONÍVEIS ============
    spacing = 5
    total_catalog_width = sum(weights) + spacing * (len(weights) - 1) + 10
    
    ax2.set_xlim(0, total_catalog_width)
    ax2.set_ylim(0, 12)
    ax2.set_aspect('equal')
    
    x_pos = 0
    
    for idx in range(len(weights)):
        width = weights[idx]
        value = values[idx]
        
        # Verificar se o produto foi selecionado
        is_selected = any(item['index'] == idx for item in selected_items)
        
        height = min(8, 3 + (value / max(values)) * 5)
        color = colors[idx % len(colors)]
        
        alpha = 0.9 if is_selected else 0.3
        edgecolor = '#27AE60' if is_selected else '#95A5A6'
        linewidth = 3 if is_selected else 1
        
        product = patches.Rectangle((x_pos, 1), width, height,
                                    linewidth=linewidth, edgecolor=edgecolor,
                                    facecolor=color, alpha=alpha)
        ax2.add_patch(product)
        
        # Adicionar rótulo
        label_color = 'white' if is_selected else 'gray'
        ax2.text(x_pos + width/2, height/2 + 1, 
                f'P{idx}\n${value}\nW:{width}',
                ha='center', va='center', fontsize=8, 
                fontweight='bold' if is_selected else 'normal',
                color=label_color,
                bbox=dict(boxstyle='round,pad=0.2', 
                         facecolor='black' if is_selected else 'lightgray', 
                         alpha=0.6 if is_selected else 0.3))
        
        # Adicionar checkmark se selecionado
        if is_selected:
            ax2.text(x_pos + width/2, height + 2, '✓', 
                    ha='center', va='bottom', fontsize=16, 
                    color='#27AE60', fontweight='bold')
        
        x_pos += width + spacing
    
    ax2.set_xlabel('Produtos Disponíveis (verde = selecionado)', fontsize=12, fontweight='bold')
    ax2.set_title('📦 Catálogo Completo de Produtos', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_yticks([])
    
    plt.tight_layout()
    return fig


def create_comparison_chart(original_size: int, optimized_size: int, gcd: int):
    """
    Cria gráfico comparativo entre algoritmo clássico e otimizado com GCD.
    """
    fig = go.Figure()
    
    reduction_percent = ((original_size - optimized_size) / original_size) * 100
    
    fig.add_trace(go.Bar(
        name='Algoritmo Clássico',
        x=['Tamanho da Tabela DP'],
        y=[original_size],
        marker_color='#E74C3C',
        text=[f'{original_size:,} células'],
        textposition='auto',
        textfont=dict(size=14, color='white', family='Arial Black')
    ))
    
    fig.add_trace(go.Bar(
        name=f'Algoritmo GCD Otimizado (GCD={gcd})',
        x=['Tamanho da Tabela DP'],
        y=[optimized_size],
        marker_color='#27AE60',
        text=[f'{optimized_size:,} células'],
        textposition='auto',
        textfont=dict(size=14, color='white', family='Arial Black')
    ))
    
    fig.update_layout(
        title=f'📊 Comparação: Redução de {reduction_percent:.1f}% no Uso de Memória',
        title_font=dict(size=18, family='Arial Black'),
        xaxis_title='',
        yaxis_title='Número de Células na Tabela DP',
        yaxis_title_font=dict(size=14, family='Arial'),
        barmode='group',
        height=400,
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12)
        )
    )
    
    return fig


# ============ INTERFACE STREAMLIT ============

st.title("🏪 SSAP - Shelf Space Allocation Problem")
st.markdown("""
### 📚 Algoritmo de Programação Dinâmica com Otimização GCD
**Baseado no artigo:** *Dynamic programming approach for solving the retail shelf-space allocation problem*  
**Autores:** Kateryna Czerniachowska & Krzysztof Lutosławski (2021)
""")

st.divider()

# Sidebar para configurações
with st.sidebar:
    st.header("⚙️ Configurações")
    
    st.subheader("Exemplo do Artigo")
    use_article_example = st.checkbox("Usar exemplo do artigo científico", value=True)
    
    if use_article_example:
        st.info("**Exemplo do artigo (Seção 4.2)**\n\n8 produtos com pesos: {20, 25, 40, 50, 60, 75, 80, 100}\n\nCapacidade: 102")
        weights = [20, 25, 40, 50, 60, 75, 80, 100]
        values = [20, 25, 40, 50, 60, 75, 80, 100]
        capacity = 102
    else:
        st.subheader("Configuração Personalizada")
        
        num_items = st.slider("Número de produtos", 3, 12, 6)
        
        st.write("**Pesos dos produtos (largura):**")
        weights = []
        cols = st.columns(3)
        for i in range(num_items):
            with cols[i % 3]:
                weight = st.number_input(f"P{i}", min_value=1, max_value=100, 
                                        value=10 + i * 5, key=f"w{i}")
                weights.append(weight)
        
        st.write("**Valores dos produtos (lucro):**")
        value_mode = st.radio("Modo de valores:", 
                             ["Igual aos pesos", "Personalizado"])
        
        if value_mode == "Igual aos pesos":
            values = weights.copy()
        else:
            values = []
            cols = st.columns(3)
            for i in range(num_items):
                with cols[i % 3]:
                    value = st.number_input(f"Valor P{i}", min_value=1, max_value=200, 
                                          value=weights[i], key=f"v{i}")
                    values.append(value)
        
        capacity = st.slider("Capacidade da prateleira", 
                           min_value=10, max_value=200, value=100)
    
    st.divider()
    st.markdown("**Referência:**")
    st.caption("Czerniachowska, K., & Lutosławski, K. (2021). Dynamic programming approach for solving the retail shelf-space allocation problem. *Procedia Computer Science, 192*, 4320-4329.")

# Executar o algoritmo
result = solve_shelf_space_gcd(weights, values, capacity)

# ============ RESULTADOS ============
st.header("📊 Resultados da Otimização")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("💰 Valor Máximo", f"${result['max_value']}", 
             help="Lucro total máximo obtido com a alocação otimizada")

with col2:
    st.metric("📦 Produtos Selecionados", len(result['selected_items']),
             help="Número de produtos diferentes alocados na prateleira")

with col3:
    total_weight = sum(item['weight'] for item in result['selected_items'])
    st.metric("📏 Espaço Utilizado", f"{total_weight}/{capacity}",
             help="Largura total utilizada vs. capacidade disponível")

with col4:
    st.metric("🔢 GCD Calculado", result['gcd'],
             help="Máximo Divisor Comum entre todos os pesos")

st.divider()

# ============ VISUALIZAÇÃO DA PRATELEIRA ============
st.header("🎨 Visualização Gráfica da Prateleira")

fig_shelf = draw_shelf_visualization(weights, values, capacity, 
                                     result['selected_items'], result['gcd'])
st.pyplot(fig_shelf)

st.divider()

# ============ COMPARAÇÃO DE PERFORMANCE ============
st.header("⚡ Análise de Performance: GCD vs. Clássico")

col1, col2 = st.columns([2, 1])

with col1:
    fig_comparison = create_comparison_chart(
        result['original_table_size'], 
        result['table_size'],
        result['gcd']
    )
    st.plotly_chart(fig_comparison, width='stretch')

with col2:
    st.subheader("📈 Métricas de Otimização")
    
    reduction = result['original_table_size'] - result['table_size']
    reduction_percent = (reduction / result['original_table_size']) * 100
    
    st.metric("Células Economizadas", f"{reduction:,}", 
             f"-{reduction_percent:.1f}%",
             help="Redução no tamanho da tabela de programação dinâmica")
    
    st.info(f"""
    **Algoritmo Clássico:**  
    Tabela: {result['original_table_size']:,} células
    
    **Algoritmo GCD Otimizado:**  
    Tabela: {result['table_size']:,} células
    
    **Economia de Memória:**  
    {reduction_percent:.2f}%
    """)

st.divider()

# ============ DETALHES DOS PRODUTOS SELECIONADOS ============
st.header("📋 Produtos Selecionados (Detalhes)")

if result['selected_items']:
    items_data = []
    for item in sorted(result['selected_items'], key=lambda x: x['index']):
        items_data.append({
            "Produto": f"P{item['index']}",
            "Largura (peso)": item['weight'],
            "Valor (lucro)": f"${item['value']}",
            "Razão Valor/Peso": f"{item['value']/item['weight']:.2f}"
        })
    
    st.dataframe(items_data, width='stretch', hide_index=True)
else:
    st.warning("Nenhum produto foi selecionado com os parâmetros atuais.")

# ============ EXPLICAÇÃO DO ALGORITMO ============
with st.expander("ℹ️ Como funciona o Algoritmo GCD?"):
    st.markdown("""
    ### Otimização por GCD (Greatest Common Divisor)
    
    O algoritmo implementado é baseado no artigo de **Czerniachowska e Lutosławski (2021)** e utiliza 
    programação dinâmica otimizada com o cálculo do Máximo Divisor Comum (GCD).
    
    #### Passos do Algoritmo:
    
    1. **Calcular o GCD** de todos os pesos dos produtos
    2. **Reduzir a dimensão** do problema dividindo capacidade e pesos pelo GCD
    3. **Criar tabela DP** com dimensões reduzidas: `[N+1][capacidade_reduzida+1]`
    4. **Preencher a tabela** usando programação dinâmica clássica
    5. **Reconstruir a solução** por backtracking para encontrar os itens selecionados
    
    #### Vantagens:
    
    - ✅ **Redução significativa** no tamanho da tabela de programação dinâmica
    - ✅ **Menor uso de memória** (economia de até 90% em alguns casos)
    - ✅ **Tempo de execução reduzido** devido ao menor número de células a processar
    - ✅ **Solução ótima garantida** (não é uma heurística)
    
    #### Complexidade:
    
    - **Clássico:** O(N × W) onde W é a capacidade
    - **GCD Otimizado:** O(N × W/GCD) - significativamente menor quando GCD > 1
    """)

st.divider()
st.caption("Desenvolvido com Streamlit | Baseado em Czerniachowska & Lutosławski (2021)")

def main():
    print("Hello from repl-nix-workspace!")


if __name__ == "__main__":
    main()
