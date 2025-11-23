import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np
import math
from typing import Dict, List, Tuple
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib
matplotlib.use("TkAgg")


def solve_shelf_space_gcd(weights: List[int], values: List[int], capacity: int) -> Dict:
    """
    Implementação do algoritmo de Programação Dinâmica otimizado por GCD
    conforme proposto por Czerniachowska e Lutosławski (2021).
    """
    n = len(weights)
    
    # 1. Calcular o GCD
    common_divisor = math.gcd(*weights)
    
    # 2. Reduzir a dimensão do problema
    scaled_capacity = capacity // common_divisor
    scaled_weights = [w // common_divisor for w in weights]
    
    # 3. Inicializar a tabela de Programação Dinâmica
    dp = [[0 for _ in range(scaled_capacity + 1)] for _ in range(n + 1)]
    
    # 4. Preencher a tabela
    for i in range(1, n + 1):
        w_current = scaled_weights[i-1]
        v_current = values[i-1]
        
        for w in range(1, scaled_capacity + 1):
            if w_current <= w:
                dp[i][w] = max(dp[i-1][w], 
                               v_current + dp[i-1][w - w_current])
            else:
                dp[i][w] = dp[i-1][w]
                
    # 5. Reconstrução da solução
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


def create_3d_shelf_visualization(weights: List[int], values: List[int], 
                                   capacity: int, selected_items: List[Dict],
                                   shelf_depth: float = 30, shelf_height: float = 25):
    """
    Cria visualização 3D realista da prateleira com produtos alocados.
    """
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Cores vibrantes para os produtos
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', 
              '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B400', '#52B788']
    
    # Desenhar estrutura da prateleira
    shelf_thickness = 2
    
    # Base da prateleira
    vertices_base = [
        [0, 0, 0], [capacity, 0, 0], 
        [capacity, shelf_depth, 0], [0, shelf_depth, 0],
        [0, 0, shelf_thickness], [capacity, 0, shelf_thickness],
        [capacity, shelf_depth, shelf_thickness], [0, shelf_depth, shelf_thickness]
    ]
    
    faces_base = [
        [vertices_base[0], vertices_base[1], vertices_base[5], vertices_base[4]],
        [vertices_base[1], vertices_base[2], vertices_base[6], vertices_base[5]],
        [vertices_base[2], vertices_base[3], vertices_base[7], vertices_base[6]],
        [vertices_base[3], vertices_base[0], vertices_base[4], vertices_base[7]],
        [vertices_base[4], vertices_base[5], vertices_base[6], vertices_base[7]],
        [vertices_base[0], vertices_base[1], vertices_base[2], vertices_base[3]]
    ]
    
    shelf_collection = Poly3DCollection(faces_base, alpha=0.3, 
                                        facecolor='#8B4513', edgecolor='#654321')
    ax.add_collection3d(shelf_collection)
    
    # Suportes laterais
    support_width = 3
    support_vertices_left = [
        [-support_width, 0, -5], [0, 0, -5],
        [0, shelf_depth, -5], [-support_width, shelf_depth, -5],
        [-support_width, 0, shelf_height], [0, 0, shelf_height],
        [0, shelf_depth, shelf_height], [-support_width, shelf_depth, shelf_height]
    ]
    
    support_faces_left = [
        [support_vertices_left[0], support_vertices_left[1], 
         support_vertices_left[5], support_vertices_left[4]],
        [support_vertices_left[1], support_vertices_left[2], 
         support_vertices_left[6], support_vertices_left[5]],
        [support_vertices_left[2], support_vertices_left[3], 
         support_vertices_left[7], support_vertices_left[6]],
        [support_vertices_left[3], support_vertices_left[0], 
         support_vertices_left[4], support_vertices_left[7]],
        [support_vertices_left[4], support_vertices_left[5], 
         support_vertices_left[6], support_vertices_left[7]],
        [support_vertices_left[0], support_vertices_left[1], 
         support_vertices_left[2], support_vertices_left[3]]
    ]
    
    support_collection_left = Poly3DCollection(support_faces_left, alpha=0.6,
                                               facecolor='#654321', edgecolor='#3E2723')
    ax.add_collection3d(support_collection_left)
    
    # Suporte direito
    support_vertices_right = [
        [capacity, 0, -5], [capacity + support_width, 0, -5],
        [capacity + support_width, shelf_depth, -5], [capacity, shelf_depth, -5],
        [capacity, 0, shelf_height], [capacity + support_width, 0, shelf_height],
        [capacity + support_width, shelf_depth, shelf_height], [capacity, shelf_depth, shelf_height]
    ]
    
    support_faces_right = [
        [support_vertices_right[0], support_vertices_right[1], 
         support_vertices_right[5], support_vertices_right[4]],
        [support_vertices_right[1], support_vertices_right[2], 
         support_vertices_right[6], support_vertices_right[5]],
        [support_vertices_right[2], support_vertices_right[3], 
         support_vertices_right[7], support_vertices_right[6]],
        [support_vertices_right[3], support_vertices_right[0], 
         support_vertices_right[4], support_vertices_right[7]],
        [support_vertices_right[4], support_vertices_right[5], 
         support_vertices_right[6], support_vertices_right[7]],
        [support_vertices_right[0], support_vertices_right[1], 
         support_vertices_right[2], support_vertices_right[3]]
    ]
    
    support_collection_right = Poly3DCollection(support_faces_right, alpha=0.6,
                                                facecolor='#654321', edgecolor='#3E2723')
    ax.add_collection3d(support_collection_right)
    
    # Desenhar produtos selecionados
    x_position = 0
    for idx, item in enumerate(reversed(selected_items)):
        item_idx = item['index']
        width = item['weight']
        value = item['value']
        
        # Altura proporcional ao valor
        height = shelf_thickness + 2 + (value / max(values)) * 15
        
        color = colors[item_idx % len(colors)]
        
        # Criar caixa do produto (3D)
        product_vertices = [
            [x_position, 0, shelf_thickness],
            [x_position + width, 0, shelf_thickness],
            [x_position + width, shelf_depth, shelf_thickness],
            [x_position, shelf_depth, shelf_thickness],
            [x_position, 0, height],
            [x_position + width, 0, height],
            [x_position + width, shelf_depth, height],
            [x_position, shelf_depth, height]
        ]
        
        product_faces = [
            [product_vertices[0], product_vertices[1], product_vertices[5], product_vertices[4]],
            [product_vertices[1], product_vertices[2], product_vertices[6], product_vertices[5]],
            [product_vertices[2], product_vertices[3], product_vertices[7], product_vertices[6]],
            [product_vertices[3], product_vertices[0], product_vertices[4], product_vertices[7]],
            [product_vertices[4], product_vertices[5], product_vertices[6], product_vertices[7]],
            [product_vertices[0], product_vertices[1], product_vertices[2], product_vertices[3]]
        ]
        
        product_collection = Poly3DCollection(product_faces, alpha=0.9,
                                              facecolor=color, edgecolor='black', linewidths=1.5)
        ax.add_collection3d(product_collection)
        
        # Adicionar etiqueta
        ax.text(x_position + width/2, shelf_depth/2, height + 2,
                f'P{item_idx}\n${value}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        x_position += width
    
    # Configurações dos eixos
    ax.set_xlabel('Largura (unidades)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Profundidade', fontsize=11, fontweight='bold')
    ax.set_zlabel('Altura', fontsize=11, fontweight='bold')
    
    ax.set_xlim(-5, capacity + 10)
    ax.set_ylim(-2, shelf_depth + 5)
    ax.set_zlim(-5, shelf_height + 5)
    
    # Título
    total_weight = sum(item['weight'] for item in selected_items)
    usage_percent = (total_weight / capacity) * 100
    
    ax.set_title(f'🏪 Visualização 3D da Prateleira Otimizada\n'
                 f'Capacidade: {capacity} | Utilizado: {total_weight} ({usage_percent:.1f}%) | '
                 f'Valor Total: ${sum(item["value"] for item in selected_items)}',
                 fontsize=13, fontweight='bold', pad=20)
    
    # Melhorar visualização
    ax.view_init(elev=20, azim=45)
    
    return fig


def create_multiple_shelves_view(weights: List[int], values: List[int], 
                                  capacity: int, selected_items: List[Dict],
                                  num_shelves: int = 4):
    """
    Cria visualização de múltiplas prateleiras em um supermercado.
    """
    fig, axes = plt.subplots(num_shelves, 1, figsize=(16, 3*num_shelves))
    
    if num_shelves == 1:
        axes = [axes]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', 
              '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B400', '#52B788']
    
    for shelf_idx, ax in enumerate(axes):
        ax.set_xlim(0, capacity + 10)
        ax.set_ylim(0, 12)
        ax.set_aspect('equal')
        
        # Desenhar prateleira
        shelf_rect = patches.Rectangle((0, 0), capacity, 10,
                                       linewidth=2, edgecolor='#2C3E50',
                                       facecolor='#ECF0F1', alpha=0.4)
        ax.add_patch(shelf_rect)
        
        # Suportes
        support_left = patches.Rectangle((-2, -1), 2, 12,
                                         linewidth=2, edgecolor='#34495E',
                                         facecolor='#7F8C8D')
        support_right = patches.Rectangle((capacity, -1), 2, 12,
                                          linewidth=2, edgecolor='#34495E',
                                          facecolor='#7F8C8D')
        ax.add_patch(support_left)
        ax.add_patch(support_right)
        
        # Simular diferentes produtos em cada prateleira
        if shelf_idx == 0:
            # Primeira prateleira com itens selecionados
            x_position = 0
            for idx, item in enumerate(reversed(selected_items)):
                item_idx = item['index']
                width = item['weight']
                value = item['value']
                
                height = min(8, 3 + (value / max(values)) * 4)
                color = colors[item_idx % len(colors)]
                
                product = FancyBboxPatch((x_position, 1), width, height,
                                        boxstyle="round,pad=0.1",
                                        linewidth=2, edgecolor='#2C3E50',
                                        facecolor=color, alpha=0.85)
                ax.add_patch(product)
                
                ax.text(x_position + width/2, height/2 + 1,
                       f'P{item_idx}\n${value}',
                       ha='center', va='center', fontsize=9,
                       fontweight='bold', color='white')
                
                x_position += width
                
            ax.set_title(f'Prateleira #{shelf_idx+1} - OTIMIZADA ✓',
                        fontsize=12, fontweight='bold', color='#27AE60')
        else:
            # Outras prateleiras com produtos simulados
            num_items = np.random.randint(3, 7)
            x_pos = 0
            for i in range(num_items):
                w = np.random.choice([15, 20, 25, 30])
                if x_pos + w > capacity:
                    break
                h = np.random.randint(4, 8)
                color = colors[np.random.randint(0, len(colors))]
                
                product = patches.Rectangle((x_pos, 1), w, h,
                                            linewidth=1.5, edgecolor='#95A5A6',
                                            facecolor=color, alpha=0.4)
                ax.add_patch(product)
                x_pos += w + 2
            
            ax.set_title(f'Prateleira #{shelf_idx+1}',
                        fontsize=12, fontweight='bold', color='#7F8C8D')
        
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.set_xticks(range(0, capacity + 1, max(1, capacity // 10)))
        ax.set_yticks([])
        ax.set_xlabel('Largura (unidades)', fontsize=10)
    
    plt.tight_layout()
    return fig


def create_animated_allocation(weights: List[int], values: List[int],
                                capacity: int, selected_items: List[Dict]):
    """
    Cria animação do processo de alocação dos produtos na prateleira.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
              '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B400', '#52B788']
    
    ax.set_xlim(0, capacity + 10)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal')
    
    # Prateleira fixa
    shelf_rect = patches.Rectangle((0, 0), capacity, 10,
                                   linewidth=3, edgecolor='#2C3E50',
                                   facecolor='#ECF0F1', alpha=0.3)
    ax.add_patch(shelf_rect)
    
    support_left = patches.Rectangle((-2, -2), 2, 14,
                                     linewidth=2, edgecolor='#34495E',
                                     facecolor='#7F8C8D')
    support_right = patches.Rectangle((capacity, -2), 2, 14,
                                      linewidth=2, edgecolor='#34495E',
                                      facecolor='#7F8C8D')
    ax.add_patch(support_left)
    ax.add_patch(support_right)
    
    ax.set_xlabel('Largura da Prateleira', fontsize=12, fontweight='bold')
    ax.set_title('🎬 Animação: Alocação Otimizada de Produtos', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_yticks([])
    
    products_patches = []
    texts = []
    
    def init():
        return products_patches + texts
    
    def animate(frame):
        if frame < len(selected_items):
            item = list(reversed(selected_items))[frame]
            item_idx = item['index']
            width = item['weight']
            value = item['value']
            
            x_position = sum(list(reversed(selected_items))[i]['weight'] 
                            for i in range(frame))
            
            height = min(8, 3 + (value / max(values)) * 5)
            color = colors[item_idx % len(colors)]
            
            product = patches.Rectangle((x_position, 1), width, height,
                                        linewidth=2, edgecolor='#2C3E50',
                                        facecolor=color, alpha=0.9)
            ax.add_patch(product)
            products_patches.append(product)
            
            text = ax.text(x_position + width/2, height/2 + 1,
                          f'P{item_idx}\n${value}',
                          ha='center', va='center', fontsize=10,
                          fontweight='bold', color='white',
                          bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor='black', alpha=0.6))
            texts.append(text)
        
        return products_patches + texts
    
    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=len(selected_items) + 5,
                        interval=800, blit=True, repeat=True)
    
    return fig, anim


# ==================== EXECUÇÃO PRINCIPAL ====================

if __name__ == "__main__":
    # Exemplo do artigo
    weights = [20, 25, 40, 50, 60, 75, 80, 100]
    values = [20, 25, 40, 50, 60, 75, 80, 100]
    capacity = 102
    
    print("="*70)
    print("🏪 SSAP - SHELF SPACE ALLOCATION PROBLEM")
    print("="*70)
    print(f"Produtos: {len(weights)}")
    print(f"Pesos: {weights}")
    print(f"Valores: {values}")
    print(f"Capacidade: {capacity}")
    print("="*70)
    
    # Resolver o problema
    result = solve_shelf_space_gcd(weights, values, capacity)
    
    print(f"\n📊 RESULTADOS:")
    print(f"   💰 Valor Máximo: ${result['max_value']}")
    print(f"   📦 Produtos Selecionados: {len(result['selected_items'])}")
    print(f"   🔢 GCD: {result['gcd']}")
    print(f"   📉 Redução da Tabela: {result['original_table_size']} → {result['table_size']}")
    
    total_weight = sum(item['weight'] for item in result['selected_items'])
    print(f"   📏 Espaço Utilizado: {total_weight}/{capacity} ({(total_weight/capacity)*100:.1f}%)")
    
    print(f"\n📋 Produtos Escolhidos:")
    for item in sorted(result['selected_items'], key=lambda x: x['index']):
        print(f"   • P{item['index']}: Largura={item['weight']}, Valor=${item['value']}")
    
    print("\n" + "="*70)
    print("📈 Gerando Visualizações...")
    print("="*70)
    
    # Criar visualizações
    print("\n1️⃣  Criando visualização 3D da prateleira...")
    fig1 = create_3d_shelf_visualization(weights, values, capacity, 
                                         result['selected_items'])
    
    print("2️⃣  Criando visualização de múltiplas prateleiras...")
    fig2 = create_multiple_shelves_view(weights, values, capacity,
                                        result['selected_items'], num_shelves=4)
    
    print("3️⃣  Criando animação de alocação...")
    fig3, anim = create_animated_allocation(weights, values, capacity,
                                            result['selected_items'])
    
    print("\n✅ Visualizações geradas com sucesso!")
    print("🖼️  Exibindo gráficos...")
    
    plt.show()
    
    print("\n" + "="*70)
    print("✨ Programa finalizado!")
    print("="*70)