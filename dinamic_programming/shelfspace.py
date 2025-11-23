import math

def solve_shelf_space_gcd(weights, values, capacity):
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
    # Referência: Seção 4.2, passo "Find the GCD among all weights" [cite: 226]
    common_divisor = math.gcd(*weights)
    
    # 2. Reduzir a dimensão do problema
    # Se a capacidade não for divisível pelo GCD, o espaço restante é inútil
    # pois todos os itens são múltiplos do GCD.
    scaled_capacity = capacity // common_divisor
    scaled_weights = [w // common_divisor for w in weights]
    
    print(f"--- Otimização GCD ---")
    print(f"GCD calculado: {common_divisor}")
    print(f"Capacidade original: {capacity} -> Capacidade reduzida: {scaled_capacity}")
    
    # 3. Inicializar a tabela de Programação Dinâmica (V)
    # Dimensão reduzida: [N+1][scaled_capacity+1]
    # Referência: Claim 1 
    dp = [[0 for _ in range(scaled_capacity + 1)] for _ in range(n + 1)]
    
    # 4. Preencher a tabela (Lógica Iterativa)
    for i in range(1, n + 1):
        w_current = scaled_weights[i-1] # Peso reduzido do item atual
        v_current = values[i-1]         # Valor do item atual
        
        for w in range(1, scaled_capacity + 1):
            if w_current <= w:
                # Max entre não levar o item vs levar o item
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
        "table_size": (n + 1) * (scaled_capacity + 1)
    }

# --- Execução com o Exemplo do Artigo (Seção 4.2) ---
# Dados extraídos do texto:
# N = 8 itens
# Pesos = {20, 25, 40, 50, 60, 75, 80, 100}
# Capacidade (W) = 102
# o artigo não fornece os valores (lucro) neste exemplo específico, apenas os pesos para demonstrar a redução da tabela. 


weights_article = [20, 25, 40, 50, 60, 75, 80, 100]
values_article = [20, 25, 40, 50, 60, 75, 80, 100] # Assumindo lucro proporcional ao tamanho
capacity_article = 102

result = solve_shelf_space_gcd(weights_article, values_article, capacity_article)

print("\n--- Resultados ---")
print(f"Valor Máximo: {result['max_value']}")
print(f"Itens Selecionados: {result['selected_items']}")
print(f"Tamanho real da tabela na memória: {result['table_size']} células")
print(f"Tamanho se fosse o algoritmo clássico: {(len(weights_article)+1) * (capacity_article+1)} células")