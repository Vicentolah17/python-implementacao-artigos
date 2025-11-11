# ------------------------------------------------------------
# Algoritmo de Moore-Hodgson (Guloso)
# ------------------------------------------------------------
# Objetivo:
# Minimizar o número de tarefas atrasadas em um conjunto de jobs (tarefas)
# Cada job possui:
#   p_j -> tempo de processamento
#   d_j -> prazo (due date)
#
# O algoritmo ordena as tarefas pelo prazo (EDD - Earliest Due Date)
# e, sempre que o tempo acumulado ultrapassar um prazo,
# remove a tarefa mais longa entre as já aceitas.
#
# Técnica utilizada: Algoritmo Guloso (Greedy)
# Complexidade: O(n log n)
# ------------------------------------------------------------

from typing import List, Tuple
import heapq  # biblioteca para trabalhar com filas de prioridade (heaps)


# Classe que representa uma tarefa (job)
class Job:
    def __init__(self, idx: int, p: int, d: int):
        self.idx = idx   # identificador da tarefa
        self.p = p       # tempo de processamento
        self.d = d       # prazo (due date)

    def __repr__(self):
        return f"Job(id={self.idx}, p={self.p}, d={self.d})"


# Função principal: algoritmo de Moore-Hodgson
def moore_hodgson(jobs: List[Job]) -> Tuple[List[Job], List[Job]]:
    """
    Implementação do algoritmo de Moore-Hodgson para minimizar
    o número de tarefas atrasadas.

    Parâmetros:
        jobs: lista de objetos Job, cada um com (id, p, d)

    Retorna:
        (on_time_jobs, late_jobs)
        - on_time_jobs: lista de tarefas concluídas no prazo (em ordem de prazo)
        - late_jobs: lista de tarefas que ficaram atrasadas

    Etapas:
        1) Ordena as tarefas pelo prazo (EDD)
        2) Percorre a lista e adiciona cada tarefa provisoriamente
        3) Se o tempo total acumulado ultrapassar o prazo atual,
           remove a tarefa com maior tempo de processamento
    """

    # Passo 1: Ordenar as tarefas pelo prazo (EDD)
    jobs_sorted = sorted(jobs, key=lambda job: (job.d, job.idx))

    # Criar um heap (fila de prioridade) para armazenar as tarefas aceitas
    # Usaremos um "max heap" de tempos de processamento (Python usa min heap por padrão)
    # Então vamos inserir os tempos como negativos (-p)
    max_heap = []  # cada elemento será (-p, id, job)
    tempo_total = 0  # soma acumulada dos tempos
    tarefas_removidas = []  # tarefas que serão atrasadas (removidas do heap)

    # Passo 2: Percorrer as tarefas em ordem de prazo
    for job in jobs_sorted:
        # Adicionar o tempo da tarefa atual
        tempo_total += job.p
        # Inserir a tarefa no heap (armazenando p como negativo)
        heapq.heappush(max_heap, (-job.p, job.idx, job))

        # Verificar se o tempo total ultrapassou o prazo da tarefa atual
        if tempo_total > job.d:
            # Remover a tarefa mais longa (maior p)
            neg_p, _, removida = heapq.heappop(max_heap)
            # Corrigir o tempo total (subtrair o tempo da tarefa removida)
            tempo_total += neg_p  # neg_p é negativo, então soma equivale a subtração
            # Adicionar a tarefa removida à lista de atrasadas
            tarefas_removidas.append(removida)

    # As tarefas restantes no heap são as que terminam no prazo
    tarefas_no_prazo = {job.idx for (_, _, job) in max_heap}
    on_time_jobs = [job for job in jobs_sorted if job.idx in tarefas_no_prazo]

    return on_time_jobs, tarefas_removidas


# Função auxiliar apenas para facilitar a execução do programa
def executar_exemplo(jobs: List[Tuple[int, int, int]]):
    """
    Recebe uma lista de tuplas (id, p, d),
    cria os objetos Job e executa o algoritmo.
    """

    # Criar objetos Job a partir da lista de tuplas
    job_objs = [Job(idx, p, d) for (idx, p, d) in jobs]

    # Executar o algoritmo
    no_prazo, atrasadas = moore_hodgson(job_objs)

    # Exibir resultados
    print("\n============================")
    print("RESULTADO DA EXECUÇÃO")
    print("============================")
    print("Tarefas concluídas no prazo (em ordem EDD):")
    for job in no_prazo:
        print(f"  -> {job}")

    print("\nTarefas atrasadas (removidas):")
    for job in atrasadas:
        print(f"  -> {job}")

    print(f"\nNúmero total de tarefas atrasadas: {len(atrasadas)}\n")

    return no_prazo, atrasadas


# ------------------------------------------------------------
# BLOCO DE TESTE
# ------------------------------------------------------------
# Este bloco só é executado quando o arquivo é rodado diretamente
# (por exemplo, com 'python moore_hodgson.py' no terminal)
# ------------------------------------------------------------
if __name__ == "__main__":
    # Exemplo de entrada de dados (id, tempo de execução, prazo)
    exemplo_tarefas = [
        (1, 4, 6),
        (2, 1, 8),
        (3, 6, 9),
        (4, 3, 11),
        (5, 6, 20),
        (6, 8, 25),
        (7, 7, 28),
        (8, 10, 35),
    ]

    # Executa o exemplo
    executar_exemplo(exemplo_tarefas)
