import heapq
from typing import List, Tuple, Optional

# Job = (processing_time p, due_date d, id_opcional)
Job = Tuple[int, int, Optional[int]]

def moore_hodgson_onlogn(jobs: List[Job]) -> Tuple[List[Job], List[Job], int]:
    """
    Versão O(n log n) do algoritmo de Moore-Hodgson usando max-heap.
    Retorna:
        - schedule_final: sequência completa (on-time + rejected)
        - on_time_jobs: apenas as tarefas pontuais (em ordem EDD)
        - num_late: número de tarefas atrasadas (ótimo)
    """
    # Passo 1: Ordenar por due date crescente (EDD) - O(n log n)
    sorted_jobs = sorted(jobs, key=lambda x: x[1])
    n = len(sorted_jobs)
    
    # Max-heap: armazenamos (-p, índice) para simular max-heap com heapq (min-heap nativo)
    heap: List[Tuple[int, int]] = []
    current_sum = 0
    removed_indices = set()  # índices das tarefas rejeitadas
    
    # Passo 2: Adiciona jobs um por um em ordem EDD
    for i, job in enumerate(sorted_jobs):
        p, d, job_id = job if len(job) == 3 else (job[0], job[1], i)
        
        # Adiciona o job atual ao conjunto
        heapq.heappush(heap, (-p, i))
        current_sum += p
        
        # Enquanto o conjunto atual não cabe até o deadline atual → remove o maior p
        while current_sum > d and heap:
            neg_p, idx = heapq.heappop(heap)
            current_sum += neg_p  # equivale a current_sum -= p_original
            removed_indices.add(idx)
    
    # Reconstrói as listas
    on_time_jobs = []
    rejected_jobs = []
    
    for i, job in enumerate(sorted_jobs):
        if i not in removed_indices:
            on_time_jobs.append(job)
        else:
            rejected_jobs.append(job)
    
    # Sequência final: tarefas pontuais (em EDD) + tarefas atrasadas (qualquer ordem)
    schedule_final = on_time_jobs + rejected_jobs
    
    return schedule_final, on_time_jobs, len(rejected_jobs)


# ========================= TESTE COM O EXEMPLO CLÁSSICO DO ARTIGO =========================
jobs = [
    (4, 6, 1),
    (1, 8, 2),
    (6, 9, 3),
    (3, 11, 4),
    (6, 20, 5),
    (8, 25, 6),
    (7, 28, 7),
    (10, 35, 8)
]

schedule, on_time, num_late = moore_hodgson_onlogn(jobs)

print(f"Número ótimo de tarefas atrasadas: {num_late}")
print("Tarefas pontuais (ordem EDD):")
for job in on_time:
    print(f"Job {job[2]}: p={job[0]}, d={job[1]}")

print("\nSequência final de execução:")
for job in schedule:
    print(f"Job {job[2]}: p={job[0]}, d={job[1]}")