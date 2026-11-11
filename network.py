import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================================
# Этап 1. Построение графа и матрицы смежности
# ==========================================
G = nx.Graph()
edges = [(1,2), (1,3), (2,3), (2,4), (3,5), (4,5), (4,6), (5,6)]
G.add_edges_from(edges)

print("="*50)
print("1. МАТРИЦА СМЕЖНОСТИ И СТЕПЕНИ")
print("="*50)
A = nx.to_numpy_array(G, nodelist=[1,2,3,4,5,6], dtype=int)
print("Матрица смежности A:")
print(A)

degrees = A.sum(axis=1)
print("\nСтепени вершин (1-6):", degrees)

A2 = A @ A
print("\nМатрица A^2 (маршруты длины 2):")
print(A2)

# ==========================================
# Этап 2. Критические элементы и разрез
# ==========================================
print("\n" + "="*50)
print("2. КРИТИЧЕСКИЕ ЭЛЕМЕНТЫ И РАЗРЕЗ")
print("="*50)
articulation_points = list(nx.articulation_points(G))
bridges = list(nx.bridges(G))
print(f"Точки сочленения: {articulation_points if articulation_points else 'НЕТ'}")
print(f"Мосты: {bridges if bridges else 'НЕТ'}")

# Проверка удаления узла 3 (пример из ТЗ)
G_test = G.copy()
G_test.remove_node(3)
is_connected_after_3 = nx.is_connected(G_test)
print(f"Связна ли сеть после удаления коммутатора 3? -> {is_connected_after_3}")

# Минимальный разрез между зонами
# Используем глобальный разрез (алгоритм Штор-Вагнера), т.к. он надежнее для неориентированных графов
cut_value, partition = nx.stoer_wagner(G)
print(f"Минимальный разрез между зонами: {cut_value}")

# ==========================================
# Этап 3. Перебор вариантов и Спектральный анализ
# ==========================================
print("\n" + "="*50)
print("3. АНАЛИЗ ВАРИАНТОВ ДОПОЛНИТЕЛЬНОГО КАБЕЛЯ (lambda_2)")
print("="*50)

variants = [(1,4), (1,5), (1,6), (2,5), (2,6), (3,4), (3,6)]
results = []

# Функция для расчета числа Фидлера (второе собств. значение Лапласиана)
def get_fiedler(graph):
    L = nx.laplacian_matrix(graph).toarray()
    eigenvalues = np.linalg.eigh(L)[0]
    return eigenvalues[1] # Индекс 1, так как индекс 0 всегда ~0

# Считаем для исходной сети
initial_lambda2 = get_fiedler(G)
print(f"Исходная сеть: min_cut = {cut_value}, lambda_2 = {initial_lambda2:.4f}")

# Считаем для всех 7 вариантов
for u, v in variants:
    G_var = G.copy()
    G_var.add_edge(u, v)
    
    # Пересчет min cut
    var_cut, _ = nx.stoer_wagner(G_var)
    
    # Пересчет критических элементов
    var_art = len(list(nx.articulation_points(G_var)))
    var_brd = len(list(nx.bridges(G_var)))
    
    # Пересчет lambda_2
    var_lambda2 = get_fiedler(G_var)
    
    results.append({
        "Кабель": f"{{{u},{v}}}",
        "Min Cut": var_cut,
        "Крит. элементов": var_art + var_brd,
        "lambda_2": var_lambda2
    })

df = pd.DataFrame(results)
print("\nТаблица результатов по вариантам:")
print(df.to_string(index=False))

# Выбор лучшего варианта
best_variant = df.loc[df['lambda_2'].idxmax()]
print("\n" + "-"*50)
print(f"ЛУЧШИЙ ВАРИАНТ (по максимальному lambda_2): {best_variant['Кабель']}")
print(f"Новый lambda_2: {best_variant['lambda_2']:.4f}")
print("-"*50)

# ==========================================
# Этап 4. Визуализация для презентации
# ==========================================

# Создадим папку для картинок, чтобы не мусорить в корне
if not os.path.exists('images'):
    os.makedirs('images')

# 1. Отрисовка исходного графа (Слайд 3)
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G, seed=42) # Фиксируем позиции для красоты
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=800, font_size=16, font_weight='bold')
plt.title("Схема сети (Серверная 1-2-3, Зал 4-5-6)")
plt.savefig("images/network_scheme.png", bbox_inches='tight')
plt.close() # Закрываем график, чтобы не блокировал программу

# 2. График lambda_2 по вариантам (Слайд 10)
plt.figure(figsize=(10, 6))
variants_names = ["Исходная"] + df["Кабель"].tolist()
lambdas = [initial_lambda2] + df["lambda_2"].tolist()

colors = ['gray'] + ['blue'] * len(df)
# Выделяем победителя красным (индекс +1, т.к. первый столбец - исходная сеть)
best_idx = df['lambda_2'].idxmax() + 1 
colors[best_idx] = 'red'

plt.bar(variants_names, lambdas, color=colors)
plt.title("Число Фидлера (lambda_2) для разных вариантов кабеля")
plt.ylabel("lambda_2 (чем больше, тем лучше)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("images/lambda2_chart.png", bbox_inches='tight')
plt.close()

print("\n" + "="*50)
print("ГРАФИКИ УСПЕШНО СОХРАНЕНЫ В ПАПКУ 'images'")
print("="*50)