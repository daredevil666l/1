import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# Данные из таблицы
x_data = [
    2, 2, 2, 3, 4, 6, 6, 11, 14, 14,
    17, 19, 20, 21, 22, 24, 26, 30, 37, 39,
    40, 44, 56, 56, 59, 59, 61, 64, 70, 70,
    81, 83, 85, 86, 91, 92, 92, 92, 95, 100
]

# Сортировка данных для ECDF
x_sorted = np.sort(x_data)
n = len(x_sorted)
y = np.arange(1, n + 1) / n  # Значения ECDF

# Первое окно: Эмпирическая функция распределения (ECDF)
plt.figure(figsize=(14, 8))
plt.step(x_sorted, y, where='post', label='ECDF')

# Добавление подписей значений F(x)
for x, f in zip(x_sorted, y):
    plt.text(
        x=x + 0.8,  # Сдвиг по X для читаемости
        y=f - 0.02, # Сдвиг по Y
        s=f"{f:.2f}",
        fontsize=8,
        rotation=45,
        ha='left',
        color='darkred'
    )

# Настройка отображения первого окна
plt.title('Эмпирическая функция распределения (ECDF) с значениями F(x)')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.grid(True)
plt.ylim(0, 1.1)
plt.xlim(0, 105)

# Добавление начальной точки (0,0)
plt.plot([0, x_sorted[0]], [0, 0], color='C0', linestyle='--')
plt.legend()
plt.show()

# Второе окно: Экспоненциальное распределение (CDF) и интегрирование f(x)
lambda_hat = 0.022  # Значение λ из формулы
x_min = 2  # Начало диапазона
x_max = 100  # Конец диапазона

# Создаём массив точек для построения графика
x_theoretical = np.linspace(x_min, x_max, 1000)
F_x_exp = 1 - np.exp(-lambda_hat * x_theoretical)  # CDF экспоненциального распределения
f_x_exp = lambda_hat * np.exp(-lambda_hat * x_theoretical)  # PDF экспоненциального распределения

# Второе окно: Экспоненциальное распределение
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Подграфик 1: CDF экспоненциального распределения
ax1.plot(x_theoretical, F_x_exp, label=f'CDF: F(x) = 1 - e^(-{lambda_hat}x)', color='blue')
ax1.set_title('Экспоненциальное распределение (CDF)')
ax1.set_xlabel('x')
ax1.set_ylabel('F(x)')
ax1.grid(True)
ax1.set_ylim(0, 1.1)
ax1.set_xlim(x_min, 105)
ax1.legend()

# Подграфик 2: PDF экспоненциального распределения с разбиением на интервалы
ax2.plot(x_theoretical, f_x_exp, label=f'PDF: f(x) = {lambda_hat}e^(-{lambda_hat}x)', color='green')

# Разбиение на интервалы для экспоненциального распределения
num_intervals_list = [6, 8, 3] # 6 вводится только один раз из-за одинаковых результатов
for num_intervals in num_intervals_list:
    interval_edges = np.linspace(x_min, x_max, num_intervals + 1)
    interval_width = (x_max - x_min) / num_intervals

    print(f"\nЭкспоненциальное распределение: Разбиение на {num_intervals} интервалов:")
    print("Границы интервалов:", [f"{edge:.1f}" for edge in interval_edges])

    areas_exp = []
    for i in range(num_intervals):
        x_start = interval_edges[i]
        x_end = interval_edges[i + 1]
        x_subinterval = np.linspace(x_start, x_end, 100)
        f_x_subinterval = lambda_hat * np.exp(-lambda_hat * x_subinterval)

        area = 0
        for j in range(len(x_subinterval) - 1):
            x1, x2 = x_subinterval[j], x_subinterval[j + 1]
            y1, y2 = f_x_subinterval[j], f_x_subinterval[j + 1]
            area += (x2 - x1) * (y1 + y2) / 2

        areas_exp.append(area)

        ax2.axvline(x=x_start, color='gray', linestyle='--', alpha=0.5)
        if i == num_intervals - 1:
            ax2.axvline(x=x_end, color='gray', linestyle='--', alpha=0.5)

    print(f"Площади под PDF (экспоненциальное) для {num_intervals} интервалов:")
    for i, area in enumerate(areas_exp):
        print(f"Интервал {i+1} ({interval_edges[i]:.1f}, {interval_edges[i+1]:.1f}): Площадь = {area:.4f}")
    total_area_exp = sum(areas_exp)
    print(f"Сумма площадей на диапазоне [{x_min}, {x_max}]: {total_area_exp:.4f}")

# Настройка подграфика PDF (экспоненциальное)
ax2.set_title('Плотность вероятности (PDF) экспоненциального распределения с разбиением на интервалы')
ax2.set_xlabel('x')
ax2.set_ylabel('f(x)')
ax2.grid(True)
ax2.set_xlim(x_min, 105)
ax2.legend()
plt.tight_layout()
plt.show()

# Третье окно: Равномерное распределение (CDF) и интегрирование f(x)
c = 2  # Нижняя граница
d = 100  # Верхняя граница
f_uniform = 1 / (d - c)  # Плотность вероятности: 1/(d-c)

# Создаём массив точек для равномерного распределения
x_uniform = np.linspace(x_min, x_max, 1000)
F_x_uniform = (x_uniform - c) / (d - c)  # CDF равномерного распределения
f_x_uniform = np.full_like(x_uniform, f_uniform)  # PDF равномерного распределения

# Третье окно: Равномерное распределение
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Подграфик 1: CDF равномерного распределения
ax1.plot(x_uniform, F_x_uniform, label=f'CDF: F(x) = (x-{c})/({d}-{c})', color='purple')
ax1.set_title('Равномерное распределение (CDF)')
ax1.set_xlabel('x')
ax1.set_ylabel('F(x)')
ax1.grid(True)
ax1.set_ylim(0, 1.1)
ax1.set_xlim(x_min, 105)
ax1.legend()

# Подграфик 2: PDF равномерного распределения с разбиением на интервалы
ax2.plot(x_uniform, f_x_uniform, label=f'PDF: f(x) = 1/({d}-{c}) = {f_uniform:.4f}', color='orange')

# Разбиение на интервалы для равномерного распределения
for num_intervals in num_intervals_list:
    interval_edges = np.linspace(x_min, x_max, num_intervals + 1)
    interval_width = (x_max - x_min) / num_intervals

    print(f"\nРавномерное распределение: Разбиение на {num_intervals} интервалов:")
    print("Границы интервалов:", [f"{edge:.1f}" for edge in interval_edges])

    areas_uniform = []
    for i in range(num_intervals):
        x_start = interval_edges[i]
        x_end = interval_edges[i + 1]
        x_subinterval = np.linspace(x_start, x_end, 100)
        f_x_subinterval = np.full_like(x_subinterval, f_uniform)

        area = 0
        for j in range(len(x_subinterval) - 1):
            x1, x2 = x_subinterval[j], x_subinterval[j + 1]
            y1, y2 = f_x_subinterval[j], f_x_subinterval[j + 1]
            area += (x2 - x1) * (y1 + y2) / 2

        areas_uniform.append(area)

        ax2.axvline(x=x_start, color='gray', linestyle='--', alpha=0.5)
        if i == num_intervals - 1:
            ax2.axvline(x=x_end, color='gray', linestyle='--', alpha=0.5)

    print(f"Площади под PDF (равномерное) для {num_intervals} интервалов:")
    for i, area in enumerate(areas_uniform):
        print(f"Интервал {i+1} ({interval_edges[i]:.1f}, {interval_edges[i+1]:.1f}): Площадь = {area:.4f}")
    total_area_uniform = sum(areas_uniform)
    print(f"Сумма площадей на диапазоне [{x_min}, {x_max}]: {total_area_uniform:.4f}")

# Настройка подграфика PDF (равномерное)
ax2.set_title('Плотность вероятности (PDF) равномерного распределения с разбиением на интервалы')
ax2.set_xlabel('x')
ax2.set_ylabel('f(x)')
ax2.grid(True)
ax2.set_xlim(x_min, 105)
ax2.legend()
plt.tight_layout()
plt.show()

# Четвёртое окно: Нормальное распределение (CDF) и интегрирование f(x)
v1 = 44.88  # Математическое ожидание
sigma = 38.17  # Стандартное отклонение

# Создаём массив точек для нормального распределения
x_normal = np.linspace(x_min, x_max, 1000)
F_x_normal = 0.5 * (1 + erf((x_normal - v1) / (sigma * np.sqrt(2))))  # CDF нормального распределения
f_x_normal = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x_normal - v1)**2) / (2 * sigma**2))  # PDF нормального распределения

# Четвёртое окно: Нормальное распределение
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Подграфик 1: CDF нормального распределения
ax1.plot(x_normal, F_x_normal, label=f'CDF: Normal(v1={v1}, σ={sigma})', color='red')
ax1.set_title('Нормальное распределение (CDF)')
ax1.set_xlabel('x')
ax1.set_ylabel('F(x)')
ax1.grid(True)
ax1.set_ylim(0, 1.1)
ax1.set_xlim(x_min, 105)
ax1.legend()

# Подграфик 2: PDF нормального распределения с разбиением на интервалы
ax2.plot(x_normal, f_x_normal, label=f'PDF: f(x) = (1/(σ√(2π)))e^(-(x-{v1})^2/(2*{sigma}^2))', color='darkred')

# Разбиение на интервалы для нормального распределения
for num_intervals in num_intervals_list:
    interval_edges = np.linspace(x_min, x_max, num_intervals + 1)
    interval_width = (x_max - x_min) / num_intervals

    print(f"\nНормальное распределение: Разбиение на {num_intervals} интервалов:")
    print("Границы интервалов:", [f"{edge:.1f}" for edge in interval_edges])

    areas_normal = []
    for i in range(num_intervals):
        x_start = interval_edges[i]
        x_end = interval_edges[i + 1]
        x_subinterval = np.linspace(x_start, x_end, 100)
        f_x_subinterval = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x_subinterval - v1)**2) / (2 * sigma**2))

        area = 0
        for j in range(len(x_subinterval) - 1):
            x1, x2 = x_subinterval[j], x_subinterval[j + 1]
            y1, y2 = f_x_subinterval[j], f_x_subinterval[j + 1]
            area += (x2 - x1) * (y1 + y2) / 2

        areas_normal.append(area)

        ax2.axvline(x=x_start, color='gray', linestyle='--', alpha=0.5)
        if i == num_intervals - 1:
            ax2.axvline(x=x_end, color='gray', linestyle='--', alpha=0.5)

    print(f"Площади под PDF (нормальное) для {num_intervals} интервалов:")
    for i, area in enumerate(areas_normal):
        print(f"Интервал {i+1} ({interval_edges[i]:.1f}, {interval_edges[i+1]:.1f}): Площадь = {area:.4f}")
    total_area_normal = sum(areas_normal)
    print(f"Сумма площадей на диапазоне [{x_min}, {x_max}]: {total_area_normal:.4f}")

# Настройка подграфика PDF (нормальное)
ax2.set_title('Плотность вероятности (PDF) нормального распределения с разбиением на интервалы')
ax2.set_xlabel('x')
ax2.set_ylabel('f(x)')
ax2.grid(True)
ax2.set_xlim(x_min, 105)
ax2.legend()
plt.tight_layout()
plt.show()

# Пятое окно: Сравнение всех функций и поиск максимального отклонения
# Создаём общий массив точек, включающий x_sorted и дополнительные точки
x_combined = np.sort(np.unique(np.concatenate([x_sorted, np.linspace(x_min, x_max, 17000)])))

# Вычисляем значения всех функций на x_combined
# Эмпирическая функция (ECDF): интерполируем значения
F_empirical = np.zeros_like(x_combined)
for i, x in enumerate(x_combined):
    idx = np.searchsorted(x_sorted, x, side='right')  # Находим индекс ближайшей точки справа
    if idx == 0:
        F_empirical[i] = 0  # До первой точки ECDF = 0
    else:
        F_empirical[i] = y[idx - 1]  # Значение ECDF на предыдущем шаге

# Экспоненциальная функция
F_exp_combined = 1 - np.exp(-lambda_hat * x_combined)

# Равномерная функция
F_uniform_combined = np.zeros_like(x_combined)
for i, x in enumerate(x_combined):
    if x < c:
        F_uniform_combined[i] = 0
    elif x > d:
        F_uniform_combined[i] = 1
    else:
        F_uniform_combined[i] = (x - c) / (d - c)

# Нормальная функция
F_normal_combined = 0.5 * (1 + erf((x_combined - v1) / (sigma * np.sqrt(2))))

# Вычисляем разности |F_empirical - F_theoretical|
diff_exp = np.abs(F_empirical - F_exp_combined)
diff_uniform = np.abs(F_empirical - F_uniform_combined)
diff_normal = np.abs(F_empirical - F_normal_combined)

# Находим максимальное отклонение
max_diff_exp = np.max(diff_exp)
max_diff_uniform = np.max(diff_uniform)
max_diff_normal = np.max(diff_normal)

# Определяем, с какой функцией достигается максимальное отклонение
max_diffs = {
    'Экспоненциальная': (max_diff_exp, diff_exp),
    'Равномерная': (max_diff_uniform, diff_uniform),
    'Нормальная': (max_diff_normal, diff_normal)
}

max_diff_label = max(max_diffs.items(), key=lambda x: x[1][0])[0]
max_diff_value = max(max_diffs.values(), key=lambda x: x[0])[0]
max_diff_idx = np.argmax(max_diffs[max_diff_label][1])
max_diff_x = x_combined[max_diff_idx]

# Определяем значения функций в точке максимального отклонения
F_empirical_at_max = F_empirical[max_diff_idx]
if max_diff_label == 'Экспоненциальная':
    F_theoretical_at_max = F_exp_combined[max_diff_idx]
elif max_diff_label == 'Равномерная':
    F_theoretical_at_max = F_uniform_combined[max_diff_idx]
else:
    F_theoretical_at_max = F_normal_combined[max_diff_idx]

# Пятое окно: Все функции на одном графике
plt.figure(figsize=(14, 8))
plt.step(x_combined, F_empirical, where='post', label='ECDF', color='black', linewidth=2)
plt.plot(x_combined, F_exp_combined, label='Экспоненциальная', color='blue')
plt.plot(x_combined, F_uniform_combined, label='Равномерная', color='purple')
plt.plot(x_combined, F_normal_combined, label='Нормальная', color='red')

# Добавляем вертикальную линию в точке максимального отклонения
plt.plot([max_diff_x, max_diff_x], [F_empirical_at_max, F_theoretical_at_max], color='green', linestyle='--', linewidth=2, label=f'Макс. отклонение ({max_diff_label})')

# Выводим информацию о максимальном отклонении
print(f"\nМаксимальное отклонение между ECDF и теоретическими функциями:")
print(f"С функцией: {max_diff_label}")
print(f"Значение отклонения: {max_diff_value:.4f}")
print(f"Точка x: {max_diff_x:.2f}")
print(f"F_empirical({max_diff_x:.2f}) = {F_empirical_at_max:.4f}")
print(f"F_{max_diff_label.lower()}({max_diff_x:.2f}) = {F_theoretical_at_max:.4f}")

# Настройка графика
plt.title('Сравнение эмпирической и теоретических функций распределения')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.grid(True)
plt.ylim(0, 1.1)
plt.xlim(x_min, 105)
plt.legend()
plt.show()

# Вывод таблицы
# Для таблицы используем только точки из x_sorted (где меняется ECDF)
print("\nТаблица значений функций распределения:")
print(f"{'i':<5} {'x':<8} {'F(x) эмпирическая':<18} {'F(x) экспоненциальная':<22} {'F(x) равномерная':<18} {'F(x) нормальная':<18}")
print("-" * 90)

# Вычисляем значения теоретических функций в точках x_sorted
F_exp_table = 1 - np.exp(-lambda_hat * x_sorted)
F_uniform_table = (x_sorted - c) / (d - c)
F_normal_table = 0.5 * (1 + erf((x_sorted - v1) / (sigma * np.sqrt(2))))

# Выводим таблицу
for i, (x, f_emp, f_exp, f_uniform, f_normal) in enumerate(zip(x_sorted, y, F_exp_table, F_uniform_table, F_normal_table), 1):
    print(f"{i:<5} {x:<8.2f} {f_emp:<18.4f} {f_exp:<22.4f} {f_uniform:<18.4f} {f_normal:<18.4f}")