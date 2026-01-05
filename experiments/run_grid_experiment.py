""" Эксперимент по созданию карт достижимости у методов генерации примитивов движения. """

import numpy as np
import argparse
import csv
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import sys

try:
    sys.path.append("../common/")
    sys.path.append("../trajectory-generation/")
    from PRIM_structs import State
    from trajectory_optimization import optimization_Newton
    from baseline_trajectory_optimization import baseline_optimization_Newton
except ImportError as e:
    print(f"Ошибка импорта модулей: {e}")
    sys.exit(1)



def generate_grid_tasks():
    """Генерирует 1200 тестовых сценариев для сетки 10x10."""
    tasks = []

    # Единое начальное состояние -- нулевые координаты, нулевой угол направление (горизонтально вправо) и нулевая кривизна:
    start_state = State(0.0, 0.0, 0.0, 0.0)
    # 12 целевых направлений с шагом 30 градусов:
    angles_raw = np.deg2rad(np.arange(0, 360, 30))

    # Сетка 10x10 в квадрате [-1, 1] x [-1, 1]
    for i in range(10):      # Индекс по Y
        for j in range(10):  # Индекс по X
            goal_x = -4.5 + j * 1
            goal_y = -4.5 + i * 1
            
            for angle_idx, angle in enumerate(angles_raw):
                # Приводим угол к диапазону [-pi, pi], чтобы обеспечить кратчайший поворот от начального угла 0.
                angle_corrected = (angle + np.pi) % (2 * np.pi) - np.pi  
                goal_state = State(goal_x, goal_y, angle_corrected, k=0.0)
                task = (i, j, angle_idx, start_state, goal_state)
                tasks.append(task)
    return tasks



def run_single_grid_test(args):
    """Выполняет один тест для обоих методов. Предназначена для worker'а внутри одного потока."""
    cell_i, cell_j, angle_idx, start, goal = args
    iters, lr, eps = 300, 0.1, 1e-2
    result_dict = {
        'cell_i': cell_i,
        'cell_j': cell_j,
        'angle_idx': angle_idx,
    }

    # --- Baseline метод ---
    try:
        steps, traj = baseline_optimization_Newton(start, goal, iters=iters, lr=lr, eps=eps)
        if traj:
            result_dict['baseline_success'] = True
        else:
            result_dict['baseline_success'] = False
    except:
        result_dict['baseline_success'] = False

    # --- Proposed метод ---
    try:
        # Распаковываем кортеж (steps, traj) или ловим TypeError, если вернулся None
        steps, traj = optimization_Newton(start, goal, iters=iters, lr=lr, eps=eps)
        if traj:
            result_dict['proposed_success'] = True
        else: # Явный случай, когда traj is None
            result_dict['proposed_success'] = False
    except:
        result_dict['proposed_success'] = False
        
    return result_dict



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Запуск эксперимента по исследованию достижимости на сетке.")
    parser.add_argument('--output', default='grid_results_corrected.csv', help="CSV-файл для сохранения результатов.")
    parser.add_argument('--workers', type=int, default=cpu_count(), help="Количество параллельных процессов.")
    
    args = parser.parse_args()

    tasks = generate_grid_tasks()
    print(f"Сгенерировано {len(tasks)} тестовых сценариев с коррекцией углов.")

    print(f"Запуск тестов на {args.workers} процессах...")
    all_results = []
    with Pool(processes=args.workers) as pool:
        for result in tqdm(pool.imap_unordered(run_single_grid_test, tasks), total=len(tasks)):
            all_results.append(result)

    print(f"Сохранение результатов в '{args.output}'...")
    fieldnames = ['cell_i', 'cell_j', 'angle_idx', 'baseline_success', 'proposed_success']
    all_results.sort(key=lambda r: (r['cell_i'], r['cell_j'], r['angle_idx']))
    
    with open(args.output, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    
    baseline_success_count = sum(1 for r in all_results if r['baseline_success'])
    proposed_success_count = sum(1 for r in all_results if r['proposed_success'])
    total_count = len(all_results)
    
    print("\n" + "="*40 + "\nЭКСПЕРИМЕНТ ЗАВЕРШЕН\n" + "="*40)
    print(f"Baseline Success Rate: {baseline_success_count / total_count * 100:.2f}%")
    print(f"Proposed Success Rate: {proposed_success_count / total_count * 100:.2f}%")
    print(f"Результаты сохранены в {args.output}")
    print("="*40)
