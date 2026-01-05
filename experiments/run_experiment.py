""" Эксперимент по сравнению методов генерации примитивов движения. """

import numpy as np
import time
import os
import argparse
import csv
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import sys
sys.path.append("../common/") 
from PRIM_structs import State
sys.path.append("../trajectory-generation/")
from trajectory_optimization import optimization_Newton
from baseline_trajectory_optimization import baseline_optimization_Newton



# --- Функции генерации и загрузки тестов ---

def generate_experiments(num_base_points=20, radius=1.0):
    experiments = []
    theta_offsets = np.linspace(0, np.pi, num=7, endpoint=True)  # начальная ориентация - всегда 0, конечную тут генерируем равномерно от 0 до pi
    angle_points = np.linspace(0, 2 * np.pi, num=num_base_points, endpoint=False)  # равномерно генерируем num_base_points точек на окружности

    for angle in angle_points:
        xf, yf = radius * np.cos(angle), radius * np.sin(angle)  # начальные координаты примитива - нулевые, а конечные - одна из angle_points на окружности радиуса radius
        for k_start in [0.0, 0.5, 1]:
            for k_end in [-1, -0.5, 0.0, 0.5, 1]:  # ещё добавляем начальную и конечную кривизну
                for theta_offset in theta_offsets:
                    start_state = State(x=0.0, y=0.0, theta=0.0, k=k_start)
                    theta_end = angle + theta_offset
                    theta_end = (theta_end + np.pi) % (2 * np.pi) - np.pi
                    goal_state = State(x=xf, y=yf, theta=theta_end, k=k_end)
                    experiments.append((start_state, goal_state))
    return experiments

def save_experiments(experiments, filename):
    with open(filename, "w") as f:
        for start, goal in experiments:
            f.write(f"{start.x} {start.y} {start.theta} {start.k} {goal.x} {goal.y} {goal.theta} {goal.k}\n")
    print(f"Сохранено {len(experiments)} сценариев в файл '{filename}'")

def load_experiments(filename):
    experiments = []
    with open(filename, "r") as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            start = State(x=parts[0], y=parts[1], theta=parts[2], k=parts[3])
            goal = State(x=parts[4], y=parts[5], theta=parts[6], k=parts[7])
            experiments.append((start, goal))
    return experiments



# --- Рабочая функция для одного потока ---

def run_single_test(args):
    """
    Выполняет один тест для обоих методов (базового и предлагаемого). Предназначена для вызова в параллельном потоке.
    """

    test_id, start, goal = args
    iters, lr, eps = 100, 0.1, 1e-2
    result_dict = {
        'id': test_id,
        'start_x': start.x, 'start_y': start.y, 'start_theta': start.theta, 'start_k': start.k,
        'goal_x': goal.x, 'goal_y': goal.y, 'goal_theta': goal.theta, 'goal_k': goal.k
    }

    # Baseline метод
    try:
        t_start = time.time()
        steps, traj = baseline_optimization_Newton(start, goal, iters=iters, lr=lr, eps=eps)
        t_end = time.time()
        result_dict['baseline_success'] = True
        result_dict['baseline_time'] = t_end - t_start
        result_dict['baseline_params'] = f"{traj.a},{traj.b},{traj.c},{traj.length}"
        result_dict['baseline_steps'] = steps
    except:  # Вырожденная матрица или другая ошибка
        result_dict['baseline_success'] = False
        result_dict['baseline_time'] = -1
        result_dict['baseline_params'] = "Error"
        result_dict['baseline_steps'] = -1

    # Proposed метод
    try:
        t_start = time.time()
        steps, traj = optimization_Newton(start, goal, iters=iters, lr=lr, eps=eps)
        t_end = time.time()
        result_dict['proposed_success'] = True
        result_dict['proposed_time'] = t_end - t_start
        result_dict['proposed_params'] = f"{traj.k1},{traj.k2},{traj.log_length}"
        result_dict['proposed_steps'] = steps
    except:
        result_dict['proposed_success'] = False
        result_dict['proposed_time'] = -1
        result_dict['proposed_params'] = "Error"
        result_dict['proposed_steps'] = -1
        
    return result_dict



# --- Основная логика скрипта ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Запуск экспериментов для сравнения методов генерации примитивов.")
    parser.add_argument('action', choices=['generate', 'run'], help="Действие: 'generate' для создания тестов, 'run' для их запуска.")
    parser.add_argument('--input', default='test_cases.txt', help="Файл с тестовыми сценариями.")
    parser.add_argument('--output', default='results.csv', help="Файл для сохранения детальных результатов (в формате CSV).")
    parser.add_argument('--workers', type=int, default=cpu_count(), help="Количество параллельных процессов для запуска.")
    
    args = parser.parse_args()

    if args.action == 'generate':
        experiments = generate_experiments()
        save_experiments(experiments, args.input)

    elif args.action == 'run':
        print(f"Загрузка тестов из '{args.input}'...")
        if not os.path.exists(args.input):
            print(f"Файл {args.input} не найден! Сначала сгенерируйте его: python run_experiment.py generate")
            sys.exit(1)
            
        experiments = load_experiments(args.input)
        # Добавляем ID к каждому тесту для удобства логирования
        tasks = [(i, start, goal) for i, (start, goal) in enumerate(experiments)]

        print(f"Запуск {len(tasks)} тестов на {args.workers} процессах...")
        
        all_results = []
        with Pool(processes=args.workers) as pool:
            # Используем tqdm для отображения прогресс-бара
            for result in tqdm(pool.imap_unordered(run_single_test, tasks), total=len(tasks)):
                all_results.append(result)

        # Сортируем результаты по ID на всякий случай
        all_results.sort(key=lambda x: x['id'])

        # Сохранение детальной статистики в CSV
        print(f"Сохранение результатов в '{args.output}'...")
        fieldnames = [
            'id', 'start_x', 'start_y', 'start_theta', 'start_k', 'goal_x', 'goal_y', 'goal_theta', 'goal_k',
            'baseline_success', 'baseline_time', 'baseline_params', 'baseline_steps',
            'proposed_success', 'proposed_time', 'proposed_params', 'proposed_steps'
        ]
        with open(args.output, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for res in all_results:
                writer.writerow(res)

        # Подведение итоговой статистики
        baseline_success = sum(1 for r in all_results if r['baseline_success'])
        proposed_success = sum(1 for r in all_results if r['proposed_success'])
        
        baseline_times = [r['baseline_time'] for r in all_results if r['baseline_success']]
        proposed_times = [r['proposed_time'] for r in all_results if r['proposed_success']]

        print("\n" + "="*40)
        print("ИТОГОВАЯ СТАТИСТИКА")
        print("="*40)

        print(f"\n--- Baseline-метод ---")
        print(f"Success Rate: {baseline_success / len(all_results) * 100:.2f}%")
        if baseline_times:
            print(f"Среднее время: {np.mean(baseline_times):.4f} сек.")
            print(f"Медианное время: {np.median(baseline_times):.4f} сек.")

        print(f"\n--- Предложенный метод ---")
        print(f"Success Rate: {proposed_success / len(all_results) * 100:.2f}%")
        if proposed_times:
            print(f"Среднее время: {np.mean(proposed_times):.4f} сек.")
            print(f"Медианное время: {np.median(proposed_times):.4f} сек.")
        print("\n" + "="*40)
