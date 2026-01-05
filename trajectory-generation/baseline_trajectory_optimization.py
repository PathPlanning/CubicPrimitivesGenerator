"""
Аналогичный файл, но использующий baseline-версию генерации примитива (через первую параметризацию).
"""

import numpy as np
import sys
sys.path.append("../common/")
from PRIM_structs import *



def baseline_get_residual(traj: ShortTrajectory) -> np.ndarray:
    final = traj.final_state()
    return np.array([traj.goal.x - final.x,
                     traj.goal.y - final.y,
                     traj.goal.theta - final.theta,
                     traj.goal.k - final.k])  # в первой (базовой) параметризации приходится включать в невязку и кривизну, ведь
                                              # тут кривизна не является параметром, который можно зафиксировать... 



def baseline_calc_Jacobian_matrix(traj: ShortTrajectory, params: np.ndarray, dk: float = 0.001, dS: float = 0.001) -> np.ndarray:
    """
    Аналогично, но матрица 4 на 4, ведь 4 параметра (a, b, c, length) и 4 компоненты в функции-невязке.
    """

    a, b, c, length = params
    traj = ShortTrajectory(traj.start, traj.goal)
    
    dF_p = baseline_get_residual(traj.set_coef_params(a+dk, b, c, length))
    dF_m = baseline_get_residual(traj.set_coef_params(a-dk, b, c, length))
    grad_a = (dF_p - dF_m) / (2 * dk)

    dF_p = baseline_get_residual(traj.set_coef_params(a, b+dk, c, length))
    dF_m = baseline_get_residual(traj.set_coef_params(a, b-dk, c, length))
    grad_b = (dF_p - dF_m) / (2 * dk)
    
    dF_p = baseline_get_residual(traj.set_coef_params(a, b, c+dk, length))
    dF_m = baseline_get_residual(traj.set_coef_params(a, b, c-dk, length))
    grad_c = (dF_p - dF_m) / (2 * dk)

    dF_p = baseline_get_residual(traj.set_coef_params(a, b, c+dk, length+dS))
    dF_m = baseline_get_residual(traj.set_coef_params(a, b, c+dk, length-dS))
    grad_S = (dF_p - dF_m) / (2 * dS)

    return np.hstack((grad_a.reshape(-1, 1), grad_b.reshape(-1, 1), grad_c.reshape(-1, 1), grad_S.reshape(-1, 1)))  # собираем все частные производные в матрицу Якоби 



def baseline_optimization_Newton(start: State, goal: State, iters: int = 2000, eps: float = 1e-2, lr: float = 0.03, redraw_trajectory = None) -> ShortTrajectory:
    """
    Аналогично предыдущей функции, но использует базовую параметризацию.
    """
    
    traj =  ShortTrajectory(start, goal)
    params = np.array([0.0, 0.0, 0.0, 1.0])  # начальные параметры траектории (первая параметризация): a, b, c, length

    steps = 0
    for i in range(iters):
        steps += 1
        curr_diff = baseline_get_residual(traj.set_coef_params(*params))
        J = baseline_calc_Jacobian_matrix(traj, params)
        params -= lr * np.linalg.inv(J) @ curr_diff

        if np.sum(curr_diff ** 2) ** 0.5 <= eps:  # если норма невязки достаточно мала, можно останавливать поиск
            break
        if redraw_trajectory:
            redraw_trajectory(traj, i)

    else:
        print("Невозможно найти траекторию! Метод Ньютона не сошёлся!")
        return None

    return steps, traj.set_coef_params(*params)  # возвращаем найденную траекторию (с найденными параметрами)
