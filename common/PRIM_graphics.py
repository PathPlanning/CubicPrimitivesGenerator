"""
В данном файле перечислены основные функции для отрисовки результатов генерации примитивов движения.
"""

import numpy as np
import matplotlib.axes as axes
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from IPython.display import display, clear_output
from PIL import Image
from typing import Optional
from PRIM_structs import *



def plot_arrow(x: float, y: float, theta: float, length: float = 0.8, width: float = 0.3, 
               fc: str = "r", ec: str = "k", ax: Optional[axes.Axes] = None) -> None:
    """
    Функция для рисования стрелки в определённом направлении.
    
        x, y, theta: координаты начала и угол направления стрелки
        ax: задаёт matplotlib.axes, где рисовать стрелку (если None, то просто в plt рисуется)
        line, width: размеры стрелки
        fc, ec: цвет стрелки
    """
    
    board = plt if (ax is None) else ax  # определяем, где рисовать стрелку
    board.arrow(x, y, length * np.cos(theta), length * np.sin(theta),
                fc=fc, ec=ec, head_width=width, head_length=width)
    


def create_live_visualizer(start_state: State, target_state: State,
                           xlim=(-3, 9), ylim=(-4, 4), figsize=(6, 4), dpi=150,
                           frequency=10, make_gif=False):
    """
    Создает "контекст" для рисования и возвращает функцию для онлайн-перерисовывания траектории.
    
    Параметры:
        start_state, target_state: объекты State (для отрисовки старта и цели),
        xlim, ylim: фиксированные границы графика по каждой из осей,
        figsize, dpi: настройки качества картинки (её размер и плотность пикселей на дюйм),
        frequency: как часто отрисовывать траекторию (по умолчанию -- раз в 10 итераций метода Ньютона),
        make_gif: если True, то помимо функции онлайн-перерисовывания возвращается функция save_gif, которую
                  можно вызвать после окончания визуализации, чтобы сохранить её в виде gif.
    """
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)  # однократно создаем фигуру и оси
    
    ax.set_aspect('equal')  # настраиваем "жесткую" сетку и границы (как в примере выше)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))  # cетка через 1 единицу
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax.grid(visible=True, which='both', color='grey', alpha=0.4)
    
    plot_arrow(start_state.x, start_state.y, start_state.theta, fc='b', ec='k', ax=ax)     # рисуем старт и цель
    plot_arrow(target_state.x, target_state.y, target_state.theta, fc='g', ec='k', ax=ax)  # (старт -- синим, цель -- зелёным)
    
    line, = ax.plot([], [], 'r-', lw=2, label='current-trajectory')  # создаем объект линии, который будем обновлять (пока пустой)
                                                                     # ('r-' означает красная сплошная линия, lw=2 - толщина)
    head_marker, = ax.plot([], [], 'ro', markersize=5)  # опционально: создаем точку для конца текущей траектории (= текущий final_state)
    
    ax.set_title("Optimization Process")
    
    # В Jupyter Notebook это заставляет отрисовать пустой график сразу
    # display(fig) # Можно раскомментировать, если график не появляется сам
    
    frames = []  # Список кадров (будет наполняться только если make_gif=True)


    # Определяем функцию обновления (принимает на вход текущую траекторию для отрисовки и номер текущей итерации для отрисовки):
    def update(trajectory, iter):
        if iter % frequency != 0:  # отрисовываем раз в frequency итераций
            return
        
        xs = trajectory.sample_x()  # получаем новые координаты точек траектории
        ys = trajectory.sample_y()

        line.set_data(xs, ys)       # ОБНОВЛЯЕМ данные в существующих объектах (очень быстро)
        if len(xs) > 0:             # обновляем точку конца траектории (опционально)
            head_marker.set_data([xs[-1]], [ys[-1]])

        # Перерисовываем холст (вроде нужно):
        plt.pause(0.001) 
        
        # Для Jupyter Notebook, чтобы была плавная анимация в одной ячейке:
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        # Если вдруг анимация не идет, в старых версиях Jupyter иногда нужны следующие 2 строки:
        clear_output(wait=True)
        display(fig)

        # --- Сохраняем отрисованную картинку в список фреймов ---
        if make_gif:
            rgba_buffer = fig.canvas.buffer_rgba()  # Конвертируем буфер графика в массив numpy, затем в картинку PIL
            image_array = np.asarray(rgba_buffer)
            im = Image.fromarray(image_array)
            if im.mode == 'RGBA':  # Конвертируем в RGB (убираем прозрачность, чтобы меньше весило)
                im = im.convert('RGB')
            frames.append(im)
        # --------------------------------------------------------
    

    # --- Функция сохранения кадров frames в итоговую gif (необходимо вызвать после окончания визуализации) ---
    # (drop_last говорит, сколько последних кадров отбросить... может быть полезно, так как в конце оптимизации обычно мало что меняется)
    def save_gif(filename="optimization.gif", fps=10, drop_last = 0):
        if len(frames) == 0:
            print("No frames captured to save.")
            return
        durations = int(1000 / fps)
        end_index = len(frames) - drop_last  # Вычисляем срез кадров
        if end_index <= 1:
            end_index = len(frames)  # Защита, если drop_last слишком большой
        print(f"Saving GIF... Total frames: {len(frames)}, Used: {end_index}")

        frames[0].save(
            filename, 
            save_all=True, 
            append_images=frames[1:end_index],  # Берем срез, отбрасывая хвост
            optimize=True, 
            duration=durations,  # <-- задержка между кадрами (число или список) в мс
            loop=0
        )
        print(f"GIF saved to {filename}")


    if make_gif:
        return update, save_gif
    else:
        return update



def show_trajectory(traj: ShortTrajectory, col: str = 'r', arrow: bool = True, ax: Optional[axes.Axes] = None) -> None:
    """
    Функция для рисования траектории.
    
        traj: траектория в виде объекта ShortTrajectory,
        col: цвет кривой,
        arrow: рисовать ли стрелку, изображающую конец траектории (её координаты и угол соответствуют финальному состоянию траектории),
        ax: задаёт matplotlib.axes, где рисовать траекторию.
    """
    
    board = plt if (ax is None) else ax  # определяем, где рисовать
    
    xc = traj.sample_x()  # получаем набор точек кривой, изобразив которые, получим вид траектории
    yc = traj.sample_y()
    board.plot(xc, yc, "-"+col)
    
    if arrow:
        final = traj.goal  # стрелка рисуется в целевом состоянии (куда должна вести траектория при правильных параметрах)
        plot_arrow(final.x, final.y, final.theta, fc=col, ax=ax)



def redraw_trajectory(prim: ShortTrajectory, col: str = 'r') -> None:
    """
    Функция, перерисовывающая траекторию (полезна для демонстрации изменений траектории
    в процессе подбора параметров). Более примитивный аналог функции update из create_live_visualizer.
    
        traj: текущая траектория как объект ShortTrajectory,
        col: цвет траектории.
    """
    
    xc = prim.sample_x()  # точки траектории 
    yc = prim.sample_y()      
    x, y, theta = prim.goal.x, prim.goal.y, prim.goal.theta  # финальные координаты и угол направления (к которым траектория стремится)
    
    clear_output(wait=True)  # очищаем предыдущий вывод
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis("equal")
    ax.grid(True)
    
    ax.plot(xc, yc, "-"+col)  # рисуем текущую траекторию
    ax.arrow(x, y, 1 * np.cos(theta), 1 * np.sin(theta),
             fc=col, ec="k", head_width=0.5, head_length=0.5)
    
    plt.pause(0.01)
    fig.canvas.draw()
    plt.show()
