import os
import time
import argparse
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

if 'get_ipython' in locals():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

import matplotlib.pyplot as plt
plt.style.use('dark_background')


# ================ ГИПЕРПАРАМЕТРЫ =======================

# цвета для бокса: "пусто" - зелёный, "возможно занято" - жёлтый, "занято" - красный
COLORS = {
    'empty': (0, 255, 0),
    'maybe': (0, 255, 255),
    'occupied': (0, 0, 255),
}
# отображать ли FPS при отрисовке бокса
SHOW_FPS = False
# отображать ли статус занятости столика при отрисовке бокса
SHOW_LABEL = False
# если человек обнаружен более ENTER_TIME_THRESHOLD секунд подряд - столик занят
ENTER_TIME_THRESHOLD = 10
# если человека нет более EXIT_TIME_THRESHOLD секунд - столик свободен
EXIT_TIME_THRESHOLD = 10
# делать детекцию не каждого подряд идущего фрейма а каждого DETECT_EVERY_N_FRAMES
# для ускорения процесса
DETECT_EVERY_N_FRAMES = 2
# размер батча
BATCH_SIZE = 128
# размер видео для сохранения, установить None чтобы сохранять в исходном разрешении
NEW_WIDTH = 800
NEW_HEIGHT = 600
# NEW_WIDTH = None
# NEW_HEIGHT = None


# ====================== ДЕТЕКЦИЯ =========================

# парсинг аргументов командной строки
parser = argparse.ArgumentParser(description='Детекция занятости столиков по видео')
parser.add_argument('--video', type=str, required=False, help='Путь к видеофайлу')
args = parser.parse_args()

# путь до видео
video_path = Path(args.video)
print(video_path)
# video_path = Path(VIDEO_EXAMPLE_PATH_2)

# инициализация модели
model = YOLO('yolo11n.pt')
# флаг что столик занят
is_occupied = False
# список для последующего создания датафрейма
df_list = []
# получение текущего времени для расчета FPS
prev_time = time.monotonic()
# счётчик последовательных секунд с обнаруженным человеком в кадре
t_detect = 0.0
# счётчик последовательных секунд с отсуствием человека в кадре
t_empty = 0.0
# счетчик фреймов
frame_number = 1
# есть ли человек в кадре
person_in_frame = 0
# стартовый флаг обнаружения человека в кадре для последующего обновления
person_detected = False
# списки для сбора фреймов
frames_buffer = []
full_frames_buffer = []
frame_numbers_buffer = []
# список с флагами - нужно ли детектить текущий кадр или проупстить для ускорения
detect_flags_buffer = []

# задать фиксированный регион столика
# roi_box = x1, y1, x2, y2 = 250, 250, 1000, 1000
roi_box = None

# захват видео для чтения
cap_read = cv2.VideoCapture(video_path)
if not cap_read.isOpened():
    raise ValueError(f'Ошибка чтения видео {video_path}')

# параметры видео
frame_width = int(cap_read.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap_read.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames_fps = int(cap_read.get(cv2.CAP_PROP_FPS))
frames_total = int(cap_read.get(cv2.CAP_PROP_FRAME_COUNT))

# если размеры видео меньше заданного размера то не делать ресайз
if NEW_WIDTH:
    NEW_WIDTH = min(frame_width, NEW_WIDTH)
if NEW_HEIGHT:
    NEW_HEIGHT = min(frame_height, NEW_HEIGHT)

# захват видео для записи
result_video_path_tmp = Path('output_tmp.mp4')
cap_write = cv2.VideoWriter(
    filename=result_video_path_tmp,
    fourcc=cv2.VideoWriter_fourcc(*'avc1'),  # mp4v, avc1, h264, XVID, DIVX
    fps=frames_fps,
    frameSize=(NEW_WIDTH, NEW_HEIGHT),
    # качество видео от 0 до 100, работает не на всех кодеках
    # params=[cv2.VIDEOWRITER_PROP_QUALITY, 80],
)

# определить регион столика (если не задан вручную)
if roi_box is None:
    # считывание первого кадра для выбора области
    ret, first_frame = cap_read.read()
    print('Выберите область столика и нажмите ENTER или SPACE')
    cv2.namedWindow('Select a screen area', cv2.WINDOW_NORMAL)
    # уменьшение размера окна т.к на видео с большим разрешением оно может не влезть в экран
    cv2.resizeWindow('Select a screen area', 1280, 720)
    roi_box = cv2.selectROI('Select a screen area', first_frame, False, False)
    cv2.destroyWindow('Select a screen area')
    if roi_box == (0, 0, 0, 0):
        print('Область не выбрана, используем весь кадр')
        roi_box = (0, 0, frame_width, frame_height)
    # перевод ROI в формат (x1, y1, x2, y2)
    x1, y1, w, h = roi_box
    roi_box = x1, y1, x2, y2 = (x1, y1, x1 + w, y1 + h)
    # вернуть захват видео к первому кадру
    cap_read.set(cv2.CAP_PROP_POS_FRAMES, 0)

print(f'''
Старт детекции видео
Длительность видео: {frames_total / frames_fps:.2f} сек
Разрешение: {frame_width}x{frame_height}
Всего кадров: {frames_total}
FPS: {frames_fps}
''')

# прогресс бар
pbar = tqdm(total=frames_total, desc='Детекция видео')

# старт чтения кадров из видео
while True:
    # чтение кадра и флага того что кадр прочитан успешно и видео не закончилось
    ret, cv2_image = cap_read.read()

    # временное ограничение для отладки
    # if frame_number > 200:
        # break
        
    # батчевая детекция - если достигнут размер батча или видео закончилось
    if len(frames_buffer) >= BATCH_SIZE or not ret:
        if frames_buffer:
            frames_to_detect = [f for f, d in zip(frames_buffer, detect_flags_buffer) if d]
            # детекция людей на батче кадров
            batch_results = model.predict(
                source=frames_to_detect,
                conf=0.25,
                iou=0.7,
                save=False,
                verbose=False,
                classes=[0],
            )
            result_idx = 0
            # итерация по каждому кадру и результату детекции
            for idx in range(len(frames_buffer)):
                # кадр в исходном разрешении
                full_frame = full_frames_buffer[idx]
                # номер кадра
                frame_num = frame_numbers_buffer[idx]
                # текущая секунда видео
                frame_duration = 1 / frames_fps
                # флаг того что текущий кадр был подан в модель
                is_detect = detect_flags_buffer[idx]
                # есть ли результат детекции этого кадра или он пропускается
                if is_detect:
                    detections = batch_results[result_idx]
                    result_idx += 1
                    # есть ли люди в кадре
                    person_detected = len(detections.boxes) > 0
                    # уверенность в том что на кадре есть человек (первый, с наибольшей уверенностью)
                    if person_detected:
                        # если кадр пропускается то сохраняется предыдущий результат
                        # что может быть не совсем корректно но для прототипа оставил так
                        conf = detections.boxes[0].conf.item()
                # расчет времени нахождения и времени отсутствия человека в кадре
                if person_detected:
                    t_detect += frame_duration
                    t_empty = 0.0
                else:
                    t_empty += frame_duration
                    t_detect = 0.0
                    conf = None
                # если человек в кадре более ENTER_TIME_THRESHOLD то установить статус что стол занят
                if not is_occupied and t_detect >= ENTER_TIME_THRESHOLD:
                    is_occupied = True
                    t_empty = 0.0
                # если человека нет в кадре более EXIT_TIME_THRESHOLD то установить статус что стол свободен
                elif is_occupied and t_empty >= EXIT_TIME_THRESHOLD:
                    is_occupied = False
                    t_detect = 0.0
                # сбор результатов детекции в список для датафрейма
                df_list.append([frame_num, conf, person_detected, is_occupied])
                # цвет бокса по состоянию
                if is_occupied:
                    color = COLORS['occupied']
                    status_text = 'OCCUPIED'
                elif t_detect > 0:
                    color = COLORS['maybe']
                    status_text = 'MAYBE'
                else:
                    color = COLORS['empty']
                    status_text = 'EMPTY'
                # отрисовка ROI на полном кадре
                cv2.rectangle(
                    img=full_frame,
                    pt1=(x1, y1),
                    pt2=(x2, y2),
                    color=color,
                    thickness=3,
                )
                # отрисовка текста состояния занятости столика на видео
                if SHOW_LABEL:
                    cv2.putText(
                        img=full_frame,
                        text=f'Status: {status_text}',
                        # координаты текста - левый верхний угол, можно изменить
                        org=(30, 40),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(255, 255, 255),
                        thickness=2,
                        lineType=cv2.LINE_AA,
                    )
                # отрисовка FPS на видео
                if SHOW_FPS:
                    current_time = time.monotonic()
                    fps = 1 / (current_time - prev_time)
                    prev_time = current_time
                    cv2.putText(
                        img=full_frame,
                        text=f'FPS: {fps:.2f}',
                        # координаты текста - левый верхний угол, можно изменить
                        org=(30, 80),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(255, 255, 255),
                        thickness=2,
                        lineType=cv2.LINE_AA,
                    )
                # ресайз кадра + запись кадра в новое видео
                resized_frame = cv2.resize(full_frame, (NEW_WIDTH, NEW_HEIGHT))
                cap_write.write(resized_frame)

        # очистка буфера
        frames_buffer = []
        full_frames_buffer = []
        frame_numbers_buffer = []
        
        # завершение цикла если видео закончилось
        if not ret:
            break
    
    # кроп кадра столика
    cropped_image = cv2_image[y1:y2, x1:x2]
    # добавление кропнутого и полного кадров и номера кадра в буфер
    frames_buffer.append(cropped_image)
    full_frames_buffer.append(cv2_image)
    frame_numbers_buffer.append(frame_number)
    # добавление флага - нужно ли дедектить этот кадр или пропустить
    detect_flags_buffer.append(frame_number % DETECT_EVERY_N_FRAMES == 0)
    # обновление номера фрейма и прогресс бара
    frame_number += 1
    pbar.update()
    
# закрыть объекты для записи и чтения видео
cap_write.release()
cap_read.release()
cv2.destroyAllWindows()

# переименование видео чтобы не было иероглифов на нелатинских буквах
result_video_path = Path(f'{video_path.stem}_result.mp4')
result_video_path_tmp.replace(result_video_path)
print(f'Видео сохранено в {result_video_path}')

# создание датафрейма
df = pd.DataFrame(df_list, columns=['frame_num', 'conf', 'person_detected', 'is_occupied'])
# вставка столбца с секундами видео
df.insert(1, 'frame_sec', df['frame_num'] / frames_fps)
# флаг что столик свооден
df['is_empty'] = ~df['is_occupied']
# пропуск первых кадров на случай если человек уже сидел в кадре и чтобы не было ложного прихода
df = df[df['frame_sec'] > ENTER_TIME_THRESHOLD + 1]
# сохранение датафрейма в csv
result_csv_path = f'{video_path.stem}_results.csv'
df.to_csv(result_csv_path, index=False)
print(f'Датафрейм сохранён в {result_csv_path}')


# ============= РАСЧЕТ ИНТЕРВАЛОВ, СТАТИСТИК И ГРАФИКИ ===========

# определение событий: уход (occupied -> empty) и приход (empty -> occupied)
df['event'] = None
df.loc[(df['is_empty'].shift(1)==False) & (df['is_empty']==True), 'event'] = 'leave'
df.loc[(df['is_empty'].shift(1)==True) & (df['is_empty']==False), 'event'] = 'arrive'
# расчет времени
arrive_times = df[df['event'] == 'arrive']['frame_sec'].values
leave_times = df[df['event'] == 'leave']['frame_sec'].values
# события прихода и ухода
n_leaves = (df['event']=='leave').sum()
n_arrives = (df['event']=='arrive').sum()
# расчет интервалов занятости стола
intervals = []
for dep in leave_times:
    next_arr = arrive_times[arrive_times > dep]
    if len(next_arr) > 0:
        intervals.append(next_arr[0] - dep)
intervals = np.array(intervals)

# графики
plt.figure(figsize=(12, 6))

# 1. состояние столика во времени
plt.subplot(2, 1, 1)
# plt.plot(df['frame_sec'], df['is_empty'], label='empty (1=пусто, 0=занят)')
plt.plot(df['frame_sec'], df['is_occupied'].astype(int), label='occupied (1=занят, 0=пусто)')
plt.title('Состояние столика во времени')
plt.xlabel('Время (сек)')
plt.ylabel('Состояние')
plt.legend()

# 2. распределение времени ожидания
plt.subplot(2, 1, 2)
plt.title('Распределение времени ожидания')
plt.xlabel('Секунды')
plt.ylabel('Количество')
if len(intervals) > 0:
    plt.hist(intervals, bins=15)
else:
    plt.text(0.5, 0.5, 'Недостаточно данных для расчета (нет пар уход->приход)', ha='center')

plt.tight_layout()
# сохранение графика в файл
plot_path = f'{video_path.stem}_analysis.png'
plt.savefig(plot_path)
print(f'График сохранён в "{plot_path}"')
# plt.show()


# ==================== ОТЧЕТ ============================

# статистики (только если есть интервалы)
if len(intervals) > 0:
    mean_interval = np.mean(intervals)
    median_interval = np.median(intervals)
    std_interval = np.std(intervals)
    min_interval = np.min(intervals)
    max_interval = np.max(intervals)
    n_intervals = len(intervals)
else:
    mean_interval = median_interval = std_interval = min_interval = max_interval = None
    n_intervals = 0

# оценка доли пустого времени
total_time = df['frame_sec'].max()
empty_ratio = df['is_empty'].mean() if len(df) > 0 else 0
empty_time = empty_ratio * total_time

# отчет
report = f'''
# ОТЧЕТ ПО АНАЛИТИКЕ
Дата: {datetime.now()}
Видео: {video_path.stem}
Общая продолжительность: {total_time:.2f} сек
Разрешение: {frame_width}x{frame_height}
Всего кадров: {frames_total}
FPS: {frames_fps}

## Статистика занятости
Время свободного места: {empty_time:.2f} сек
Коэффициент свободного места: {empty_ratio:.2%}
Количество событий ухода: {n_leaves}
Количество событий прихода: {n_arrives}

## Статистика времени ожидания
'''

if n_intervals > 0:
    report += f'''
Среднее время ожидания: {mean_interval:.2f} сек
Медиана времени ожидания: {median_interval:.2f} сек
Стандартное отклонение: {std_interval:.2f} сек
Минимальное время ожидания: {min_interval:.2f} сек
Максимальное время ожидания: {max_interval:.2f} сек
Количество выборок: {n_intervals}
'''
else:
    report += 'Недостаточно данных для расчета (нет пар уход->приход)'
print(report)

# сохранение отчета в txt
result_txt_path = f'{video_path.stem}_results.txt'
Path(result_txt_path).write_text(report, encoding='utf-8')
print(f'Отчёт сохранён в {result_txt_path}')
