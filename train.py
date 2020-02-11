#! /usr/bin/python3

#import functions
import neuroagent
import manager
import time
import random
import urllib3

# гиперпараметры ==============================================================

# case_id   описание
# 1         L-1(Поиск клада. Известный Лабиринт. 2 обвала. 1 монстр.)
# 6         L-2(Поиск клада. Неизвестный Лабиринт. Без обвалов. Без монстра.)
# 7         L-3-1(Поиск клада. Неизвестный Лабиринт. 1 обвал. Без монстра)
# 4         L-4-1 (Поиск клада. Неизвестный Лабиринт. 2 обвала. Без монстра. )
# 2         L-5 (Поиск клада. Неизвестный Лабиринт. 2 обвала. 1 монстр.)
# 11        L-3-2 (Поиск клада. Неизвестный Лабиринт. без обвалов, 1 монстр)
# 12        L-4-2 (Поиск клада. Неизвестный Лабиринт. 1 обвал. 1 монстр.)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
case_id = 2 # id кейса задачи
user_id = 30761 # id пользователя
tid = 0
hashid = 0
sess_parms = [user_id, case_id, tid, hashid]

data_path = "./data/" # путь до папки с данными
agent_name = "garry_007" # название агента
#url = "https://mooped.net/local/its/game/agentaction/" # url сервера

# параметры нейросети
inp_N = 40  # кол-во чисел, описывающих состояние игры
hidden_N = 1024   # произвольно подбираемое число
out_N = 9       # кол-во возможных действий агента
nnet_filename = data_path + agent_name + '.nn'
nnet_parms = [nnet_filename, inp_N, hidden_N, out_N]

# создать или загрузить агента, которого будут тренировать
nagent = neuroagent.NAgent(sess_parms, nnet_parms)

# параметры обучения нейросети
alpha = 0.09 # фактор обучения
gamma = 0.9 # фактор дисконтирования
delta = 0.00001# коэф-т уменьшения alpha
batch_size = 10

#map_numbers = list(range(1,251))
#map_numbers  = random.sample(map_numbers1, 50)
map_numbers = [1]

attempts_per_map = 100 # количество попыток на каждую карту

# запуск обучения ===================================================
map_parms = [map_numbers, attempts_per_map]
learn_parms = [alpha, gamma, delta, batch_size]

# создать менеджера и запустить проверку / обучение
obs = manager.Manager(sess_parms, map_parms, learn_parms)

start = time.time()
obs.train(nagent, 1)
end = time.time()
print("Время обучения составило (мин) ", round((end - start)/60))
