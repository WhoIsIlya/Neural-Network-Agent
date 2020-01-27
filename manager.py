#! /usr/bin/python3

import numpy as np
import functions
import random

class Manager:

  user_id = 0 # id пользователя
  case_id = 0 # id задачи
  tid = 0  # id tournament
  hashid = 0  # id control game

  map_numbers = [] # номера карт
  attempts_count = 0 # количество попыток на карте

  total_games_count = 0 # общее количество игр

  alpha = 0 # фактор обучения
  gamma = 0 # фактор дисконтирования
  delta = 0 # коэф-т уменьшения alpha
  batchnum = 10 # размер пакета обучения

  loss = 0  # ошибка выбора действия агентом

# PUBLIC METHODS ==============================================================

  # Запуск тренировки
  def train(self, qa, iter_count = 1):
    print("Start training.")
    print("Iteration count: ", iter_count, ", attempt count: ", self.attempts_count, ", map numbers: ", self.map_numbers)
    self.field_count = 0  # инициация счетчика заполненных полей таблицы
    lost_map_list = []

    for it in range(1, iter_count + 1):
      iter_games_count = 0
      iter_wins_count = 0
      iter_score_count = 0
      random.shuffle(self.map_numbers)
      for num in range(len(self.map_numbers)):
        for attempt_num in range(1, self.attempts_count + 1):
          # запустить для агента одну игру на карте <map_num> с параметрами обучения alpha и gamma
          # или игру-кейс для контрольного тестирования (тогда должно совпадать кол-во карт и кол-во хэшей!)
          if self.hashid == 0:
            hash_id = 0
            map_num = self.map_numbers[num]
          else:
            hash_id = self.hashid[num]
            map_num = 0
          #print(self.hashid)
          code, score, marked_fieldsQ = qa.playGame(map_num, self.alpha, self.gamma, self.batch_size, self.tid, hashid=hash_id)
          self.field_count = marked_fieldsQ

          if (code == None):
            print("Connection failed for: map = {0}, attempt = {1}".format(map_num, attempt_num))
          else:
            is_win = 0
            if code == 2:
              is_win = 1
              iter_score_count += score
            else:
              lost_map_list.append(map_num)

            iter_games_count += 1
            iter_wins_count += is_win

          #print("************************************")
          if (self.tid == 0):
            print("map_num: ", num+1, ", attempt: ", attempt_num, ", new_fields = ", self.field_count, ", wins=", iter_wins_count, ", alpha = {:8.6f}".format(self.alpha))
          else:
            print("Iteration: ", it, ", tid: ", self.tid, ", new_fields = ", self.field_count, ", wins=", iter_wins_count, ", alpha = {:8.6f}".format(self.alpha))
          
          self.alpha -= self.delta
          if self.alpha < 0.01: self.alpha = 0.01

          
          #self.db_conn.commit()  # записать изменения в базу

      self.total_games_count += iter_games_count
      iter_win_rate = (iter_wins_count * 100) / iter_games_count
      iter_score_rate = iter_score_count / iter_games_count

      print("Games: ", iter_games_count, ", Win rate: {:6.2f}, Score rate: {:6.2f}".format(iter_win_rate, iter_score_rate))
      print("Lost Game Maps:", lost_map_list)

      # сохранение нейросети
      qa.__saveNnet__()

      #values = (self.total_games_count, iter_win_rate, iter_score_rate)

# PRIVATE METHODS =============================================================

  # sess_parms = [user_id, case_id, data_files_path, url]
  # map_parms = [from_map_num, to_map_num, attempts_per_map]
  # learn_parms = [alpha, gamma]
  def __init__(self, sess_parms, map_parms, learn_parms):
    self.user_id = sess_parms[0]
    self.case_id = sess_parms[1]
    self.tid = sess_parms[2]
    self.hashid = sess_parms[3]
    self.map_numbers = map_parms[0]
    self.attempts_count = map_parms[1]
    self.alpha = learn_parms[0]
    self.gamma = learn_parms[1]
    self.delta = learn_parms[2]
    self.batch_size = learn_parms[3]
