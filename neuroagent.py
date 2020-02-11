#! /usr/bin/python3

import numpy as np
import random as rnd
import functions
import pandas as pd

class NAgent:
  nnetfileName = "./data/garry_007.nn"

  user_id = None
  case_id = None
  url = None
  nnet = None
  rmsprop_cache = None
  grad_buffer = None

  prev_act = None
  prev_score = None
  prev_hash = None
  prev_weights_dict = None
  decay_rate = 0.95  # decay factor for RMSProp leaky sum of grad^2

  alpha = 0.1 # learning rate
  gamma = 0.9 # discount coef
  delta = 0.0001 # LR descent coef
  batch_size = 10
  help_degree = 0.9 # part of help of correct weights
  dropout = 0.5 # part of neurons in hidden layers to dropout

  xs, hs, h2s, errs, zs, rs = [], [], [], [], [], []
  a_xs, a_hs, a_h2s, a_zerrs = None, None, None, None

  episode = pd.DataFrame(columns= ['hash_s', 'act', 'reward', 'hash_news'])

  gamesQ = 0  # счетчик игр - нужен для реализации батча при обучении

  all_acts_dict = {
    "none": ["noAct", "noAct"], "take": ["noAct", "Take"],
    "go_forward": ["noAct", "Go"], "go_right": ["onRight", "Go"],
    "go_back": ["upSideDn", "Go"], "go_left": ["onLeft", "Go"],
    "shoot_forward": ["noAct", "Shoot"], "shoot_right": ["onRight", "Shoot"],
    "shoot_back": ["upSideDn", "Shoot"], "shoot_left": ["onLeft", "Shoot"]
  }

  all_acts_nums_dict = {
    "none": 0, "take": 1, "go_forward": 2,  "go_right": 3, "go_back": 4,
    "go_left": 5, "shoot_forward": 6, "shoot_right": 7, "shoot_back": 8,
    "shoot_left": 9
  }
  all_acts_list = [ "none", "take", "go_forward", "go_right", "go_back", "go_left",
                    "shoot_forward", "shoot_right", "shoot_back", "shoot_left"  ]

  colnames = ["hash", "take", "go_forward", "go_right", "go_back", "go_left", "shoot_forward", "shoot_right",
              "shoot_back", "shoot_left"]

  fin_codes = ['--- agent is dead ---', '---- time is over ---', '!!! agent is WINNER !!!']

# PUBLIC METHODS ==============================================================

  def playGame(self, map_num, alpha, gamma, batch_size=10, tid=0, hashid=0):
    self.alpha = alpha
    self.gamma = gamma 
    self.batch_size = batch_size
    self.help_degree = 0.5
    self.dropout = 0.5
    request_code = None # код завершения хода
    curr_score = None   # набранные очки

    # запрашиваем состояние начальной пещеры, выполняя пустой ход
    acts = self.all_acts_dict["none"]
    request = functions.connectToServer(self.user_id, self.case_id, map_num, acts, tid, hashid)

    if request != None:  # связь с сервером установлена, можно играть
      # распарсиваем ответ сервера
      request_error = request["error"]
      percept = request["text"]
      curr_score = percept['iagent']["score"]

      # инициализация переменных, фиксирующих предыдущее состояние и ход
      curr_hash = self.__getHash__(percept)
      self.prev_act = "none"
      self.prev_score = curr_score
      self.prev_hash = curr_hash

      # создание таблицы ходов для запоминания игры (эпизода)
      rec = {'hash_s': [curr_hash], 'act': ["none"], 'reward': [0], 'hash_news': [curr_hash]}
      self.episode = pd.DataFrame(columns= rec.keys())

      # начинаем игру
      while request_error == None:  # пока никакой ошибки нет (нет завершения игры)
        if request != None:
          ''' # выбираем для текущего состояния ход, если состояние новое, то добавляем его в базу
          политики (полезностей) +4 записи; корректируем кол-во новых полей в базе данных   '''
          curr_act = self.__chooseAct__(curr_hash)
          acts = self.all_acts_dict[curr_act]
          # запоминаем набранное до выбранного хода кол-во очков и хэш текущего состояния s, выбранное действие
          self.prev_score = curr_score
          self.prev_hash = curr_hash
          self.prev_act = curr_act

        # запрашиваем ответ от сервера: сообщаем серверу выбранный ход и получаем новое состояние s'
        request = functions.connectToServer(self.user_id, self.case_id, map_num, acts, tid, hashid)

        if request != None:
          # распарсиваем ответ сервера
          request_error = request["error"]
          percept = request["text"]
          curr_score = percept["iagent"]["score"]
          request_code = int(percept["code"])

          curr_hash = self.__getHash__(percept)

          # ----- дополнение таблицы ходов и списка бонусов информацией о новом ходе -----
          reward = curr_score - self.prev_score
          self.rs.append(reward)
          # ---------- эта таблица нужна только для контроля ----------------
          rec = {'hash_s': [self.prev_hash], 'act': [curr_act], 'reward': [reward], 'hash_news': [curr_hash]}
          step1 = pd.DataFrame(data= rec)
          self.episode = pd.concat([self.episode, step1])

          # обновление полезности последнего хода и обучение нейросети
          if request_code in [0, 1, 2]:  # игра завершилась, обучение агента
            self.gamesQ += 1  # счетчик игр
            self.episode.index = list(range(self.episode.shape[0]))
            self.__update_nnet__()
            print('------ Код завершения = ', self.fin_codes[request_code], ' --------')

        else:
          print("WARNING! Server was not responded.")

    return request_code, curr_score, self.gamesQ

  # PRIVATE METHODS =============================================================

  def __init__(self, sess_parms, nnet_parms):
    self.user_id = sess_parms[0]
    self.case_id = sess_parms[1]
    self.tid = sess_parms[2]
    self.hashid = sess_parms[3]
    self.__createNnet__(nnet_parms)


  # создать или загрузить нейросеть
  def __createNnet__(self, nnet_parms):
    self.nnet = functions.openNnet(nnet_parms)

  # сохранить нейросеть
  def __saveNnet__(self):
    functions.saveNnet(self.nnet, self.nnetfileName)

  def __getHash__(self, percept):
    """
    :param percept: полученное текущее восприятие ситуации
    :return: хэш, символьная строка, кодирующая ситуацию
    """
    is_monster_alive = str(int(percept["worldinfo"]["ismonsteralive"]))
    newcaveopenedQ = len(percept["iagent"]["knowCaves"])
    # !!!!!!!!!!! ----- ONLY FOR 4x4 ------- !!!!!!!!!!!!
    unknowncavesQ = 16 - newcaveopenedQ
    if unknowncavesQ > 2:
      unknowncave_count = '3'
    else:
      unknowncave_count = str(unknowncavesQ)
    arrow_count = str(int(percept["iagent"]["arrowcount"]))
    legs_count = str(int(percept["iagent"]["legscount"]))
    curr_cave = percept["currentcave"]
    curr_cave_state = str(int(curr_cave["isGold"])) + str(int(curr_cave["isWind"])) \
                      + str(int(curr_cave["isBones"]))  + str(int(curr_cave["isHole"]))
    front_cave_state = self.__getNearCaveState__(percept["perception"]["front_cave"])
    back_cave_state = self.__getNearCaveState__(percept["perception"]["behind_cave"])
    left_cave_state = self.__getNearCaveState__(percept["perception"]["left_cave"])
    right_cave_state = self.__getNearCaveState__(percept["perception"]["right_cave"])
    front_left_cave_state = self.__getNearCaveState__(percept["perception"]["front_left_cave"])
    front_right_cave_state = self.__getNearCaveState__(percept["perception"]["front_right_cave"])
    behind_left_cave_state = self.__getNearCaveState__(percept["perception"]["behind_left_cave"])
    behind_right_cave_state = self.__getNearCaveState__(percept["perception"]["behind_right_cave"])
    res = is_monster_alive + arrow_count + legs_count + unknowncave_count + curr_cave_state
    res = res + front_left_cave_state + front_cave_state + front_right_cave_state  + right_cave_state
    res = res + behind_right_cave_state + back_cave_state + behind_left_cave_state + left_cave_state

    return res

  # подправить искусственно веса, используя состояние пещеры
  # weights - np.array весов: weights = np.array(curr_weights_row[1:])
  #  hash - строка хеш-кода
  def __correctWeights__(self, weights, hash, min_w=0, max_w=1):
    actshift = {"go":1, "shoot":5}
    #dirshift = {"forward":0, "back":1, "left":2, "right":3}
    dirshift = {"forward":0, "right":1, "back":2, "left":3}
    #caveshifts = {"forward":7, "back":11, "left":15, "right":19}
    caveshifts = {"forward":12, "right":20, "back":28, "left":36}
    if hash[4] == '1':  # надо брать клад
      weights = np.ones(len(weights)) * min_w
      weights[0] = max_w
    else:
      weights[0] = min_w  # клада нет - брать не надо
      if (hash[6] == '0'):  # лучше не стрелять - если рядом нет монстра
        for ii in range(4):
          weights[actshift["shoot"] + ii] = min_w+0.01
      if (hash[0] == '0') or (hash[1] == '0'):  # не надо пытаться стрелять - монстр мертв или стрел нет
        for ii in range(4):
          weights[actshift["shoot"] + ii] = min_w

      for cavedir in ["forward", "right", "back", "left"]:
        if hash[caveshifts[cavedir]] == '2': # wall
          weights[actshift["go"]+dirshift[cavedir]] = min_w
          weights[actshift["shoot"] + dirshift[cavedir]] = min_w
        if ((hash[caveshifts[cavedir] + 3] == '1') and (hash[2] == '1')):  # не надо ходить в яму, если одна нога!!!
          weights[actshift["go"] + dirshift[cavedir]] = min_w

    return weights

  # получить состояние пещеры
  def __getNearCaveState__(self, cave):
    cave_state = "2222"
    if cave["isWall"] == 0:
      cave_state = "0222"
      if cave["isVisiable"] == 1:  # Обновлено!!! - считается, что пещера ВИДИМА!!
        cave_state = "1" + str(int(cave["isWind"])) + str(int(cave["isBones"])) + str(int(cave["isHole"]))
    return cave_state

  # получить случайное действие с вероятностью, зависящей от его веса
  def __getActionByWeight__(self, curr_weights_dict):
    acts = np.array(list(curr_weights_dict.keys()))
    weights = np.array(list(curr_weights_dict.values()), dtype=float)
    # исключаем из лотереи заведомо проигрышные ходы и снижаем вероятность выбора просто плохих ходов
    limit_weight = 0  # устанавливаем порог заведомо проигрышных ходов
    max_weight = np.max(weights)
    #if (max_weight <= limit_weight): limit_weight = max_weight - 10 # страхуем себя на случай безвыходной ситуации
    if (max_weight <= limit_weight): limit_weight = weights[weights.argsort()[-2]] # страхуем себя на случай безвыходной ситуации
    acts = acts[weights >= limit_weight]
    weights = weights[weights >= limit_weight]

    min_weight = np.min(weights)
    weights = weights - min_weight + 0.001
    weights_array = weights / np.sum(weights)

    curr_act = rnd.choices(population=list(acts), weights=weights_array)[0]
    acts = self.all_acts_dict[curr_act]

    return curr_act, acts

  # преобразуем символьный хэш в числовой входной вектор для нейросети
  def __hash2vec__(self, curr_hash):
    vec = np.zeros(len(curr_hash))
    split_str = list(curr_hash)
    split_str = (' '.join(list(split_str)))
    vec = [int(x) for x in split_str.split()]
    return vec

  # строит вектор one-hot в зависимости от выбранного действия (100000000 - для 'take')
  def __act_to_yvec__(self, curr_act):
    act_yvec = np.zeros(9)
    if curr_act == "take":
      act_yvec[0] = 1
    if curr_act == "go_forward":
      act_yvec[1] = 1
    if curr_act == "go_right":
      act_yvec[2] = 1
    if curr_act == "go_back":
      act_yvec[3] = 1
    if curr_act == "go_left":
      act_yvec[4] = 1
    if curr_act == "shoot_forward":
      act_yvec[5] = 1
    if curr_act == "shoot_right":
      act_yvec[6] = 1
    if curr_act == "shoot_back":
      act_yvec[7] = 1
    if curr_act == "shoot_left":
      act_yvec[8] = 1

    return act_yvec

  def __discount_rewards__(self, r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r) * 1.0
    running_add = 0.0
    for t in reversed(range(0, r.size)):
      running_add = running_add * gamma + r[t]
      discounted_r[t] = running_add

    return discounted_r

  # выбираем для текущего состояния c curr_hash ход,
  # если состояние новое, то добавляем его в базу политики (полезностей) +4
  def __chooseAct__(self, curr_hash):
    #colnames = ["hash", "take", "go_forward", "go_right", "go_back", "go_left", "shoot_forward", "shoot_right", "shoot_back", "shoot_left"]
    x = self.__hash2vec__(curr_hash)

    # расчет выхода нейросети
    ynet, h = functions.policy_forward(model= self.nnet, x=x)

    if rnd.random() < self.help_degree:
      # корректируем веса для очевидных случаев
      weights = list(self.__correctWeights__(np.array(ynet), curr_hash))
    else:
      weights = list(ynet)

    weights.insert(0, curr_hash)
    curr_weights_row = tuple(weights)

    curr_weights_dict = dict(zip(self.colnames, curr_weights_row))
    del curr_weights_dict["hash"]
    self.prev_weights_dict = curr_weights_dict

    curr_act, acts = self.__getActionByWeight__(curr_weights_dict)

    # ------------------ запоминаем состояние нейросети для будущего обучения ------------
    # record various intermediates (needed later for backprop)
    self.xs.append(x)  # gane state - input for nnet
    self.hs.append(h)  # hidden state
    #self.zs.append(z)  # observation
    yvec = self.__act_to_yvec__(curr_act)
    self.errs.append(yvec - ynet)  # error of the action that was taken to be take

    return curr_act

  # -----------------------------------------------------------------------------------------
  # -------------- обучение нейросети на результатах завершившейся игры ---------------------
  def __update_nnet__(self):
    """
    обновление полезности всех сделанных ходов после завершения одной игры
    :param self.episode: содержит результаты всех ходов в иде кортежа (hash_s, act, reward, hash_news)
    :param alpha: с параметром обучения
    :param gamma: и с параметром дисконта
    :return: обновление базы данных или Q-таблицы
    """
    model = self.nnet
    self.rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() }
    self.grad_buffer = { k : np.zeros_like(v) for k,v in model.items() }

    # stack together all inputs, hidden states, action errors, and rewards for this episode
    a_xs = np.vstack(self.xs)
    a_hs = np.vstack(self.hs)
    #a_zs = np.vstack(self.zs)
    a_errs = np.vstack(self.errs)
    a_rs = np.vstack(self.rs)

    # reset array memory for new game
    self.xs, self.hs, self.zs, self.errs, self.rs = [], [], [], [], []

    # рассчитываем дисконтированный бонус для каждого хода в игре
    #r = self.episode['reward']
    discounted_r = self.__discount_rewards__(a_rs, self.gamma)

    # нормируем дисконтированные веса
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    #discounted_r -= np.mean(discounted_r)
    discounted_r /= np.std(discounted_r)

    # взвешивание ошибок с учетом нормированной дисконтированной полезности
    #a_errs *= discounted_r

    # расчет ошибок на входе в функцию активации (учет производной функции активации)
    #f_act_val = functions.sigmoid(a_zs)
    #a_zerrs = a_errs * f_act_val * (1 - f_act_val)
    a_zerrs = a_errs * discounted_r

    if self.gamesQ % self.batch_size == 1: # начинаем накапливать информацию о новом пакете игр
      self.a_xs = a_xs.copy()
      self.a_hs = a_hs.copy()
      self.a_zerrs = a_zerrs.copy()

    else:
      self.a_xs = np.vstack((self.a_xs, a_xs))
      self.a_hs = np.vstack((self.a_hs, a_hs))
      self.a_zerrs = np.vstack((self.a_zerrs, a_zerrs))

    # расчет суммарного антиградиента функции ошибки для весов
    gradW = functions.policy_backward(self.nnet, a_xs, a_hs, a_zerrs)
    self.grad_buffer = gradW
    
    # update buffers that add up gradients over a batch
    for k in self.nnet: self.grad_buffer[k] += gradW[k]  # accumulate grad over batch

    if self.gamesQ % self.batch_size == 0: # корректировка весов - метод rmsprop
      for k, v in self.nnet.items():
        for k,v in gradW.items():
          g_big = self.grad_buffer[k]
          i_mass = 0
          j_mass = 0
          if k == "W1":
              g = np.zeros((1024,40))
              while i_mass<1024:
                while j_mass<40:
                  g[i_mass][j_mass] = g_big[i_mass][j_mass]
                  j_mass += 1
                i_mass += 1
                j_mass = 0
          if k == "W2":
              g = np.zeros((1024,9))
              while i_mass<1024:
                    g[i_mass] = g_big[i_mass]
                    i_mass += 1
              g = np.transpose(g)
          self.rmsprop_cache[k] = self.decay_rate * self.rmsprop_cache[k] + (1 - self.decay_rate) * g ** 2
          self.nnet[k] += self.alpha * g / (np.sqrt(self.rmsprop_cache[k]) + 1e-5)
          self.grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer
