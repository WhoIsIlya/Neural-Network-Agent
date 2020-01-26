#! /usr/bin/python3

import requests as req
import numpy as np
import pickle
import os

# работа с сервером ===========================================================

# ===================  установка связи с сервером  =====================
def connectToServer(user_id, case_id, map_num, acts = ['noAct', 'noAct'], tid=0, hashid=0):
  http_proxy = "http://10.0.0.4:3128"
  https_proxy = "https://10.0.0.4:3128"

  proxyDict = {
    "http": http_proxy,
    "https": https_proxy
  }

  if tid != 0:  # если соревнования
      url = 'https://mooped.net/local/its/tournament/agentaction/'
      resp = req.get(url,
                   params={
                     'tid': tid,
                     'userid': user_id,
                     'passive': acts[0], 'active': acts[1],
                     'checksituations': 1},
                   #proxies=proxyDict,
                   verify=False)
  elif hashid != 0:  # если контрольное тестирование
      url = 'https://mooped.net/local/its/tests/agentaction/'
      resp = req.get(url,
                     params={
                       'hash': hashid,
                       'passive': acts[0], 'active': acts[1],
                       'checksituations': 1},
                     #proxies=proxyDict,
                     verify=False)
  else:   # если нет соревнований и контрольного тестирования, то тестируем на карте mapnum
      url = "https://mooped.net/local/its/game/agentaction/"
      resp = req.get(url,
                   params={
                     'caseid': case_id,
                     'userid': user_id,
                     'mapnum': map_num,
                     'passive': acts[0], 'active': acts[1],
                     'checksituations': 1},
                   #proxies=proxyDict,
                   verify=False)
  # , proxies = proxyDict

  json = None
  if resp.status_code == 200:
    #print("----- соединение установлено -----")
    json = resp.json()
  return json

# ===================  работа с нейросетью ========================
# логистическая функция активации
def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

# создать и инициализровать нейросеть с заданными характеристиками для игры
# inp_N, out_N - кол-во нейронов во входном и выходном слоях
# hidden_nums - список кол-ва нейронов в скрытых слоях
# выход: объект класса, хранящий всю информацию для расчета нейросети
def createMyNnet(inp_N, hidden_nums, out_N):
    model = {}
    model['W1'] = np.random.randn(hidden_nums[0], inp_N) / np.sqrt(inp_N)
    if len(hidden_nums) > 1:
        for lnum in range(1,len(hidden_nums)):
            wname = 'W' + str(lnum+1)
            model[wname] = np.random.randn(hidden_nums[lnum], hidden_nums[lnum-1]) / np.sqrt(hidden_nums[lnum-1])
    model['Wn'] = np.random.randn(out_N, hidden_nums[-1]) / np.sqrt(hidden_nums[-1])
    return model

# считать нейросеть из файла или создать нейросеть, если нет такого файла
def openNnet(nnet_parms):
    file_name = nnet_parms[0]
    try:
        model = readNnet(file_name)
        print('model загружена из ', file_name)
    except:
        #inp_N = nnet_parms[1]
        inp_N = nnet_parms[1]
        hidden_N = nnet_parms[2]
        out_N = nnet_parms[3]
        model = createNnet(inp_N, hidden_N, out_N)

    return model

# создать и инициализровать нейросеть с заданными характеристиками для игры
# inp_N, hidden_N out_N - кол-во нейронов во входном, скрытом и выходном слоях
# выход: объект класса, хранящий всю информацию для расчета нейросети
def createNnet(inp_N, hidden_N, out_N):
    model = {}
    model['W1'] = np.random.randn(hidden_N, inp_N) / np.sqrt(inp_N)
    model['W2'] = np.random.randn(out_N, hidden_N) / np.sqrt(hidden_N)
    return model

# считать нейросеть из файла
def readNnet(file_name):
    model = pickle.load(open(file_name, 'rb'))
    return model

# считать нейросеть из файла
def saveNnet(model, file_name):
    pickle.dump(model, open(file_name, 'wb'))
    pass

def discount_rewards(r, gamma):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

def policy_forward(model, x):
  h = np.dot(model['W1'], x)
  h[h<0] = 0 # ReLU nonlinearity
  logp = np.dot(model['W2'], h)
  p = sigmoid(logp)
  return p, h # return probability of taking action 2, and hidden state

def policy_backward(model, epx, eph, epdlogp):
  """ backward pass. (eph is array of intermediate hidden states) """
  dW2 = np.dot(eph.T, epdlogp).ravel()
  dh = np.outer(epdlogp, model['W2'])
  dh[eph <= 0] = 0 # backpro prelu
  dW1 = np.dot(dh.T, epx)
  return {'W1':dW1, 'W2':dW2}

