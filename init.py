import numpy as np
import cv2
import os
import pickle
import random


def unpickle(file):
  fo = open(file, 'rb')
  dict = pickle.load(fo, encoding='bytes')
  fo.close()
  return dict

batch_names = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
data = []
labels = []

for i in range(0, len(batch_names)):
  t = unpickle(batch_names[i])
  d = t[b'data']
  labels.extend(t[b'labels'])
  batch, _ = t[b'data'].shape
  for i in range(0, batch):
    p = d[i].reshape([3, 1024]).transpose().reshape([32, 32, 3])
    data.append(p)

def genxy(count = None):
  if not count:
    return np.stack(data).astype(np.float32), np.stack(labels).astype(np.float32)
  x = []
  y = []
  for _ in range(0, count):
    pick = random.randint(0, len(data) - 1)
    x.append(data[pick])
    tmp = np.ndarray(dtype=np.float32, shape=(10))
    for i in range(0, 10):
      if labels[pick] == i:
        tmp[i] = 0.8
      else:
        tmp[i] = 0.2
    y.append(tmp)
  return np.stack(x).astype(np.float32) / 256.0, np.stack(y).astype(np.float32)