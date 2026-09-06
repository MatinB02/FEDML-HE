# -*- coding: utf-8 -*-
# Models/DNN.py

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class DNN(nn.Module, ABC):
    name = "DNN"
    studentType = ""
    learning_rate_global = 0
    learning_rate_local = 0
    keepProbTeacher = 0
    keepProbStudent = 0
    learningRate_DecayEnable = False
    inputNormalization = False

    def __init__(self):
        super(DNN, self).__init__()
