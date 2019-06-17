import tensorflow as tf
import os
import numpy
import random
import time
import numpy as np
from corpus import Corpus
from models.model import Model
from models.vbpr import VBPR

class HBPR(VBPR):
    def __init__(self, session, corpus, sampler, k, k2, factor_reg, bias_reg):
        # 现在载入brand和price信息，将其与visual features结合起来
        corpus.load_heuristics()

        VBPR.__init__(self, session, corpus, sampler, k, k2, factor_reg, bias_reg)