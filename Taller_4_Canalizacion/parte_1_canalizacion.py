#%%
"""
# Taller parte 1 
### Parte 1:
## Autor:Wilgen Correa
### Fecha; 30-04-2021
### Objetivo: Un cuaderno, escrito de sus manos, repitiendo y si desean modificando el primer cuaderno.
"""

# Importar las librerias
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np 
import pandas as pd
import seaborn as sb
import tensorflow as tf
from tensorflow import keras
from tensorflow.estimator import LinearClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
#
print(tf.__version__)


