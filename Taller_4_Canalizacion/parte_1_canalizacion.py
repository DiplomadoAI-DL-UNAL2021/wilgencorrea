#%% [markdown]
"""
# Taller parte 1 
### Parte 1:
## Autor:Wilgen Correa
### Fecha; 30-04-2021
### Objetivo: Un cuaderno, escrito de sus manos acerca del manejo de tensores.
"""

# Importar las librerías

import tensorflow as tf
import numpy as np
#
print(tf.__version__)


#%% [markdown]
"""
 ### Declaración de una constante y su visualización
"""
t0 = tf.constant(4)
print(t0)

t1 = tf.constant([1, 3, 4])
print(t1)

t23 = tf.constant([[1, 2, 3],[4, 5, 6]])
print(t23)
#%% [markdown]
"""
### Declaración de constantes a partir de una matriz
"""
t235 = tf.constant([[[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]],
                    [[16,17,18,19,20],[21,22,23,24,25],[26,27,28,29,30]]])
print(t235)
#%% [markdown]
"""
### Se  utiliza numpy 
"""
print(t235.numpy())
print(t235.numpy().shape)
#%% [markdown]
"""
### Algebra mínima de tensores.
"""
a = tf.constant([[1,2],[3,4]])
b = tf.constant([[5,6],[7,8]])

print(a,"\n")
print(b,"\n")
print("Suma de tensores a+b:  \n",tf.add(a,b),"\n")
print("Multiplicación de tensores a*b:  \n",tf.multiply(a,b),"\n")
print(tf.matmul(a,b),"\n")

#%% [markdown]
"""
### Otras forma de realizar operaciones aritméticas con los tensores 
"""
print("Suma de tensores a+b:  \n",a+b,"\n")
print("Multiplicación de tensores a*b:  \n", a*b, "\n")
print(a@b, "\n")

#%% [markdown]
"""
## Funcion de reducción
"""
c = tf.constant([[4.0, 5.0],[10.0,1.0]])

print(c,"\n")
print("Máximo elemento: ",tf.reduce_max(c),"\n")
print("Mínimo elemento: ",tf.reduce_min(c),"\n")
print("Media del tensor:  ",tf.reduce_mean(c),"\n")
print("Retorna la posición en el tensor del elemento mas grande: ",tf.argmax(c),"\n")
print("Uso de función de reducción: ",tf.nn.softmax(c),"\n")
print(tf.reduce_sum(tf.nn.softmax(c),axis=1),"\n")
#%% [markdown]
"""
## Tipo, forma y dimensión
"""

t = tf.constant([
  [[0, 1, 2, 3, 4],
   [5, 6, 7, 8, 9]],
  [[10, 11, 12, 13, 14],
   [15, 16, 17, 18, 19]],
  [[20, 21, 22, 23, 24],
   [25, 26, 27, 28, 29]],])
print("Tipo de tensor: \n",t.dtype,"\n")
print("Descripciones del tensor: \n",t.shape,"\n")
print("Dimensiones del tensor: \n",t.ndim,"\n")
print("Descripción t.shape[2]: \n",t.shape[2],"\n")
print("Tamano del tensor: \n",tf.size(t.numpy()),"\n")

#%% [markdown]
"""
# Indexación
Indexación en un solo eje
Los índices empiezan en cero.
Los índices negativos cuentan hacia atrás desde el final.
Los dos puntos (:) se utilizan para los cortes. inicio:final:salto
"""

t1 =  tf.constant([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
print("Primero: ", t1[0])
print("Segundo: ", t1[1])
print("Ultimo: ", t1[-1].numpy())

#%% [markdown]
"""
## Extracción de parte de un tensor: slices¶
"""
print('Todo', t1[:].numpy())
print('Antes la posición 4 ', t1[:4].numpy())
print('Desde la posición 4 hasta el final', t1[4:].numpy())
print('Desde la posición 2 hasta anterior a 7', t1[4:7].numpy())
print('Todos los elementos en posición par ', t1[::2].numpy())
print('Invertido todo el orden', t1[::-1].numpy())

#%% [markdown]
"""
# Indexación multi-eje
"""
t23 = tf.constant([[1, 2, 3],[4, 5, 6]])
print(t23.numpy())

# Algunas indexaciones 
print('Posición 1,1 = ',t23[1,1].numpy())
print('Segunda fila: ', t23[1,:].numpy())
print('Segunda columna: ', t23[:,1].numpy())
print('Última columna: ', t23[:,-1].numpy())
print('Primer elemento de la última columna: ', t23[0,-1].numpy())
print('Saltarse la primera columna: \n', t23[:,1:].numpy()) 

#%% [markdown]
"""
# Ejemplo de cubo de un tensor
"""
t = tf.constant([
  [[0, 1, 2, 3, 4],
   [5, 6, 7, 8, 9]],
  [[10, 11, 12, 13, 14],
   [15, 16, 17, 18, 19]],
  [[20, 21, 22, 23, 24],
   [25, 26, 27, 28, 29]],]) 
print(t.numpy())

#%% [markdown]
"""
### Extracciones de info del tensor
"""
print('Extrae la primera capa\n', t[0,:,:])
print('Extrae la segunda capa\n', t[1])

#%% [markdown]
"""
### Manipular formas y remodelando el tensor
"""
x = tf.constant([[1],[2],[3],[4]])
print(x.shape)
print(x.shape.as_list())
print("Remodelando el tensor \n")
reshaped = tf.reshape(x,[2,2])
print(reshaped.numpy())
print(x.numpy())

#%% [markdown]
"""
### Aplanar un tensor
"""
print("t actual: \n", t.numpy())
flat = tf.reshape(t, [-1])
print("\n t aplanado: \n", flat.numpy())

#%% [markdown]
"""
### Mirar un tensor como una lista
"""
print(t.shape.as_list())
print("tf.reshape(t, [2*3,5]).....:\n")
print(tf.reshape(t, [2*3,5]))
print("tf.reshape(t, [2,3*5]).....:\n")
print(tf.reshape(t, [2,3*5]))

#%% [markdown]
"""
### Otras impresiones con reshape
"""
print("Tensor original:: ",t.shape.as_list())
print(tf.reshape(t, [-1,5]))
print(tf.reshape(t, [3,-1]))
print(tf.reshape(t, [3,2,5])) #Es el mismo original

#%% [markdown]
"""
### Definición Conversión de tipos. Cast
"""
f64_tensor = tf.constant([2.0, 4.0, 6.0], dtype = tf.float64)
print(f64_tensor)

f16_tensor = tf.cast(f64_tensor,dtype= tf.float16)
print(f16_tensor)

u8_tensor = tf.cast(f16_tensor, dtype = tf.uint8)
print(u8_tensor)

#%% [markdown]
"""
### Radiofusión (broadcasting)
La radiodifusión es un concepto tomado de la función equivalente en NumPy . En resumen, bajo ciertas condiciones, los tensores más pequeños se "estiran" 
automáticamente para adaptarse a tensores más grandes cuando se ejecutan operaciones combinadas en ellos.

El caso más simple y común es cuando intenta multiplicar o agregar un tensor a un escalar. En ese caso, el escalar se transmite para que tenga la misma forma que el otro argumento.
"""
x = tf.constant([1, 2 ,3])
y = tf.constant(2)
z = tf.constant([2, 2 ,2])

# el mismo resultado
print(tf.multiply(x,2))
print(x*y)
print(x*z)

#%% [markdown]
"""
### Remodelado
"""
x = tf.reshape(x, [3,1])
y = tf.range(1,5)
print("Tensor X: ",x, "\n")
print("Tensor Y: ",y, "\n")

#%%
"""
### Multiplicación
Del mismo modo, los ejes con longitud 1 se pueden estirar para que coincidan con los otros argumentos. 
Ambos argumentos se pueden estirar en el mismo cálculo.

En este caso, una matriz de 3x1 se multiplica por elementos por una matriz de 1x4 para producir 
una matriz de 3x4. Observe que el 1 inicial es opcional: la forma de y es [4]. En matemáticas 
esta multiplicación se conoce como producto externo.
"""
print(tf.multiply(x,y))

#%% [markdown]
"""
### tf.broadcast_to
"""
print(tf.broadcast_to([1,2,3], [3,3]))

#%% [markdown]
"""
### tf.convert_to_tensor
La mayoría de las operaciones, como tf.matmul y tf.reshape toman argumentos de la clase tf.Tensor. 
Sin embargo, notará que en el caso anterior, se aceptan objetos de Python con forma de tensores.

La mayoría, pero no todas, las operaciones llaman a tf.convert_to_tensor con argumentos no tensoriales. 
Existe un registro de conversiones, y la mayoría de las clases de objetos como ndarray , TensorShape ,
de Python, y tf.Variable se convertirán todas automáticamente.
"""
#%% [markdown]
"""
### Tensores irregulares (ragged tensors) 
Un tensor con números variables de elementos a lo largo de algún eje se llama "irregular".
 Utilice tf.ragged.RaggedTensor para datos irregulares.

Por ejemplo, esto no se puede representar como un tensor regular:
"""
ragged_list = [
    [0, 1, 2, 3],
    [4, 5],
    [6, 7, 8],
    [9]]

try:
    tensor = tf.constant(ragged_list)
except Exception as e:
     print(f"{type(e).__name__}: {e}")
     
# En su lugar, cree un tf.RaggedTensor usando tf.ragged.constant :
ragged_t = tf.ragged.constant(ragged_list)
print(ragged_t.shape)


#%% [markdown]
"""
Tensores de strings
"""
st = tf.constant("Este tensor string")
print(st)

st = tf.constant(["Este tensor string",
                 "Cadena 2",
                 "Cadena 3",
                 "🥳"])
print(st)
print(tf.strings.split(st))

#%% [markdown]
"""
### string to number
"""
st = tf.constant("1 10 10.4")
print(tf.strings.to_number(tf.strings.split(st, " ")))

#%% [markdown]
"""
### Tensores dispersos. SparseTensor
Se crea en tensor de 3 filas por 4 columnas [3,4], en la celda [0,1] esta el elemento 1
en la celda [1,2] esta el elemento 2 
en el ejemplo segundo:
Se crea en tensor de 3 filas por 10 columnas [3,10], en la celda [0,3] esta el elemento 10
en la celda [2,4] esta el elemento 20.     
"""
#  tensor disperso Ejemplo 1
sparse_tensor = tf.sparse.SparseTensor(indices = [[0,1], [1,2]],
                                       values = [1,2],
                                       dense_shape
                                        =[3,4])
print(sparse_tensor, " Ejemplo 1\n")

# convierte a tensor denso
print(tf.sparse.to_dense(sparse_tensor))


#  tensor disperso Ejemplo 2
st1 = tf.SparseTensor(indices=[[0, 3], [2, 4]],
                      values=[10, 20],
                      dense_shape=[3, 10])

print(st1, " Ejemplo 2\n")

# convierte a tensor denso
print(tf.sparse.to_dense(st1))
