from __future__ import absolute_import
from __future__ import print_function

import tensorflow.keras as keras
import tensorflow as tf

import numpy as np
import random
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from keras import backend as K
from dataset_generation import data_generator
from create_net_functional import create_base_net


num_classes = 20
epochs = 40

def euclid_dis(vects):
  x,y = vects
  sum_square = K.sum(K.square(x-y), axis=1, keepdims=True)
  return K.sqrt(K.maximum(sum_square, K.epsilon()))
 
def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)
 
def contrastive_loss(y_true, y_pred):
    y_true=tf.dtypes.cast(y_true, tf.float64)
    y_pred=tf.dtypes.cast(y_pred, tf.float64)
    margin = 0.6
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
    
def create_pairs(x, digit_indices):
  pairs = []
  labels = []
   
  n=min([len(digit_indices[d]) for d in range(num_classes)]) -1
   
  for d in range(num_classes):
    for i in range(n):
      z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
      pairs += [[x[z1], x[z2]]]
      inc = random.randrange(1, num_classes)
      dn = (d + inc) % num_classes
      z1, z2 = digit_indices[d][i], digit_indices[dn][i]
      pairs += [[x[z1], x[z2]]]
      labels += [1,0]
  return np.array(pairs), np.array(labels)

def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.2
    return np.mean(pred == y_true)

def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.2, y_true.dtype)))

datagen = data_generator(DATASET_PATH = 'C:/PATH/TO/THE/IMAGE/', shuffle_sel=True)
data_class_set, data_array, label, data_name = datagen.data_label_set_gen()
x_train, x_test, y_train, y_test, z_train, z_test = datagen.train_val_split(data_array, label, data_name, test_prob=0.2)

print(x_train.shape)

#define the image size
input_shape = (60, 80, 3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

#should have to be normalize input value -> if not it will converge same accuracy and loss value
#also keras 2.2.4 has related problem(converging same value things.). so write down below commend
#pip install -U --force-reinstall --no-dependencies git+https://github.com/datumbox/keras@fork/keras2.2.4
x_train /= 255
x_test /= 255
 
input_shape = x_train.shape[1:]

# create training+test positive and negative pairs
digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
tr_pairs, tr_y = create_pairs(x_train, digit_indices)
 
digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
te_pairs, te_y = create_pairs(x_test, digit_indices)
 
# network definition
base_network = create_base_net(input_shape)
	
input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

processed_a = base_network(input_a)
processed_b = base_network(input_b)
 
sum_square = K.sum(K.square(processed_a-processed_b), axis=1, keepdims=True)
distance = K.sqrt(K.maximum(sum_square, K.epsilon()))
model = Model([input_a, input_b], distance)

my_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5', save_freq='epoch', save_weights_only=False),
    tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_images=True),
]

model.compile(loss=contrastive_loss, optimizer=Adam(lr=0.0008), metrics=[accuracy])
model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y, batch_size=5, epochs=epochs, 
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y), callbacks=my_callbacks)

# compute final accuracy on training and test sets
y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(tr_y, y_pred)
y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(te_y, y_pred)
 
print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
