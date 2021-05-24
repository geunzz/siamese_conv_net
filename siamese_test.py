import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from keras import backend as K
import numpy as np
from dataset_generation import data_generator

flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('DISTANCE_THRESHOLD', 0.2, 'class judgement threshold distance.')
flags.DEFINE_string('TEST_DATA_DIR', 'C:/projects/dataset/thermal_image/80_60_test_dataset/',
                'Directory to put the test data.')
flags.DEFINE_string('REP_DATA_DIR', 'C:/projects/dataset/thermal_image/80_60_rep_dataset/',
                'Directory to put the representative data.')
flags.DEFINE_string('MODEL_DIR', 'model.40-0.01.h5',
                'Directory to put the model.(.h5)')
flags.DEFINE_boolean('TEST_ANOMALY', False, 'test anomaly classes = True, know classes = False.')

#test_prob = 1 means all data will use for test.
datagen_test = data_generator(DATASET_PATH = FLAGS.TEST_DATA_DIR, shuffle_sel=False)
data_class_set, data_array, label, data_name = datagen_test.data_label_set_gen()
_, x_test, _, y_test, _, z_test = datagen_test.train_val_split(data_array, label, data_name, test_prob=1)

datagen_rep = data_generator(DATASET_PATH = FLAGS.REP_DATA_DIR, shuffle_sel=False)
data_class_set_rep, data_array_rep, label_rep, data_name_rep = datagen_rep.data_label_set_gen()
_, x_rep, _, y_rep, _, z_rep = datagen_rep.train_val_split(data_array_rep, label_rep, data_name_rep, test_prob=1)
 
def contrastive_loss(y_true, y_pred):
    y_true=tf.dtypes.cast(y_true, tf.float64)
    y_pred=tf.dtypes.cast(y_pred, tf.float64)
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

print(x_test.shape)
x_test = x_test.astype('float32')
y_test = y_test.astype('float32')
x_rep = x_rep.astype('float32')
y_rep = y_rep.astype('float32')

x_test /= 255
x_rep /= 255

input_shape = x_test.shape[1:]
input_shape = (80, 60, 3)

loaded_model = load_model(FLAGS.MODEL_DIR, custom_objects={'contrastive_loss': contrastive_loss})
loaded_model.summary()

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

right_answer = 0
total_answer = 0
for i in range(0,len(x_test)):
    dist_cmp = []
    dist_arr = []
    avg_rep_dist = []
    img1 = np.array(x_test[i])[np.newaxis, :, :, :] # Tensor compatible
    for j in range(0,len(x_rep)):
        img2 = np.array(x_rep[j])[np.newaxis, :, :, :]
        dist = loaded_model([img1, img2])
        dist_val = tf.reshape(dist, [-1])
        #change dist_val from tensor to numpy
        dist_val_numpy = dist_val.numpy()[-1]
        dist_cmp.append([dist_val_numpy, y_test[i], y_rep[j]])
        dist_arr.append(dist_val_numpy)

    dist_cmp = np.array(dist_cmp)
    for k in range(0,int(max(y_rep)+1)):
        rows = np.where(dist_cmp[:,2] == k)
        #save class id value(k) and average distance compare with representative images.
        avg_rep_dist.append([k, np.mean(dist_cmp[rows], axis=0)[0]])

    inference_class = [np.argmin(avg_rep_dist, axis=0)[1]][0]
    inference_class_dist = round(np.min(avg_rep_dist, axis=0)[1], 5)

    if FLAGS.TEST_ANOMALY:
        if inference_class_dist < FLAGS.DISTANCE_THRESHOLD:
            answer = 'X'
        else:
            answer = 'O'
    else:
        if int(y_test[i]) == int(inference_class) and inference_class_dist < FLAGS.DISTANCE_THRESHOLD:
            answer = 'O'
        else:
            answer = 'X'
    
    if answer == 'O':
        right_answer = right_answer + 1
        total_answer = total_answer + 1
    else:
        total_answer = total_answer + 1

    answer_accuracy = round(right_answer * 100 / total_answer, 2)

    print('name :', z_test[i], '  class :', inference_class, 
    '  distance :', inference_class_dist, '  answer :', answer ,'  accuracy :', answer_accuracy, '[%]')
