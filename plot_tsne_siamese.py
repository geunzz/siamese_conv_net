
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from keras import backend as K
from sklearn.manifold import TSNE
from dataset_generation import data_generator
import numpy as np
import os
import shutil

flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('MODEL_DIR', 'model.40-0.01.h5', 'Directory to put the model.(.h5)')
flags.DEFINE_string('TEST_DATA_DIR', 'C:/projects/dataset/thermal_image/80_60_test_dataset/', 
                    'Directory to put the test data.')
flags.DEFINE_string('REP_DATA_DIR', 'C:/projects/dataset/thermal_image/80_60_rep_dataset/', 
                    'Directory to put the representative data.')

colors =  'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'navy', 'darkviolet', 'pink', 'hotpink', 'orangered', 'coral',\
          'yellowgreen', 'lightgray', 'maroon', 'salmon', 'gold', 'slategrey', 'cadetblue'
labels =  'face01', 'face02', 'face03', 'face04', 'face05', 'face06', 'face07', 'face08', 'face09', 'face10',\
          'face11', 'face12', 'face13', 'face14', 'face15', 'face16', 'face17', 'face18', 'face19', 'face20'

def contrastive_loss(y_true, y_pred):
    y_true=tf.dtypes.cast(y_true, tf.float64)
    y_pred=tf.dtypes.cast(y_pred, tf.float64)
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

datagen_test = data_generator(DATASET_PATH = FLAGS.TEST_DATA_DIR, shuffle_sel=False)
data_class_set, data_array, label, data_name = datagen_test.data_label_set_gen()
#test_prob = 1 means all data will use for test.
_, x_test, _, y_test, _, z_test = datagen_test.train_val_split(data_array, label, data_name, test_prob=1)

print(x_test.shape)
x_test = x_test.astype('float32') #test data
y_test = y_test.astype('float32') #label

x_test /= 255

input_shape = x_test.shape[1:]
input_shape = (80, 60, 3)

loaded_model = load_model(FLAGS.MODEL_DIR, custom_objects={'contrastive_loss': contrastive_loss})
loaded_model.summary()

new_model = Model(loaded_model.inputs, loaded_model.layers[-7].output)

new_model.set_weights(loaded_model.get_weights())
new_model.summary()
# new_model.save("no_softargmax_" + str(fname))

X=[]
for i in range(0,len(x_test)):
    img1 = np.array(x_test[i])[np.newaxis, :, :, :]
    zero_img = np.zeros(input_shape)
    img2 = np.array(zero_img)[np.newaxis, :, :, :]
    subs = new_model([img1, img2])
    subs_val = tf.reshape(subs, [-1])
    subs_val_numpy = subs_val.numpy()
    X.append(subs_val_numpy)
    if i%100 == 0:
        print('Computing the ', str(i), '-th feature vector... (', str(i), '/', str(len(x_test)-1), ')')

# Fit and transform with a TSNE
tsne = TSNE(n_components=2, random_state=0)

# Project the data in 2D
X_2d = tsne.fit_transform(X)

# Visualize the data
target_ids = range(len(y_test))

from matplotlib import pyplot as plt

plt.figure(figsize=(5, 5))

for i, c, label in zip(target_ids, colors, labels):
    #plt.scatter(X_2d[y_test == i, 0], X_2d[y_test == i, 1], c=c, label=label, s=1 , alpha=0.5)
    plt.scatter(X_2d[y_test == i, 0], X_2d[y_test == i, 1], c=c, s=1 , alpha=0.3)

num_class = len(labels)
coordinate_class = [X_2d, y_test]
rep_set= []
for j in range(num_class):
    index_dist_tot = []
    for k in range(len(X_2d)):
        dist_tot = 0
        if j == y_test[k]:
            for l in range(len(X_2d)):
                if j == y_test[l]:
                    dist = np.sqrt(np.square(coordinate_class[0][k][0] - coordinate_class[0][l][0])
                            +np.square(coordinate_class[0][k][1] - coordinate_class[0][l][1]))
                    dist_tot = dist + dist_tot
        if dist_tot != 0:
            index_dist_tot.append([k, dist_tot]) #j번째 class에서 k좌표와 다른 좌표간의 총 거리
    min_k_index = index_dist_tot[np.argmin(index_dist_tot, axis = 0)[1]][0] #j번째 class의 minimum distance를 만족하는 X_2d의 index 값
    rep_set.append([j, min_k_index, X_2d[min_k_index], z_test[min_k_index]])

    #rep set에 포함되는 data가 결정되면 rep dataset 위치로 파일을 옮겨줌
    for dirs in os.listdir(FLAGS.TEST_DATA_DIR):
        before_path = os.path.join(FLAGS.TEST_DATA_DIR, dirs, z_test[min_k_index])
        if os.path.isfile(before_path):
            move_path = os.path.join(FLAGS.REP_DATA_DIR, dirs)
            if os.path.isdir(move_path):
                shutil.move(before_path, move_path)
            else:
                os.mkdir(move_path)
                shutil.move(before_path, move_path)

    plt.scatter(X_2d[min_k_index, 0], X_2d[min_k_index, 1], c='black', s=10 ,marker='o')
print(rep_set)

plt.legend()
plt.show()

