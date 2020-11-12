# importação da bibliotecas
import numpy as np
import xlsxwriter
import matplotlib.pyplot as pyplot
import itertools
import seaborn as sns
import time
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras import models
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import *
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model
from collections import defaultdict
import itertools
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import matplotlib.cm as cm
from vis.utils import utils
from keras import activations
from vis.utils import utils
from vis.visualization import visualize_saliency, overlay, visualize_cam
from vis.utils import utils
from keras import activations
import os



counter=1
# matplotlib inline
np.random.seed(2017)
def pandas_classification_report(y_true, y_pred):
    metrics_summary = precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred)

    avg = list(precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred,
            average='weighted'))

    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index)

    support = class_report_df.loc['support']
    total = support.sum()
    avg[-1] = total

    class_report_df['avg / total'] = avg

    return class_report_df.T
def report2dict(cr):
    # Parse rows
    tmp = list()
    for row in cr.split("\n"):
        parsed_row = [x for x in row.split("  ") if len(x) > 0]
        if len(parsed_row) > 0:
            tmp.append(parsed_row)

    # Store in dictionary
    measures = tmp[0]

    D_class_data = defaultdict(dict)
    for row in tmp[1:]:
        class_label = row[0]
        for j, m in enumerate(measures):
            D_class_data[class_label][m.strip()] = float(row[j + 1].strip())
    return D_class_data
# Constantes da Simulação ou caso
row = 0
col = 0
#counter = 3
nclasses = 7
img_rows = 224
img_cols = 224
epochs = 2
train_batch_size = 10
valid_batch_size = 8
test_batch_size = 13
num_of_train_samples = 1333
num_of_valid_samples = 351
num_of_test_samples = 169
class_names = ['krishtattoo', 'mattbeckerich', 'arzabe','pabloortiz', 'mikeridenball','manuraccoon','dynoz']
    # preprocessamento das imagens
train_path = 'C:/Users/Adm/Desktop/Código tcc/dataset7class/train'
valid_path = 'C:/Users/Adm/Desktop/Código tcc/dataset7class/valid'
test_path = 'C:/Users/Adm/Desktop/Código tcc/dataset7class/test'
figpath = 'C:/Users/Adm/Desktop/Código tcc/code7classes/'
arqui= 'linear_'
#train_datagen = ImageDataGenerator(rescale=1. / 255,
#                                   rotation_range=40,
#                                   width_shift_range=0.2,
#                                   height_shift_range=0.2,
#                                   shear_range=0.2,
#                                   zoom_range=0.2,
#                                   horizontal_flip=True,
#                                   fill_mode='nearest')
train_datagen = ImageDataGenerator()
train_batches = train_datagen.flow_from_directory(
    train_path, target_size=(img_rows, img_cols), classes=class_names, batch_size=train_batch_size)
valid_batches = ImageDataGenerator().flow_from_directory(
    valid_path, target_size=(img_rows, img_cols), classes=class_names, batch_size=valid_batch_size)
test_batches = ImageDataGenerator().flow_from_directory(
    test_path, target_size=(img_rows, img_cols), classes=class_names, batch_size=test_batch_size, shuffle = False)
    # imgs, labels = next(train_batches)
    # plots(imgs, titles=labels)

vgg16model = keras.applications.vgg16.VGG16()
    # vgg16model.summary()

vgg16model.layers.pop()

model = Sequential()
for layer in vgg16model.layers:
    model.add(layer)

    # for layer in model.layers[:-6]:
for layer in model.layers:
    layer.trainable = False
#model.add(Dense(75, activation='softmax', name="classificador75"))
    #model.add(Dense(4096, activation='softmax', name="classificador4096"))
    #model.add(Dense(2048, activation='softmax', name="classificador2048"))
    #model.add(Dense(1024, activation='softmax', name="classificador1024"))
model.add(Dense(nclasses, activation='softmax', name="classificador"))
model.summary()

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit_generator(train_batches, steps_per_epoch=num_of_train_samples // train_batch_size,
                              validation_data=valid_batches, validation_steps=num_of_valid_samples // valid_batch_size, epochs=epochs)
layer_idx = utils.find_layer_idx(model, 'classificador')

imgDIR=('C:/Users/Adm/Desktop/Código tcc/dataset7class/test/mattbeckerich/')
directory = os.fsencode(imgDIR)
for file in os.listdir(directory):
     filename = os.fsdecode(file)
     if filename.endswith(".jpg"):
         img1 = utils.load_img(imgDIR+filename, target_size=(224, 224))
         #model.layers[layer_idx].activation = activations.softmax
         #model = utils.apply_modifications(model)
         plt.figure(1)
         plt.subplot(2,4,1)
         plt.imshow(img1)
         plt.title(filename)
         for filterU in range(0,7):
             grads=visualize_saliency(model, layer_idx, filter_indices= filterU, seed_input= img1)
             plt.subplot(2,4,2+filterU)
             plt.title(str(class_names[filterU]))
             plt.imshow(grads, cmap='jet')
             figManager = plt.get_current_fig_manager()
             figManager.full_screen_toggle()
         plt.savefig(figpath+filename[0:-4]+'saliencia'+'.png')
         plt.figure(figsize=(14, 10))
         plt.subplot(2,4,1)
         plt.imshow(img1)
         plt.title(filename)
         for filterU in range(0,7):
             grads=visualize_saliency(model, layer_idx, filter_indices= filterU, seed_input= img1)
             plt.subplot(2,4,2+filterU)
             plt.title(str(class_names[filterU]))
             jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
             plt.imshow(overlay(jet_heatmap, img1))
             figManager = plt.get_current_fig_manager()
             figManager.full_screen_toggle()
         #plt.show()
         plt.savefig(figpath+filename[0:-4]+'sobreposicao'+'.png')
         continue
     else:
         continue
