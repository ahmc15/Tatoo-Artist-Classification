# importação da bibliotecas
import numpy as np
import keras
import xlsxwriter
from keras.models import Sequential
from keras import models
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import *
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from collections import defaultdict
from keras.optimizers import Adadelta
import itertools
# import necessary modules
import seaborn as sns
import time
import matplotlib.pyplot as plt
import pandas as pd
# matplotlib inline
# np.random.seed(2017)
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import preprocess_input, decode_predictions
from keras.preprocessing import image

import csv
from keras.models import Model
#import cv2

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
def salvarExcelMetricas(history,figpath,nclasses,arqui,counter):
    row = 0
    col = 0
    #saving training data to excel spreadsheet
    workbook = xlsxwriter.Workbook(figpath+str(nclasses)+'classes_'+arqui+'_treinamento'+str(counter)+'.xlsx')
    worksheet = workbook.add_worksheet()
    # list all data in history
    d = history.history
    for key in d.keys():
        row += 1
        worksheet.write(row, col, key)
        for item in d[key]:
            worksheet.write(row, col + 1, item)
            row += 1
    col += 2
    workbook.close()
    print("Métricas foram salvas no Excel")

def savefigtrain(history,figpath,nclasses,arqui,counter):
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Acurácia do Modelo')
    plt.ylabel('Acurácia')
    plt.xlabel('Épocas')
    plt.legend(['Treino', 'Validação'], loc='upper left')
    plt.savefig(figpath+str(nclasses)+'classes_'+arqui+str(counter)+'Acuracia'+'.png')
    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Perda do Modelo')
    plt.ylabel('Perda')
    plt.xlabel('Épocas')
    plt.legend(['Treino', 'Validação'], loc='upper left')
    plt.savefig(figpath+str(nclasses)+'classes_'+arqui+str(counter)+'Perda'+'.png')
    print("Gráficos do Treinamento Salvos")

def confusionmatrix(model,valid_batches,num_of_valid_samples,valid_batch_size,figpath,nclasses,arqui,counter,class_names):
    try:
        Y_pred = model.predict_generator(valid_batches, steps=num_of_valid_samples // valid_batch_size)
        y_pred = np.argmax(Y_pred, axis=1)
        true_classes = valid_batches.classes
        print(y_pred)
        print(true_classes)
        class_labels = list(valid_batches.class_indices.keys())
        eval=model.evaluate_generator(valid_batches,steps=num_of_valid_samples // valid_batch_size)
        with open(figpath+str(nclasses)+'classes_'+arqui+'_teste_perda_acuracia'+str(counter)+'.txt', 'w') as f:
            for item in eval:
                f.write("%s\n" % item)
        confusion = confusion_matrix(true_classes, y_pred)
        confusion_m = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(14, 10))
        ax = sns.heatmap(confusion_m, cmap="YlGnBu", annot=confusion, fmt='d', vmin=0, vmax=1)
        ax.yaxis.set_ticklabels(class_names, rotation=0, ha='right', fontsize=12)
        ax.xaxis.set_ticklabels(class_names, rotation=0, ha='right', fontsize=12)
        plt.xlabel('Predito pelo modelo', fontsize=15)
        plt.ylabel('Classes Reais', fontsize=15)
        plt.savefig(figpath+str(nclasses)+'classes_'+arqui+'_matrix'+str(counter)+'.png')
        report2dict(classification_report(true_classes, y_pred, target_names=class_names))
        report = pd.DataFrame(report2dict(classification_report(true_classes, y_pred, target_names=class_names))).T
        report.to_csv(figpath+str(nclasses)+'classes_'+arqui+'_scoref1_'+str(counter)+'.csv')
        matriz = pd.DataFrame(confusion)
        matriz.to_csv(figpath+str(nclasses)+'classes_'+arqui+'_confusion_numero'+str(counter)+'.csv')
        matrizcsv = pd.DataFrame(confusion_m)
        matrizcsv.to_csv(figpath+str(nclasses)+'classes_'+arqui+'_confusion_porcentagem'+str(counter)+'.csv')
        print("Matriz de Confusão foi salva")
    except:
        pass

    print("Matriz de Confusão foi salva")

def saliencia():
    # layer_idx = utils.find_layer_idx(model, 'classificador')
    #
    # directory = os.fsencode(imgDIR)
    # for file in os.listdir(directory):
    #      filename = os.fsdecode(file)
    #      if filename.endswith(".jpg"):
    #          img1 = utils.load_img(imgDIR+filename, target_size=(224, 224))
    #          #model.layers[layer_idx].activation = activations.softmax
    #          #model = utils.apply_modifications(model)
    #          plt.figure(1)
    #          plt.subplot(2,4,1)
    #          plt.imshow(img1)
    #          plt.title(filename)
    #          for filterU in range(0,7):
    #              grads=visualize_saliency(model, layer_idx, filter_indices= filterU, seed_input= img1)
    #              plt.subplot(2,4,2+filterU)
    #              plt.title(str(class_names[filterU]))
    #              plt.imshow(grads, cmap='jet')
    #          plt.figure(figsize=(14, 10))
    #          plt.subplot(2,4,1)
    #          plt.imshow(img1)
    #          plt.title(filename)
    #          for filterU in range(0,7):
    #              grads=visualize_saliency(model, layer_idx, filter_indices= filterU, seed_input= img1)
    #              plt.subplot(2,4,2+filterU)
    #              plt.title(str(class_names[filterU]))
    #              jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
    #              plt.imshow(overlay(jet_heatmap, img1))
    #              figManager = plt.get_current_fig_manager()
    #              figManager.full_screen_toggle()
    #          #plt.show()
    #          plt.savefig(figpath+filename[0:-4]+'.png')
    #          continue
    #      else:
    #          continue
    pass

def Classificador(epocas,counter,tipoIMG):
        nclasses = 7
        img_rows = 224
        img_cols = 224
        train_batch_size = 10
        valid_batch_size = 11

        if counter == 4:
            num_of_train_samples = 1452
            num_of_valid_samples = 363
        else:
            num_of_train_samples = 1471
            num_of_valid_samples = 363

        class_names = ['Krish', 'MattBeckerich', 'Arzabe','PabloOrtiz', 'MikeRudenball','ManuRaccoon','Dynoz']
        # preprocessamento das imagens
        # train_path = 'C:/Users/Adm/Desktop/Código tcc/dataset7class/trainSEG'
        # valid_path = 'C:/Users/Adm/Desktop/Código tcc/dataset7class/validSEG'
        # test_path = 'C:/Users/Adm/Desktop/Código tcc/dataset7class/test'
        # imgDIR=('C:/Users/Adm/Desktop/Código tcc/dataset7class/test/mattbeckerich/')
        # figpath = 'C:/Users/Adm/Desktop/Código tcc/code7classes/'
        # arqui= 'segmentada_'
        train_path = 'D:/KfoldsTatuadores/'+tipoIMG+'/fold'+str(counter)+'/train/'
        valid_path = 'D:/KfoldsTatuadores/'+tipoIMG+'/fold'+str(counter)+'/valid/'
        # imgDIR=('C:/Users/Adm/Desktop/Código tcc/dataset7class/test/mattbeckerich/')
        figpath = 'D:/resultadosClassificador/'
        arqui= tipoIMG+'_'
        train_datagen = ImageDataGenerator(rescale=1. / 255,
        #                                   rotation_range=40,
        #                                   width_shift_range=0.2,
        #                                   height_shift_range=0.2,
        #                                   shear_range=0.2,
                                           # zoom_range=0.2,
                                           # horizontal_flip=True,
                                           fill_mode='nearest')

        train_batches = train_datagen.flow_from_directory(train_path, target_size=(img_rows, img_cols), classes=class_names, batch_size=train_batch_size)
        valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(img_rows, img_cols), classes=class_names, batch_size=valid_batch_size)

        vgg16model = keras.applications.vgg16.VGG16()
        vgg16model.layers.pop()
        model = Sequential()
        for layer in vgg16model.layers:
            model.add(layer)
        for layer in model.layers:
            layer.trainable = False
        # model.add(Dense(100, activation='softmax', name="classificador75"))
        model.add(Dense(nclasses, activation='softmax', name="classificador"))
        model.summary()
        model.compile(optimizer='adadelta',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        history = model.fit_generator(train_batches, steps_per_epoch=num_of_train_samples // train_batch_size,
                                      validation_data=valid_batches, validation_steps=num_of_valid_samples // valid_batch_size, epochs=epocas)
        #save model weights
        # model.save(figpath+str(nclasses)+'classes_'+arqui+str(counter)+'weights.h5')
        # print('Model saved to H5 file')
        # Tutorial @ https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
        savefigtrain(history,figpath,nclasses,arqui,counter)
        salvarExcelMetricas(history,figpath,nclasses,arqui,counter)
        confusionmatrix(model,valid_batches,num_of_valid_samples,valid_batch_size,figpath,nclasses,arqui,counter,class_names)

def main():
    Parametros=[]
    with open('parametrosclassificador.csv') as csvfile:
        ArquivoCSV=csv.reader(csvfile, delimiter=';')
        for row in ArquivoCSV:
            Parametros.append(row[0].split('\t'))
    Parametros = Parametros[6:]

    for linha in Parametros:
        epocas = int(linha[0])
        tipoIMG = linha[1]
        counter = int(linha[2])
        Classificador(epocas,counter,tipoIMG)

if __name__ == "__main__":
    main()
