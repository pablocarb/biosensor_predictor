import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import classification_report
from tensorflow import keras
from keras.models import Model
from keras.layers import LSTM, Dense, Input
from keras.layers import Concatenate
from keras import metrics
from tensorflow.keras.optimizers import Adam
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def get_even_clusters(X, cluster_size):
    n_clusters = int(np.ceil(len(X)/cluster_size))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto').fit(X)
    centers = kmeans.cluster_centers_
    centers = centers.reshape(-1,1,X.shape[-1]).repeat(cluster_size, 1).reshape(-1, X.shape[-1])
    distance_matrix = cdist(X, centers)
    clusters = linear_sum_assignment(distance_matrix)[1]//cluster_size
    clusters_sep = []
    for i in range(len(dict.fromkeys(clusters))):
        a = []
        for p in range(len(clusters)):
            if clusters[p] == i:
                a.append(p)
        clusters_sep.append(a)
    return clusters_sep

aa_m = np.load('/media/hector/Datos/Doctorado/Articulo Raul/new_aa_matrix.npy')
fp_m = np.load('/media/hector/Datos/Doctorado/Articulo Raul/new_fingerprints_matrix.npy')
cl_m = np.load('/media/hector/Datos/Doctorado/Articulo Raul/clustering_matrix.npy')

lbl = np.ones(len(fp_m))
for i in range(1, len(lbl),2):
    lbl[i] = 0
    
aa_train, aa_test, fp_train, fp_test, lbl_train, lbl_test, cl_train, cl_test = train_test_split(aa_m, fp_m, lbl, cl_m, test_size = 0.20)

# Input layers shape
i1_shape = np.shape(aa_m)[1:]
i2_shape = np.shape(fp_m)[1]

# Model parameters
loss_function = 'binary_crossentropy'
opt = Adam(learning_rate=0.0001)
metrics = ['accuracy']

num_folds = 5
batch_size = 50
num_epochs = 30 

lstm_units = 90
densefp_units = 70
dense_units = 70

verbosity = 0
kfold = KFold(n_splits = num_folds)
# Branch for sequences matrixes. A LSTM layer is used
input1 = Input(shape = i1_shape)
i11 = LSTM(lstm_units, activation='relu', recurrent_activation='sigmoid',
           dropout=0.2, recurrent_dropout=0.01)(input1)

# Branch for fingerprints
input2 = Input(shape = i2_shape)
i21 = Dense(densefp_units, activation='relu')(input2)

# Joint of branches
concat = Concatenate()([i11, i21])

# # Other layers TEST (DROPOUTS)
dense1 = Dense(dense_units, activation='relu')(concat)

# # Output layer
out = Dense(1)(dense1)

# # Creation and compilation of the model
model = Model(inputs = [input1, input2], outputs = out)
model.compile(loss = loss_function, optimizer = opt, metrics = metrics)
print(model.summary())

# Defining the per-fold score lists
acc_per_fold = []
loss_per_fold = []

fold_num = 1

# Saving the initial model weights
Wsave = model.get_weights()

# Clustering the data in even-sized clusters
all_train = get_even_clusters(cl_train, (len(cl_train) // num_folds)+1)

for i in range(num_folds):
    val = all_train[i]
    t = all_train[0:i] + all_train[i+1:]
    train = []
    for j in t:
        train = train + j
    model.set_weights(Wsave)
    
    # Generate a print
    print('------------------------------------------------------------------')
    print(f'Training for fold {fold_num} ...')
    
    
    history = model.fit([aa_train[train], fp_train[train]], lbl_train[train],
                      validation_data=([aa_train[val], fp_train[val]], lbl_train[val]),
                      batch_size = batch_size, 
                      epochs = num_epochs, 
                      verbose = verbosity)
    
    # summarize history for accuracy
    plt.figure(dpi=100)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.figure(dpi=100)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    fold_loss = history.history['val_loss'][-1]
    fold_accuracy = history.history['val_accuracy'][-1] * 100
    print(f'Score for fold {fold_num}: {list(history.history.keys())[2]} of {fold_loss}; {list(history.history.keys())[3]} of {fold_accuracy} %')

    acc_per_fold.append(fold_accuracy)
    loss_per_fold.append(fold_loss)
    
    # Increasing fold number
    fold_num = fold_num + 1

print('------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]} %')
print('------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} % (+- {np.std(acc_per_fold)} %)')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------')

print('Is the model validated prepared to be trained?')
answer = input("Press 'train' for training the model, press any other key for not training it: ")
if answer == 'train':
  print('Starting training...')

  model.set_weights(Wsave)

  history = model.fit([aa_train, fp_train], lbl_train,
                      batch_size = batch_size, 
                      epochs = num_epochs, 
                      verbose = verbosity)

  loss = history.history['loss'][-1]
  accuracy = history.history['accuracy'][-1] * 100
  print('------------------------------------------------------------------')
  print('Scores:')
  print(f'> Accuracy: {accuracy} %')
  print(f'> Loss: {loss}')
  print('------------------------------------------------------------------')

  results = model.evaluate([aa_test, fp_test], lbl_test, 
                            verbose=0)
  print(f'{model.metrics_names[0]} of {results[0]}; {model.metrics_names[1]} of {results[1]*100} %')

  preds_test = model.predict([aa_test, fp_test]).ravel()
  fpr, tpr, thresholds = roc_curve(lbl_test, preds_test)
  auc_value = auc(fpr, tpr)

  plt.figure(dpi=100)
  plt.plot([0, 1], [0, 1], 'k--')
  plt.plot(fpr, tpr, label='AUC = {:.3f})'.format(auc_value))
  plt.xlabel('False positive rate')
  plt.ylabel('True positive rate')
  plt.title('ROC curve')
  plt.legend(loc='best')
  plt.show()

  preds_binary = preds_test.copy()
  for i in range(len(preds_binary)):
    if preds_binary[i] > 0.5:
      preds_binary[i] = 1
    else:
      preds_binary[i] = 0

  print(classification_report(lbl_test, preds_binary))

else:
  print('Model discarded')
