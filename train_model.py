import tensorflow as tf
import numpy as np, os, csv, cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D, BatchNormalization, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2
import keras.optimizers as optimizers
from keras import backend as K
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint


### Function to read the training and validation set as a single variable before model execution
def readLog(filename, nb_classes=3):
    image = []
    category = []
 
    with open(filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            img = cv2.cvtColor(cv2.imread(row[0]).astype('float32'), cv2.COLOR_BGR2Lab)
            if img == None:
              print('Error in reading img in file {0}'.format(row[0]))
            type = np_utils.to_categorical(float(row[1]), nb_classes)
            image.append(img)
            category.append(type)
        category = np_utils.to_categorical(category, nb_classes)        
    return image, category
    
### Generator function to read the training and validation data on the fly
#def generator(samples, batch_size=32, nb_classes=3):
#    num_samples = len(samples)
#    while 1: # Loop forever so the generator never terminates
#        sklearn.utils.shuffle(samples)
#        for offset in range(0, num_samples, batch_size):
#            batch_samples = samples[offset:offset+batch_size]
#
#            images = []
#            img_category = []
##            if (mode=='center'):
#            for batch_sample in batch_samples:
#              image = mpimg.imread(batch_sample[0]).astype('float32')
#              img_type = float(batch_sample[1])
#              images.append(image)
#              img_category.append(img_type)
#
#            # trim image to only see section with road
#            X_train = np.array(images)
#            y_train = np_utils.to_categorical(np.array(img_category)-1, nb_classes)
##            y_train = np.array(img_category)
#            
##            gen_output = tuple(sklearn.utils.shuffle(X_train, y_train))
#            yield sklearn.utils.shuffle(X_train, y_train)


def generator(samples, batch_size=32, nb_classes=2):
  num_samples = len(samples)
  while 1:
    rand_ints = np.random.randint(1, high = num_samples, size = batch_size)
    batch_samples = [ samples[i] for i in rand_ints]
    images = []
    img_category= []
    for batch_sample in batch_samples:
      try:
        images.append(cv2.resize(cv2.cvtColor(cv2.imread(batch_sample[0]).astype('float32'), cv2.COLOR_BGR2Lab), dsize = (192, 256)))
      except:
        print(batch_sample[0])          
      img_category.append(batch_sample[1])
      
    X_train = np.array(images)
    Y_train = np_utils.to_categorical(np.array(img_category).astype('float32'), nb_classes)
    
    yield sklearn.utils.shuffle(X_train, Y_train)

def locnet():
    b = np.zeros((2, 3), dtype='float32')
    b[0, 0] = 1
    b[1, 1] = 1
    W = np.zeros((50, 6), dtype='float32')
    weights = [W, b.flatten()]
    locnet = Sequential()

    locnet.add(Convolution2D(16, (7, 7), padding='valid', input_shape=inputShape))
    locnet.add(MaxPooling2D(pool_size=(2, 2)))
    locnet.add(Convolution2D(32, (5, 5), padding='valid'))
    locnet.add(MaxPooling2D(pool_size=(2, 2)))
    locnet.add(Convolution2D(64, (3, 3), padding='valid'))
    locnet.add(MaxPooling2D(pool_size=(2, 2)))

    locnet.add(Flatten())
    locnet.add(Dense(128))
    locnet.add(Activation('elu'))
    locnet.add(Dense(64))
    locnet.add(Activation('elu'))
    locnet.add(Dense(6, weights=weights))

    return locnet

### User model parameters

def train_model(log_file):
  
#  log_file = './img_key_train.csv'
      
  csv_path_parts = csv_path.split('_')
  img_class = csv_path_parts[-1].split('.')[0]
  
  num_epochs = 200
  batchSize = 32              # Select batch size
  Activation_type = 'relu'    # Select activation type
  nb_samples = 2
  inputShape = (256, 192, 3)
  regularization_rate = 0.04
  dropout_prob = 0.4
  
  ### Read the drive log csv file
  samples = []
  with open(log_file) as csvfile:
      reader = csv.reader(csvfile)
      for line in reader:
          samples.append(line)
          
  csvfile.close()
  print('Total number of samples = {}'.format(len(samples)))
  
  ### Split captured data as Training and Validation Set
  
  train_samples, validation_samples = train_test_split(samples, test_size=0.2)
  
  
  ### compile and train the model using the generator function
  train_generator = generator(train_samples, batch_size=batchSize)
  validation_generator = generator(validation_samples, batch_size=batchSize)
  print('Generator initialized...')
  
  #data = np.load('img_data.npz')
  #img_data = data['img_data']
  #label_data = data['label_data']
  
  print('Training...')
  ### Setup the Keras model
  model = Sequential()
  model.add(Cropping2D(cropping=((65,20), (0,0)), input_shape=inputShape))
  model.add(Lambda(lambda x: x/255.0 - 0.5))
  #model.add(BatchNormalization())
  #model.add(SpatialTransformer(localization_net=locnet(), output_size=(512, 512), input_shape=inputShape))
  model.add(Convolution2D(512, (5, 5), padding='same', subsample=(2,2), activation=Activation_type, 
                          kernel_regularizer=l2(regularization_rate)))
  #model.add(BatchNormalization())
  #model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Convolution2D(128, (3, 3), padding='same', activation=Activation_type, kernel_regularizer=l2(regularization_rate)))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Convolution2D(128, (5, 5), padding='same', activation=Activation_type, kernel_regularizer=l2(regularization_rate)))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  # model.add(BatchNormalization())
  model.add(Convolution2D(128, (7, 7), padding='same', activation=Activation_type, kernel_regularizer=l2(regularization_rate)))
  #model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Convolution2D(256, (5, 5), padding='same', activation=Activation_type, kernel_regularizer=l2(regularization_rate)))
  model.add(Convolution2D(256, (5, 5), padding='same', activation=Activation_type, kernel_regularizer=l2(regularization_rate)))
  #model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Convolution2D(128, (3, 3), padding='same', activation=Activation_type, kernel_regularizer=l2(regularization_rate)))
  #model.add(BatchNormalization())
  model.add(Convolution2D(128, (3, 3), padding='same', activation=Activation_type, kernel_regularizer=l2(regularization_rate)))
  #model.add(Convolution2D(128, (3, 3), padding='same', activation=Activation_type, kernel_regularizer=l2(regularization_rate)))
  model.add(MaxPooling2D(pool_size=(4, 4)))
  model.add(Flatten())
  #model.add(Dense(64))
  model.add(Dropout(dropout_prob))
  model.add(Dense(nb_samples, activation="softmax"))
  
  
  optimize_adam = optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-9)
  model.compile(loss='categorical_crossentropy', optimizer= optimize_adam, metrics=['accuracy'])
  
  checkpointer = ModelCheckpoint(filepath= os.sep.join(('./output','_'.join(('img_class',str(img_class),'weights.hdf5')))), verbose=1, save_best_only=True, save_weights_only=True)
  
  #history_object = model.fit(img_data, label_data, batch_size= batchSize, 
  #                                     validation_split= 0.2, shuffle = 1, 
  #                                     epochs = num_epochs, callbacks = [checkpointer], verbose = 1)
  #
  
  #a = next(train_generator)
#  model_op = -1
  try:
    history_object = model.fit_generator(train_generator, steps_per_epoch = np.floor(len(train_samples)/batchSize), 
                                         validation_data =validation_generator, validation_steps = np.floor(len(validation_samples)/batchSize).astype('int'), 
                                         epochs=num_epochs, verbose = 1, callbacks = [checkpointer])
    
    
    train_generator.close()
    validation_generator.close()
    
  #    model_op = model
  #    model.save('./output/model.h5')
    
    ### print the keys contained in the history object
    print(history_object.history.keys())
    
    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model loss : Type ' + str(img_class))
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
    plt.savefig('model_loss_progression.png')
    
    plt.plot(history_object.history['acc'])
    plt.plot(history_object.history['val_acc'])
    plt.title('model accuracy : Type ' + str(img_class))
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
    plt.savefig('model_accuracy_progression.png')
    return model
  
  except:
    train_generator.close()
    validation_generator.close()
    return -1
    
if __name__ == '__main__':
  
  os.chdir("/Data/cerv_cancer") 
  
  for i in range(1,4):
    csv_path = '.'.join(('_'.join(('img_key_train_process', str(i))), 'csv'))
    model = train_model(csv_path)
    
    if model != -1:
      model.save(os.sep.join(('./output', '_'.join(('img_class',str(i),'model.h5')))))

    
