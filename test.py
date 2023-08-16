import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class test:
    
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) =  mnist.load_data() #tuples where x is the pixdl data and y is the classification
    '''
    x_train =  tf.keras.utils.normalize(x_train, axis=1) #change pixel data from 0-255 to 0-1
    x_test =  tf.keras.utils.normalize(x_test, axis=1) #change pixel data from 0-255 to 0-1
    print(x_train.shape, y_train.shape)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape = (28,28))) #changes 28 x 28 image to a single layer of pixels
    model.add(tf.keras.layers.Dense(128, activation='relu')) #using activation function relu
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))  #layer to represent the 10 output digits 0-9 
                                                                # with a probability of how likely it is that digit
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10)

    model.save('handwritten.model')
    '''
    model = tf.keras.models.load_model('handwritten.model')
    loss, accuracy = model.evaluate(x_test, y_test)

    print(loss)
    print(accuracy)

    while os.path.isfile(f"/Users/ayushsingh/Downloads/digits/twomy.png"):
        try:
            img = cv2.imread(f"/Users/ayushsingh/Downloads/digits/twomy.png")[:,:,0]
            img = np.invert(np.array([img]))
            prediction = model.predict(img)
            print(f"This digit is probably a {np.argmax(prediction)}")
            plt.imshow(img[0], cmap=plt.cm.binary)
            plt.show()
        except:
            print("ERROR")
    