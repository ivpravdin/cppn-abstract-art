import tensorflow as tf
from random import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Activation
from tensorflow.keras import Input
from tensorflow.keras.initializers import RandomUniform, RandomNormal, Constant
import matplotlib.pyplot as plt
import cv2 as cv
from random import random
import numpy as np


class Model():
    
    def __init__(self, input_shape = 2, n_hidden_layers = 8, n_neurons = 16, activation = 'tanh', init = RandomNormal(), *args):
        self.model = Sequential()

        self.model.add(Input(shape = (input_shape)))

        initializer = init(*args)
        
        self.model.add(Dense(units = n_neurons, kernel_initializer = initializer, use_bias=True, bias_initializer = 'glorot_uniform'))
        self.model.add(Activation(activation))
        
        for _ in range(n_hidden_layers):
            self.model.add(Dense(units = n_neurons, kernel_initializer = initializer, use_bias=False))
            self.model.add(Activation(activation))
        
        self.model.add(Dense(units = 1, kernel_initializer = initializer, use_bias=False))
        self.model.add(Activation('sigmoid'))
        
    def predict(self, data):
        return self.model.predict(data)

def generate_pixel_map(w, h, zoom):
    body = cv.resize(cv.imread('body.jpg', 0), (w, h))
    pixel_map = np.zeros(shape=(h, w, 3))
    pixel_map = pixel_map.tolist()
    a = 0
    for i in range(h):
            for j in range(w):
                pixel_map[i][j][0]=a
            a+=1
    a = 0
    """for i in range((int)(w/2)):
            for j in range(a, h-a):
                pixel_map[j][i]=a
            a+=1
    a = 0
    for i in range((int)(w/2)):
            for j in range(a, h-a):
                pixel_map[j][w-1-i]=a
            a+=1
    a = 0
    for i in range((int)(h/2)):
            for j in range(a, w-a):
                pixel_map[h-1-i][j]=a
            a+=1"""
    pixel_map = np.array(pixel_map)
    max_px = pixel_map.max()
    pixel_map = pixel_map.tolist()
    for i in range(h):
        for j in range(w):
            pixel_map[i][j][0] = (pixel_map[i][j][0]/max_px-0.5)*zoom
            #pixel_map[i][j][1] = (pixel_map[i][j][1]/max_px-0.5)*zoom
            pixel_map[i][j][1] = np.sqrt((i/h-0.5)**2+(j/w-0.5)**2)
            if i>h/2:
                pixel_map[i][j][2] = 0
            else:
                pixel_map[i][j][2] = ((1-(body[i][j]/255))-0.5)*pixel_map[i][j][0]
            #pixel_map[i][j][2]=((i/h)-0.5)**2+((j/w)-0.5)**2
            
    pixel_map = np.array(pixel_map)
    return pixel_map
    
def generate_pixel_map_with_colors(w, h, img, zoom):
    pixel_map = []
    for i in range(h):
        for j in range(w):
            pixel_map.extend([((i/h-0.5)*zoom), ((j/w-0.5)*zoom), *img[i][j]])
    return pixel_map

def generate(w, h, model, zoom):
    pixel_map = np.reshape(np.array(generate_pixel_map(w, h, zoom)), (h * w, model.model.input.shape[1]))
    img = model.predict(pixel_map)
    return img

def generate_with_img(w, h, img, model, zoom):
    img = (img / 255) - 0.5
    pixel_map = np.reshape(np.array(generate_pixel_map_with_colors(w, h, img, zoom)), (h * w, model.model.input.shape[1]))
    img = model.predict(pixel_map)
    return img
    
#for _ in range(10): #хорошие параметры
#    model = Model(n_hidden_layers = 12, std = 1, n_neurons = 16)
#    cv.imwrite(f'experiment/{i}.jpg', generate(2000, 2000, model, 0.3))
#    i+=1

"""
h = 500
w = 500
speed_weight = 0.005
i = 0
for n in range(0, 8):
    speed = (random()*2-1)*speed_weight
    for j in range(0, 30):
        img = generate(w, h, model, 0.003)
        cv.imwrite(f'animation_experiment_4/{i}.jpg', np.reshape((img*255).astype(int), (h, w, 3)))
        pixel_map = np.reshape(np.array(generate_pixel_map(w, h, 3)), (h * w, 2))
        weights_layer_1 = np.array(model.model.layers[0].get_weights()) + speed
        weights_layer_1[1] = model.model.layers[0].get_weights()[1]
        model.model.layers[0].set_weights(weights_layer_1)
        for lay in model.model.layers:
            if 'dense' in lay.name and lay != model.model.layers[0]: 
                weights = np.array(lay.get_weights())
                weights += weights*speed
                lay.set_weights(weights)
        i += 1
"""