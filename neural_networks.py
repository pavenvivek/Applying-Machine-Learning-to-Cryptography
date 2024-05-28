#!/usr/local/bin/python3
#
# orient.py : a image classifier
#
# Submitted by : Paventhan Vivekanandan, username : pvivekan
#
#

import numpy as np
import random
import math
from Pyfhel import Pyfhel, PyPtxt, PyPtxt
#import sys

def read_data(filename):
    fh = open(filename, 'r')
    data = []
    pos = []
    img = []

    for line in fh:
        values = line.split()
        img.append(values[0])
        pos.append(values[1])
        #k = 2
        #pixels = []
        #for p in range(0, 192):
        #    curr_pixel = []
        #    for rgb in range(0, 3):
        #        curr_pixel.append(values[k])
        #        k = k+1
        #    pixels.append(curr_pixel)
        
        data.append(values[2:])

    return (data, pos, img)

def write_data(filename, network, b1, b2, input_index):
    fh = open(filename, 'w')

    fh.write("IND ")
    for i in range(0, len(input_index)):
        fh.write(str(input_index[i]) + " ")
    fh.write("\n")
    
    fh.write("B " + str(b1) + " " + str(b2) + "\n")
    fh.write("I " + str(len(network.input_layer.nodes)) + "\n")

    fh.write("H " + str(len(network.hidden_layers.nodes)) + " ")
    fh.write(str(len(network.hidden_layers.nodes[0].weights)) + " ")
    for i in range(0, len(network.hidden_layers.nodes)):
        for j in range(0, len(network.hidden_layers.nodes[i].weights)):
            fh.write(str(network.hidden_layers.nodes[i].weights[j]) + " ")
    fh.write("\n")

    fh.write("O " + str(len(network.output_layer.nodes)) + " ")
    fh.write(str(len(network.output_layer.nodes[0].weights)) + " ")
    for i in range(0, len(network.output_layer.nodes)):
        for j in range(0, len(network.output_layer.nodes[i].weights)):
            fh.write(str(network.output_layer.nodes[i].weights[j]) + " ")
    fh.write("\n")

def read_network(filename):
    fh = open(filename, 'r')
    
    data = fh.readline()
    values = data.split()

    input_index = values[1:]
    input_index = list(map(lambda x : int(x), input_index))
    
    data = fh.readline()
    values = data.split()
    
    #b1 = float(values[1])
    #b2 = float(values[2])

    #discretizing bias values
    b1 = float(values[1])
    #b1 = round(b1/100, 2) * 100
    b2 = float(values[2])
    #b2 = round(b2/100, 2) * 100
    
    network = neural_network()
    input_layer = None
    hidden_layers = None
    output_layer = None

    for line in fh:
    #data = fh.readline()
        values = line.split()
        
        
        if values[0] == "I":
            l_nodes = int(values[1])
    
            inode = []
            for i in range(0, l_nodes):
                new_node = node("i" + str(i))
                inode.append(new_node)
                
            input_layer = layer("input")
            input_layer.nodes = inode
        else:
            l_nodes = int(values[1])
            l_weights = int(values[2])
    
            hnode = []
            st_ind = 3
            for i in range(0, l_nodes):
                new_node = node("i" + str(i))
                new_node.weights = []
                for j in range(0, l_weights):
                    new_node.weights.append(float(values[st_ind + j]))
                    
                    # discretizing the network
                    #curr_weight = float(values[st_ind + j])
                    #curr_weight = round(curr_weight/100, 2) * 100
                    #new_node.weights.append(curr_weight)
                    
                    #print(str(values[st_ind + j]) + " ")
                st_ind = st_ind + l_weights
                hnode.append(new_node)
            #print("\n")
            
            if values[0] == "H":    
                hidden_layers = layer("hidden")
                hidden_layers.nodes = hnode
            else:
                output_layer = layer("output")
                output_layer.nodes = hnode
                
    network.input_layer = input_layer
    network.hidden_layers = hidden_layers
    network.output_layer = output_layer
    
    return (network, b1, b2, input_index)
                
            
            
            

class node(object):
    
    def __init__(self, name):
        self.inputs = None
        self.weights = None
        self.adjusted_weights = None
        self.output = None
        self.value = None # for input node
        self.name = name
        
class layer(object):
        
    def __init__(self, name):
        self.nodes = None
        self.type = None
        self.name = name
        
class neural_network(object):      
    
    def __init__(self):  
        self.hidden_layers = None
        self.input_layer = None
        self.output_layer = None
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoidPrime(self, x):
        return x * (1 - x)
    
    def relu(self, x):
        return max(0, x)

    def reluPrime(self, x):
        if x <= 0:
            return 0
        else:
            return 1
        
    def sign(self, x):
        if x < 0:
            return -1
        else:
            return 1

    def hardSigmoid(self, x):
        return max(0, min(1, x * 0.2 + 0.5))

    def hardSigmoidPrime(self, x):
        if x <= 0:
            return 0
        else:
            if x >= 1:
                return 0
            else:
                return 0.2
            
        #return max(0, min(1, x * 0.2 + 0.5))
    
    def activation(self, x):
        return self.sigmoid(x)
        #return self.relu(x)
        #return self.sign(x)
        #return self.hardSigmoid(x)

    def activationPrime(self, x):
        return self.sigmoidPrime(x)
        #return self.reluPrime(x)
        #return self.hardSigmoidPrime(x)
    
        
def construct_neural_network(data_length):

    network = neural_network()

    inode = []
    for i in range(0, data_length):
        new_node = node("i" + str(i))
        inode.append(new_node)
    
    input_layer = layer("input")
    input_layer.nodes = inode
       
    hnode = []
    for i in range(0, data_length):
        h_new_node = node("h" + str(i))
        hnode.append(h_new_node)

   
    hidden_layer = layer("hidden")
    hidden_layer.nodes = hnode
    
    o1 = node("o1")
    o2 = node("o2")
    o3 = node("o3")
    o4 = node("o4")

    output_layer = layer("output")
    output_layer.nodes = [o1, o2, o3, o4]

    network.input_layer = input_layer
    network.hidden_layers = hidden_layer
    network.output_layer = output_layer
    
    return network


def feed_forward_and_back_propagation(network, data_map, input_index, b1, b2):
    
    #print("data_map -> {}".format(data_map))
        
    EPOCHS = 1
    e = 0
    
    learning_rate = 0.5
    
    while e < EPOCHS:
        count = 0
        while count < len(data_map):  # while len(data_map) != 0:  if shuffling enabled
            #print(data_map[0])
            (data, expected_output) = data_map[count] # data_map[0] if shuffling enabled
    
            for i in range(0, len(network.input_layer.nodes)):
                network.input_layer.nodes[i].value = (int(data[input_index[i]])/255)
            
            for i in range(0, len(network.hidden_layers.nodes)):
                h = network.hidden_layers.nodes[i]
                h.inputs = list(map(lambda x : x.value, network.input_layer.nodes))
                
                if count == 0 and e == 0:
                    h.weights = [round(random.random(), 2) for i in range(0, len(h.inputs))]
                #h.weights = weights["hidden"][i]
                
                #print("input list -> {}, w -> {}".format(h.inputs, h.weights))
                
                h.output = network.activation((sum([i*j for (i,j) in zip(h.inputs, h.weights)])) + b1)
                
            output = []
            for i in range(0, len(network.output_layer.nodes)):
                o = network.output_layer.nodes[i]
                o.inputs = list(map(lambda x : x.output, network.hidden_layers.nodes))
                
                if count == 0 and e == 0:
                    o.weights = [round(random.random(), 2) for i in range(0, len(o.inputs))]
                #o.weights = weights["output"][i]
                
                #print("i -> {}, input list -> {}, w -> {}".format(i, o.inputs, o.weights))
                
                o.output = network.activation(sum([i*j for (i,j) in zip(o.inputs, o.weights)]) + b2)
                output.append(o.output)
                
            #E = [math.pow(i - j, 2) for (i,j) in zip(expected_output, output)]
            #E_total = sum(E)
            
            #adjusted_weights = []
            
            for i in range(0, len(network.output_layer.nodes)):
                d1 = 2 * (network.output_layer.nodes[i].output - expected_output[i])
                
                #print("1 -> {}, 2 -> {}".format(network.output_layer.nodes[i].output, expected_output[i]))
                
                d2 = network.activationPrime(network.output_layer.nodes[i].output)
                network.output_layer.nodes[i].adjusted_weights = []
                
                for j in range(0, len(network.hidden_layers.nodes)):
                    d3 = network.hidden_layers.nodes[j].output
                    w_i_j = network.output_layer.nodes[i].weights[j]
                    gradient = d1 * d2 * d3
                    #adjusted_weights.append(w_i_j - (0.5 * d1 * d2 * d3))
                    
                    #if network.output_layer.nodes[i].adjusted_weights is None:
                    #    network.output_layer.nodes[i].adjusted_weights = []
                    
                    network.output_layer.nodes[i].adjusted_weights.append(w_i_j - (learning_rate * gradient))
                    #print("value: d1 -> {}, d2 -> {}, d3 -> {}".format(d1, d2, d3))
                    #print("{}".format(network.output_layer.nodes[i].adjusted_weights))
                    
            for i in range(0, len(network.hidden_layers.nodes)):
                d_sum = 0
                for j in range(0, len(network.output_layer.nodes)):
                    d1 = 2 * (network.output_layer.nodes[j].output - expected_output[j])
                    d2 = network.activationPrime(network.output_layer.nodes[j].output)
                    #d3 = network.hidden_layers.nodes[j].output
                    w_i_j = network.output_layer.nodes[j].weights[i]
                    d_sum = d_sum + (w_i_j * d1 * d2)
                    
                d3 = network.activationPrime(network.hidden_layers.nodes[i].output)
                network.hidden_layers.nodes[i].adjusted_weights = []
                
                for k in range(0, len(network.input_layer.nodes)):
                    d4 = network.hidden_layers.nodes[i].inputs[k]
                    w_i_k = network.hidden_layers.nodes[i].weights[k]
                    gradient = d_sum * d3 * d4
                    
                    #if network.hidden_layers.nodes[i].adjusted_weights is None:
                    #    network.hidden_layers.nodes[i].adjusted_weights = []
                        
                    network.hidden_layers.nodes[i].adjusted_weights.append(w_i_k - (learning_rate * gradient))
                    
            for i in range(0, len(network.output_layer.nodes)):
                o = network.output_layer.nodes[i]
                o.weights = o.adjusted_weights
                
            for i in range(0, len(network.hidden_layers.nodes)):
                h = network.hidden_layers.nodes[i]
                h.weights = h.adjusted_weights
            
            #enable shuffling -> takes lot of time    
            #data_map = np.delete(data_map, 0,0)
            #np.random.shuffle(data_map)  # shuffling data for stochastic approach

            count = count + 1
        e = e + 1
    
    return network
    
def classify(network, b1, b2, data, pos, img, input_index, filename):
    
    count = 0
    #b1 = 0.35
    #b2 = 0.60
    
    expected_output = {}
    expected_output[0] = 0
    expected_output[90] = 1
    expected_output[180] = 2
    expected_output[270] = 3

    correctly_classified = 0
    
    fh = open(filename, 'w')
    
    while count < len(data):
        
        for i in range(0, len(network.input_layer.nodes)):
            network.input_layer.nodes[i].value = (int(data[count][input_index[i]])/255) 
        
        for i in range(0, len(network.hidden_layers.nodes)):
            h = network.hidden_layers.nodes[i]
            h.inputs = list(map(lambda x : x.value, network.input_layer.nodes))
            
            h.output = network.activation(sum([i*j for (i,j) in zip(h.inputs, h.weights)]) + b1) # * b1)
            
        output = []
        for i in range(0, len(network.output_layer.nodes)):
            o = network.output_layer.nodes[i]
            o.inputs = list(map(lambda x : x.output, network.hidden_layers.nodes))
            
            o.output = network.activation(sum([i*j for (i,j) in zip(o.inputs, o.weights)]) + b2) # * b2)
            output.append(o.output)
            
        max_i = output.index(max(output))
        
        if expected_output[int(pos[count])] == max_i:
            correctly_classified = correctly_classified + 1
            fh.write(img[count] + " " + str(pos[count]) + "\n")
            
        count = count + 1
    
    
    print("Correctly classified: {} out of {}".format(correctly_classified, len(data)))
    print("Incorrectly classified: {} out of {}".format(len(data) - correctly_classified, len(data)))
    
    classification_accuracy = (correctly_classified/len(data)) * 100
    
    print("Classification accuracy: {}%".format(round(classification_accuracy, 2)))
    
    
def he_classify(network, b1, b2, data, pos, img, input_index, filename):
    
    count = 0
    #b1 = 0.35
    #b2 = 0.60
    
    expected_output = {}
    expected_output[0] = 0
    expected_output[90] = 1
    expected_output[180] = 2
    expected_output[270] = 3

    correctly_classified = 0
    
    #p_val = 65537  # for this value the noise on cipher-text grows over the limit at the hidden layer
    #p_val = 1964769281  # setting large prime allows computation at deeper layers 
    he = Pyfhel()
    #he.contextGen(p=p_val)
    he.contextGen(p=1964769281, m=8192, base=2, sec=192, flagBatching=True)
    he.keyGen()
    #he.relinKeyGen(60, 10)
    
    fh = open(filename, 'w')
    
    for i in range(0, len(network.hidden_layers.nodes)):
            h = network.hidden_layers.nodes[i]
            
            for j in range(0, len(h.weights)):
                h.weights[j] = he.encodeFrac(h.weights[j])
                
    for i in range(0, len(network.output_layer.nodes)):
            o = network.output_layer.nodes[i]
            
            for j in range(0, len(o.weights)):
                o.weights[j] = he.encodeFrac(o.weights[j])
    
    while count < len(data):
        
        for i in range(0, len(network.input_layer.nodes)):
            network.input_layer.nodes[i].value = he.encrypt(int(data[count][input_index[i]])/255)
        
        for i in range(0, len(network.hidden_layers.nodes)):
            h = network.hidden_layers.nodes[i]            
            h.inputs = list(map(lambda x : x.value, network.input_layer.nodes))
            
            sum_c = he.encrypt(0.0)
            for (i,j) in zip(h.inputs, h.weights):
                c = i * j
                sum_c = sum_c + c #he.add(c, sum_c, in_new_ctxt=False)
            
            sum_c = sum_c + he.encodeFrac(b1)
            h.output = he.encrypt(network.activation(he.decode(he.decrypt(sum_c))))
            
        output = []
        for i in range(0, len(network.output_layer.nodes)):
            o = network.output_layer.nodes[i]
            o.inputs = list(map(lambda x : x.output, network.hidden_layers.nodes))
            
            sum_c = he.encrypt(0.0)
            for (i,j) in zip(o.inputs, o.weights):
                c = i * j #he.multiply(i, j, in_new_ctxt=True)
                sum_c = sum_c + c

            #print("noiseLevel: sum_c -> {}".format(he.noiseLevel(sum_c)))
            sum_c = sum_c + he.encodeFrac(b2)
            o.output = network.activation(he.decode(he.decrypt(sum_c)))
            output.append(o.output)
            
        max_i = output.index(max(output))
        
        if expected_output[int(pos[count])] == max_i:
            correctly_classified = correctly_classified + 1
            fh.write(img[count] + " " + str(pos[count]) + "\n")
            
        count = count + 1
    
    
    print("Correctly classified: {} out of {}".format(correctly_classified, len(data)))
    print("Incorrectly classified: {} out of {}".format(len(data) - correctly_classified, len(data)))
    
    classification_accuracy = (correctly_classified/len(data)) * 100
    
    print("Classification accuracy: {}%".format(round(classification_accuracy, 2)))


def nnet_classifier(run_type, data_file, model_file):

    (data, pos, img) = read_data(data_file)
    
    if run_type == "train":    
    #(tdata, tpos, timg) = read_data(sys.argv[2])
    
        expected_output = []
        
        for i in range(0, len(pos)):
            if int(pos[i]) == 0:
                output = [1, 0, 0, 0]
            elif int(pos[i]) == 90:
                output = [0, 1, 0, 0]
            elif int(pos[i]) == 180:
                output = [0, 0, 1, 0]
            elif int(pos[i]) == 270:
                output = [0, 0, 0, 1]
    
            expected_output.append(output)
        
        input_index = [2, 47, 185, 119, 122, 170, 146, 173, 176, 149, 191, 188] #, 11, 98, 167, 164, 41, 38, 8, 182] #list(map(lambda x : int(x), pos)) 
        
        network = construct_neural_network(len(input_index))
        
        #print("data -> {}".format(pos))
        b1 = 0.35  #bias 1
        b2 = 0.60  #bias 2
    
        data = np.array(data)
        expected_output = np.array(expected_output)
    
        #print("data -> {}".format(data[0]))
        #print("output -> {}".format(list(zip(data, expected_output))))
    
        data = np.array(list(zip(data, expected_output)))
        network = feed_forward_and_back_propagation(network, data, input_index, b1, b2)
        write_data(model_file, network, b1, b2, input_index)
    else:    
        (network, b1, b2, input_index) = read_network(model_file) 
        he_classify(network, b1, b2, data, pos, img, input_index, "output.txt")
    #pass
    
    