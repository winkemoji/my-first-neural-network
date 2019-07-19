import numpy
# scipy.special for the sigmoid function expit()
import scipy.special

import matplotlib.pyplot


# neural network class definition
class NeuralNetwork:
    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):

        # set number of nodes in each input,hidden,output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # link weight matrices,wih and who
        # weights inside the arrays are w_i_j, where link is from node i th node j in the next layer
        # w11 w21
        # w12 w22 etc
        self.wih = numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who = numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))

        # learning rate
        self.lr = learningrate

        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    # train the neural network
    def train(self, inputs_list, target_list):

        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list,ndmin=2).T
        targets = numpy.array(target_list,ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)

        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)

        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # error is the (target-actual)
        output_errors = targets - final_outputs

        # hidden layer error is the output_errors,split by weights,recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors*final_outputs*(1.0-final_outputs)), numpy.transpose(hidden_outputs))

        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)), numpy.transpose(inputs))
        pass

    # query the neural network
    def query(self,inputs_list):

        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list,ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih,inputs)

        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who,hidden_outputs)

        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
    pass


# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 100
output_nodes = 10

# learning rate
learning_rate = 0.3

# create instance of neural network
n = NeuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

# load the mnist training data CSV file into a list
training_data_file = open('mnist_csv/mnist_train.csv')
training_data_list = training_data_file.readlines()
training_data_file.close()

# train the neural network
i = 1;
# go through all records in the training data set
for record in training_data_list:

    # split the record by the ',' commas
    all_values = record.split(',')

    # scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:])/255.0 * 0.99)+0.01

    # create the target output values(all 0.01,except the desired label which is 0.99)
    targets = numpy.zeros(output_nodes)+0.01

    # all_values[0] is the target label for this record
    targets[int(all_values[0])] = 0.99
    n.train(inputs,targets)
    print("train:", i)
    i += 1
    pass

# load the mnist test data CSV file into a list
test_data_file = open('mnist_csv/mnist_test.csv','r')
test_data_list = test_data_file.readlines()
test_data_file.close()


# test the neural network
# scorecard for how well the network performs, initially empty
scorecard = []
# go through all the records in the test data set for record in test_data_list:
for record in test_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    #correct answer is first value
    correct_label = int(all_values[0])
    print(correct_label,"correct label")
    #scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:])/255.0 * 0.99)+0.01
    # query the network
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    print(label,"nerual network's answer")
    # append correct or incorrect to list
    if(label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass

print(scorecard)

s = [scorecard.count(0), scorecard.count(1)]

matplotlib.pyplot.rcParams['font.sans-serif']=['SimHei']
matplotlib.pyplot.pie(s,autopct='%1.1f%%',shadow=False,startangle=150)
matplotlib.pyplot.title("正确率")
matplotlib.pyplot.show()

