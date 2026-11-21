from math import log
import numpy as np
import matplotlib.pyplot as plt 

#Activation Function and Derivative
def sigmoid(x): return 1 / (1 + np.e**(-1*x))
def sigmoidPrime(x): return np.e**(-1*x) / ((1 + np.e**(-1*x))**2)
def antiSigmoid(x): return  np.log(x/(1-x))

def lossFunction(y, y_hat): return 1/2 * np.sum((y_hat-y)**2)
def lossPrime(y,y_hat): return y_hat-y

def softmax(logits): return np.exp(logits)/np.sum(np.exp(logits))

def softmaxJacobian(logits):
    jacobian = np.zeros((len(logits), len(logits)))
    
    s = np.sum(np.exp(logits))

    for i in range(len(logits)):
        
        for j in range(len(logits)):
            if i==j:
                jacobian[i][j] = np.exp(logits[i])/s - np.exp(2*logits[i])/(s**2)
            else:
                jacobian[i][j] = np.exp(logits[i]) * np.exp(logits[j]) * -1/(s**2)
    
    return jacobian




def initialize():
    weights = [ np.random.randn(184,784) * np.sqrt(1/784), np.random.randn(43,184)*np.sqrt(1/184), np.random.randn(10,43)*np.sqrt(1/43) ]
    biases = [ np.zeros(184), np.zeros(43), np.zeros(10) ]
    
    return weights, biases


def extractTrainingData():


    trainingFile = open("/Users/ademII/Documents/GitHub/ML-Team-F26/Lab1/mnist_train.csv", "r")

    lines = [ [int(num) for num in  line.split(",")] for line in trainingFile.readlines()]

    trainingFile.close()

    
    y = []
    for line in lines:
        addend = [0] * 10
        label = int(round(line[0]))
        addend[label] = 1
        y.append(addend)

        

    x = [ np.array(line[1:], dtype=float)/255 for line in lines]

    return x,y


def extractTestData():
    testFile = open("/Users/ademII/Documents/GitHub/ML-Team-F26/Lab1/mnist_test.csv", "r")
    
    lines = [ [int(num) for num in  line.split(",")] for line in testFile.readlines()]
    
    testFile.close()
    
        
    y = []
    for line in lines:
        addend = [0] * 10
        label = int(round(line[0]))
        addend[label] = 1
        y.append(addend)
    
            
    
    x = [ np.array(line[1:], dtype=float)/255 for line in lines]
    
    return x,y
    










#TODO A feed forward of the network where A_vec is the activation function, weights is a list of all the weight matrices, biases is a list of all the bias vectors, and inp is the input, return the output as a vector
def forward(A_vec, weightsList, biasesList, inp):


    for i in range(len(weightsList)-1):
        inp = A_vec(weightsList[i]@inp + biasesList[i])

    inp = softmax(weightsList[-1]@inp +biasesList[-1] )

    return inp

def cachedForward(A_vec, weightsList,biasesList,inp):

    preActivations = [inp]
    activations = [inp]

    for i in range(len(weightsList)-1):
        
        inp = weightsList[i]@inp + biasesList[i]
        preActivations.append(inp)

        inp = A_vec(inp)
        activations.append(inp)

    
    inp = (weightsList[-1]@inp +biasesList[-1] )
    preActivations.append(inp)

    inp = softmax(inp)
    activations.append(inp)
    
    return preActivations, activations,inp

def testForward(A_vec, weightsList, biasesList,inp, y,):

    for i in range(len(weightsList)-1):
            inp = A_vec(weightsList[i]@inp + biasesList[i])
    
    inp = softmax(weightsList[-1]@inp +biasesList[-1] )

    
    
    return lossFunction(y,inp)



def backprop( weightsList,biasesList, activations, preActivations ,  y, y_hat, lr):

    loss = lossFunction(y,y_hat)

     
    
    weightGrads=[None] * len(weightsList)
    activationGrads = [None]* len(weightsList)
    biasGrads=[None] * len(biasesList) 

    activationGrads[-1] = softmaxJacobian(preActivations[-1])@lossPrime(y,y_hat)


    for i in reversed(range(len(weightsList))):
        
        weightGrads[i] = np.outer(activationGrads[i],activations[i])
        biasGrads[i]=(activationGrads[i])

        if i>0:
            activationGrads[i-1]=(sigmoidPrime(preActivations[i])* (weightsList[i].T@activationGrads[i]))

    for i in range(len(weightsList)):

        weightsList[i] -= lr*weightGrads[i]
        biasesList[i]-= lr*biasGrads[i]

    return weightsList,biasesList,loss

        




#TODO This is where you back propogate by calculating the deltas and updating the weights and biases, try different learning rates and see what works
def one_epoch(inp,y, weights, biases, lr, A_vec=sigmoid):


    
    
    preActivations, activations, y_hat = cachedForward(A_vec, weights, biases,inp )
    weights, biases,loss = backprop(weights,biases,activations,preActivations,y,y_hat,lr )

    return weights, biases,loss


def one_cycle(weights, biases , trainingX, trainingY):

    

    for i in range(len(trainingX)):
       weights,biases,loss = one_epoch(trainingX[i], trainingY[i], weights, biases, 0.01)
       
    return weights,biases,loss

def testCycle(weights, biases, testX, testY):
    total = 0 

    for i in range(len(testX)):
        total += testForward(sigmoid, weights, biases, testX[i],testY[i])

    total/=len(testX)
    
    return total
        





def saveParams(weights, biases):

    weights0 = open("weights0.csv",'w')

    for i in range(len(weights[0])):
        for j in range(len(weights[0][i])):
            weights0.write(str(weights[0][i][j]))
            weights0.write(',')
        weights0.write('\n')

    weights0.close()



    weights1 = open("weights1.csv",'w')

    for i in range(len(weights[1])):
        for j in range(len(weights[1][i])):
            weights1.write(str(weights[1][i][j]))
            weights1.write(',')
        weights1.write('\n')

    weights1.close()

    weights2 = open("weights2.csv",'w')
    
    for i in range(len(weights[2])):
        for j in range(len(weights[2][i])):
            weights2.write(str(weights[2][i][j]))
            weights2.write(',')
        weights2.write('\n')
    
    weights2.close()

    biasesFile = open("biases.csv",'w')

    for i in range(len(biases)):
        for j in range(len(biases[i])):
            biasesFile.write(str(biases[i][j]))
            biasesFile.write(',')
        biasesFile.write('\n')

    biasesFile.close()



print("program start")

weights, biases = initialize()
print("parameters initialized")


trainingX, trainingY = extractTrainingData()
testX, testY = extractTestData()
print("vectors extracted")


trainingLosses=[]
testLosses=[]


print("training started")
for i in range(50):

    weights, biases,loss = one_cycle(weights, biases, trainingX, trainingY)
    
    trainingLosses.append(loss)
    testLosses.append(testCycle(weights,biases,testX,testY))



print("training finished")
print(*trainingLosses)
print(*testLosses)

figure, axes = plt.subplots(figsize=(8,5))
axes.plot(trainingLosses, label="training loss")
axes.plot(testLosses, label="test loss")
axes.set_xlabel("Cycle")
axes.set_ylabel("Loss")
axes.set_title("training vs. test loss")
axes.grid=True
axes.legend()
plt.show()


saveParams(weights,biases)
print("parameters saved")
