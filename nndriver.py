import time
import sys
import numpy as np
from nn import NN
from csvreader import readcsv

def main():

    patternfile = 'rd.csv'
    targetfile = 'trd.csv'

    #patternfile = 'pattern.csv'
    #argetfile = 'target.csv'

    patterns, targets = readcsv(patternfile, targetfile)

    number_patterns = len(patterns)
    pattern_length = len(patterns[0])
    number_targets = len(targets)
    target_length = len(targets[0])

    errormsg = "Unequal number of patterns and targets"
    assert number_patterns==number_targets, errormsg

    number_iterations = 1000
    number_hidden_nodes = 15
	
    nn = NN(patterns,
            targets,
            number_patterns,
            pattern_length,
            number_hidden_nodes,
            target_length)

    #np.random.seed(12345)
    np.random.seed(int(time.time()))
    nn.beta = 1
    nn.eta = 0.05

    nn.init_weights_random(-1.0,1.0)

# compute error and gradient
    for i in range(number_iterations):
        #print("Inputs: ", nn.inputnodes)
        #print("Iteration: ", i)

        #nn.print_weight_matrix(nn.w1, "Hidden Weight Matrix:")
        #print("Hidden Nodes:\n", nn.hiddennodes)
        #print("Activated Hidden Nodes:\n", nn.ghiddennodes)
        
        
        #nn.print_weight_matrix(nn.w2, "Output Weight Matrix:")
        #print("Output Nodes:\n",nn.outputnodes)
        #print("Activated Output  Nodes:\n", nn.goutputnodes)

        nn.total_sq_error_and_gradient()
        nn.update_weights()
        
        #print("Targets: ", nn.targets)
        #print("Absolute Errors:", nn.errors)
        print(i,",",nn.total_error)
        print(i,",",nn.total_error,file=sys.stderr)

    #compute outputs for input patterns
    nn.compute_nn_outputs()

    for i in range(number_patterns):
        print(nn.nn_outputs[i],",",nn.input_targets[i])

    num_close_zero = 0
    num_far_zero = 0
    num_false_zero = 0
    num_close_one = 0
    num_far_one = 0
    num_false_one = 0
    for i in range(0, 50):
        if (nn.nn_outputs[i] > 0.5):
            num_false_zero += 1
        elif (nn.nn_outputs[i] < 0.05):
            num_close_zero += 1
        else:
            num_far_zero += 1

    for i in range(50, 100):
        if (nn.nn_outputs[i] < 0.5):
            num_false_one += 1
        elif (nn.nn_outputs[i] > 0.95):
            num_close_one += 1
        else:
            num_far_one += 1
           
    print("Zeros results: ", num_close_zero, "/", num_far_zero, "/", num_false_zero)
    print("Ones results: ", num_close_one, "/", num_far_one, "/", num_false_one)

if __name__== "__main__":
    main()