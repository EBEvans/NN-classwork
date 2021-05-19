import numpy as np

class NN:

	def __init__(self, 
				input_patterns,
				input_targets,
				num_patterns,
				num_inputs, 
				num_hidden, 
				num_outputs):

		# Data for Neural Network Computations
		self.M = num_inputs # number of input nodes
		self.N = num_hidden # number of hidden nodes
		self.P = num_outputs # number of output nodes

		self.num_patterns = num_patterns # number of training examples

		self.input_patterns = input_patterns # number of input patterns
		self.input_targets = input_targets # number of targets

		self.nn_outputs = np.zeros((num_patterns,self.P)) # neural net output for each pattern 
		
		self.beta = 1 # activation function parameter
		self.eta = 0.01 # learning rate

		self.inputnodes = np.zeros(1+self.M) # plus 1 for bias node
		self.inputnodes[-1] = 1 # set bias node to 1
		self.hiddennodes = np.zeros(1+self.N) # plus 1 for bias node
		self.hiddennodes[-1] = 1 # set bias node to 1
		self.ghiddennodes = np.zeros(1+self.N) # hidden nodes after activation
		self.ghiddennodes[-1] = 1 # for hidden to output matrix multiplication
		self.outputnodes = np.zeros(self.P) # no bias node
		self.goutputnodes = np.zeros(self.P) # ouput nodes after activation

		self.targets = np.zeros(self.P) # target(s)
		self.errors = np.zeros(self.P) # error(s)
		self.square_errors = np.zeros(self.P) # square errors
		self.sum_square_errors = 0.0 # sum square errors over all outputs

		self.total_error = 0.0 # sum square errors over all patterns
		
		self.w1 = np.zeros((self.N, 1+self.M)) # input to hidden weight matrix
		self.w2 = np.zeros((self.P, 1+self.N)) # hidden to output weight matrix

		self.dw1 = np.zeros((self.N, 1+self.M)) # gradient weight matrix
		self.dw2 = np.zeros((self.P, 1+self.N)) # gradient weight matrix

		self.deltaoutput = np.zeros(self.P) # deltas at output layer
		self.deltahidden = np.zeros(1+self.N) # deltas at hidden layer

	# functions for performing backpropagation - computing error and gradient

	# copy pattern and target into input nodes and targets
	def copy_pattern_and_target_to_inputs(self, mu):
		for k in range(self.M):
			self.inputnodes[k] = self.input_patterns[mu][k]
		for i in range(self.P):
			self.targets[i] = self.input_targets[mu][i]

	# copy pattern into input nodes
	def copy_pattern_to_inputs(self, mu):
		for k in range(self.M):
			self.inputnodes[k] = self.input_patterns[mu][k]

	# copy nn output to outputs
	def copy_nnoutput_to_output(self, mu):
		for i in range(self.P):
			self.nn_outputs[mu][i] = self.goutputnodes[i]

	# hidden nodes = w1 weights * input nodes 
	def w1_matrix_vector_multiplication(self):
		for j in range(self.N):
			s = 0
			for k in range(1+self.M):
				s += self.w1[j][k]*self.inputnodes[k]
			self.hiddennodes[j] = s

	# ghidden nodes = activation(hidden nodes)
	def hidden_activation(self):
		for j in range(self.N):
			self.ghiddennodes[j] = self.activation(self.hiddennodes[j], self.beta )

	# output nodes = w2 wights * ghidden nodes
	def w2_matrix_vector_multiplication(self):
		for i in range(self.P):
			s = 0
			for j in range(1+self.N):
				s += self.w2[i][j]*self.ghiddennodes[j]
			self.outputnodes[i] = s

	# goutput nodes = activation(output notes)
	def output_activation(self):
		for i in range(self.P):
			self.goutputnodes[i] = self.activation(self.outputnodes[i], self.beta )

	# actiovation function
	def activation(self,h,beta):
		return 1.0/(1.0+np.exp(-beta*h))

	# errors = targets - goutput nodes
	def compute_errors(self):
		for i in range(self.P):
			self.errors[i] = self.targets[i]-self.goutputnodes[i]

	# square errors = errors * errors
	def compute_square_errors(self):
		for i in range(self.P):
			self.square_errors[i] = self.errors[i]*self.errors[i]

	# sum_square_errors = sum(square errors) (over all output nodes)
	def compute_sum_square_errors(self):
		self.sum_square_errors = 0.0
		for i in range(self.P):
			self.sum_square_errors += self.square_errors[i]

	# sum square error of all patterns
	def accumulate_total_error(self):
		self.total_error += self.sum_square_errors

	# compute deltas at output layer
	def compute_output_deltas(self):
		for i in range(self.P):
			self.deltaoutput[i] = self.beta * self.goutputnodes[i] \
			* (1.0-self.goutputnodes[i]) * self.errors[i]

	# compute deltas at hidden layer
	def compute_hidden_deltas(self):
		for j in range(1+self.N):
			tmpsum = 0.0
			for i in range(self.P):
				tmpsum += self.w2[i][j] * self.deltaoutput[i]
			self.deltahidden[j] = self.beta * self.ghiddennodes[j] \
			* (1.0-self.ghiddennodes[j]) * tmpsum

	# sum gradient computation over all patterns
	def accumulate_gradient(self):
		for i in range(self.P):
			for j in range(1+self.N):
				self.dw2[i][j] -= self.deltaoutput[i]*self.ghiddennodes[j]
		for j in range(self.N):
			for k in range(1+self.M):
				self.dw1[j][k] -= self.deltahidden[j]*self.inputnodes[k]

	# calculate total error
	def total_sq_error(self):
		self.total_error = 0.0
		for mu in range(self.num_patterns):
			self.copy_pattern_and_target_to_inputs(mu)
			self.w1_matrix_vector_multiplication()
			self.hidden_activation()
			self.w2_matrix_vector_multiplication()
			self.output_activation()
			self.compute_errors() 
			self.compute_square_errors()
			self.compute_sum_square_errors()
			self.accumulate_total_error()

	# calculate total error and its gradient
	def total_sq_error_and_gradient(self):
		self.total_error = 0.0
		for i in range(self.P):
			for j in range(1+self.N):
				self.dw2[i][j] = 0.0
		for j in range(self.N):
			for k in range(1+self.M):
				self.dw1[j][k] = 0.0		

		for mu in range(self.num_patterns):
			self.copy_pattern_and_target_to_inputs(mu)
			self.w1_matrix_vector_multiplication()
			self.hidden_activation()
			self.w2_matrix_vector_multiplication()
			self.output_activation()
			self.compute_errors()
			self.compute_square_errors()
			self.compute_sum_square_errors()
			self.accumulate_total_error()
			self.compute_output_deltas()
			self.compute_hidden_deltas()
			self.accumulate_gradient()

	def compute_nn_outputs(self):
		for mu in range(self.num_patterns):
			self.copy_pattern_and_target_to_inputs(mu)
			self.w1_matrix_vector_multiplication()
			self.hidden_activation()
			self.w2_matrix_vector_multiplication()
			self.output_activation()
			self.copy_nnoutput_to_output(mu)

	def update_weights(self):
		for j in range(self.N):
			for k in range(1+self.M):
				self.w1[j][k] = self.w1[j][k] - self.eta*self.dw1[j][k]
		for i in range(self.P):
			for j in range(1+self.N):
				self.w2[i][j] = self.w2[i][j] - self.eta*self.dw2[i][j]

	def init_weights_random(self, a, b):
		for j in range(self.N):
			for k in range(1+self.M):
				self.w1[j][k] = (b-a)*np.random.random_sample() + a
		for i in range(self.P):
			for j in range(1+self.N):
				self.w2[i][j] = (b-a)*np.random.random_sample() + a

	def init_weights_one(self):
		for j in range(self.N):
			for k in range(1+self.M):
				self.w1[j][k] = 1
		for i in range(self.P):
			for j in range(1+self.N):
				self.w2[i][j] = 1

	def print_array(self, a, str):
		print(str)
		for i in range(a.size):
			print(a[i],end =" ")
		print("")

	def print_weight_matrix(self, w, str):
		print(str)
		nrows, ncols = w.shape
		for i in range(nrows):
			for j in range(ncols):
				print(w[i][j],end=" ")
			print("")