#!/usr/bin
import tensorflow as tf

class GenericFloatAdder:
	m_operandA = None
	m_operandB = None
	def __init__(self):
		print("--->init generic adder")
		self.m_operandA = tf.placeholder(tf.float32)
		self.m_operandB = tf.placeholder(tf.float32)

	def node(self):
		if (self.m_operandA is None or self.m_operandB is None):
			print("Operands need to be defined before adding...")
			print("operandA: ", self.m_operandA)
			print("operandB: ", self.m_operandB)
			raise ValueError 
		return tf.add(self.m_operandA, self.m_operandB)

class WeightedAdder(GenericFloatAdder):
	m_weight = None

	def __init__(self, floatWeight):
		print("--->init weighted adder")
		self.m_weight = floatWeight
		super().__init__()

	# override base class
	def node(self):
		if (self.m_weight is None):
			print("Weight is uninitialized.")
			raise Exception 
		return tf.multiply(self.m_operandA + self.m_operandB, self.m_weight) 

class SimpleGraph:
	# encapsulate session object
	__m_session = None 
	__m_floatAdder = None

	def __init__(self):
		# create a session as needed
		print("--->init simple graph\n")
		if (self.__m_session is None):
			self.__m_session = tf.Session()
		self.__m_floatAdder = GenericFloatAdder()

	def computeNodes(self, arrNodes):
		# compute graph (aka, run session) 
		self.computeGraph(arrNodes)

	def addAndComputeNodes(self, arrNodes):
		print("--->adding nodes\n")
		if (len(arrNodes) < 2):
			print("--->insufficient number of nodes:", len(arrNodes))
			return

		sumNode = tf.add(arrNodes[0], arrNodes[1])
		self.computeGraph(sumNode)

	def addFloats(self, arrFloatsA, arrFloatsB):
		if (len(arrFloatsA) == 0 or len(arrFloatsB) == 0):
			print("Unable to add empty float tensors.")
			return

		# Use "feed_dict" to map inputs to place holders
		# in adder node
		print("--->adding floats\n")
		print(self.__m_session.run(
			self.__m_floatAdder.node(), 
			{ self.__m_floatAdder.m_operandA: arrFloatsA,
			  self.__m_floatAdder.m_operandB: arrFloatsB }))

	def addAndScaleFloats(self, arrFloats, floatScalar):
		if (len(arrFloats) < 2):
			print("Insufficient number of floats passed to adder.")
			return

		weightedAdder = WeightedAdder(floatScalar)
		print("--->adding and scaling floats")
		print(self.__m_session.run(
			weightedAdder.node(),
			{ weightedAdder.m_operandA: arrFloats[0],
			  weightedAdder.m_operandB: arrFloats[1] }))


	def computeGraph(self, arrNodes):
		print("--->computing nodes\n")
		print(self.__m_session.run(arrNodes))

def main():
	print("<<< Start main >>>\n")
	# create two nodes
	nodeA = tf.constant(1.0, dtype=tf.float32)
	nodeB = tf.constant(2.0)
	arrOfNodes = [nodeA, nodeB]

	# compute graph (aka, run session) 
	graph = SimpleGraph()
	graph.computeNodes(arrOfNodes)
	graph.addAndComputeNodes(arrOfNodes)
	graph.addFloats([1.5], [2.5])
	graph.addAndScaleFloats([1.0, 1.0], 5)


# Quick demos
# * Encapsulation
#	try:
#		if (graph.__msession is None):
#			print("This should fail.")
#	except:
#		print("Can't access simple graph's session obj.")
		
	print("<<< End main >>>\n")
	
if (__name__ == "__main__"):
	main()