#!/usr/bin
import tensorflow as tf

class SimpleGraph:
	m_session = None 

	def __init__(self):
		# create a session as needed
		print("--->init simple graph\n")
		if (self.m_session is None):
			self.m_session = tf.Session()

	def computeNodes(self, arrNodes):
		# compute graph (aka, run session) 
		self.computeGraph(arrNodes)

	def addAndComputeNodes(self, arrNodes):
		print("--->adding nodes\n")
		if (len(arrNodes) >= 2):
			sumNode = tf.add(arrNodes[0], arrNodes[1])
			self.computeGraph(sumNode)
		else:
			print("--->insufficient number of nodes:", len(arrNodes))

	def computeGraph(self, arrNodes):
		print("--->computing nodes\n")
		print(self.m_session.run(arrNodes))

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
		
	print("<<< End main >>>\n")
	
if (__name__ == "__main__"):
	main()