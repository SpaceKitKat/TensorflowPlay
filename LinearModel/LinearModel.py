#!/usr/bin
import tensorflow as tf
# 
# Linear model
#

# create one -> generates, initializes and returns a linear model
# train one -> takes a linear model, cost function, and some data and optimizes its parameters
# get data -> loads some data into a structure and returns it
# run one -> takes model and inputs then prints outputs					 

c_FLOAT_TYPE = tf.float32
Session = tf.Session()

class LinearModel:
	w = None
	x = None
	b = None
	isInitialized = False
	def __init__(self, weight, bias):
		print("--->Creating linear model")
		# Use float type to conserve memory
		self.w = tf.Variable([weight * 1.0], dtype=tf.float32)
		self.b = tf.Variable([bias * 1.0], dtype=tf.float32)
		self.x = tf.placeholder(dtype=tf.float32)

		# Initialize variables
		Session.run(tf.global_variables_initializer())
		self.isInitialized = True
		print("--->Created and initialized model.")

	def TrainLinearModel(self, costFunc, data):
		print("--->Training linear model")
		return None

	def RunLinearModel(self, arrInputs):
		print("--->Running linear model")
		# Ensure model and inputs are valid before running
		if (len(arrInputs) < 1):
			print("Insufficient inputs; length must be greater than 1.")
			raise Exception
		if (not self.isInitialized):
			print("Model is uninitialized...")
			raise Exception

		linearModel = self.w * self.x + self.b
		print(Session.run(linearModel, { self.x: arrInputs }))

def main():
	print("<<< Start main >>>\n")

	identityLinearModel = LinearModel(1, 0)
	identityLinearModel.RunLinearModel([0, 2, 3, 5, 7])

	print("<<< End main >>>\n")
	
if (__name__ == "__main__"):
	main()