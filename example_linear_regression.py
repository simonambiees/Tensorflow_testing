import tensorflow as tf

# hello = tf.constant("Hello! Tensorflow")

session = tf.Session()

# print(session.run(hello))

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)  # implicitly float32
node3 = tf.add(node1, node2)

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # a short cut for tf.add(a,b)
# print(session.run(adder_node, feed_dict={a: [1, 3], b: [2, 4]}))
# print(session.run(adder_node, feed_dict={a: [1, 3], b: [2, 4]})[0])

"""
# BASIC IDEA!!!!!!!!!
y_model = tf.multiply(X, w)
cost = tf.square(Y = y_model)
train_op =- tf.train.GradientDescentOptimizer(0.01).minimize(cost)
"""

# Build graph using TF operations
# X and Y data
# x_train = [1, 2, 3]
# y_train = [1, 2, 3]


# Using placeholder X and Y to input
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")
# Our hypothesis XW+b
hypothesis = X * W + b
# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(cost)
# launch the graph in a session
sess = tf.Session()
# Initializes global variables in the graph
sess.run(tf.global_variables_initializer())
# Fit the line
while True:
    step = 1
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={X: [1, 2, 3, 4, 5], Y: [7, 12, 17, 22, 27]})
    if step % 20 == 0:
        print(step, "   Cost: ", cost_val, "    Weight: ", W_val, " Bias: ", b_val)
        step += 1

print(sess.run(hypothesis, feed_dict={X: [5]}))
print(sess.run(hypothesis, feed_dict={X: [10]}))
print(sess.run(hypothesis, feed_dict={X: [20, 30]}))
