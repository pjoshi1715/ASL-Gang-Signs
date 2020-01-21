# Import data set from input data provided by an NYU Professor and Importing Tensorflow
import input_data
import tensorflow.compat.v1 as tf #needed to fix an error with the placeholder attribute that was in V.1
tf.disable_v2_behavior() 
mnist = input_data.read_data_sets("data/", one_hot=True)#type of data set

# Set parameters for the training of the bot
learning_rate = 0.01 #value found on internet for optimal training
training_iteration = 30 #how many times we want to run
batch_size = 100 # size of the batch
display_step = 2 # the increase of the print every time
# TF graph input
x = tf.placeholder("float", [None, 784]) # based on the size of data given by the data set
y = tf.placeholder("float", [None, 10]) # 10 classes for each of the digits

# Now we need to create a model

# Set the weights of the model
W = tf.Variable(tf.zeros([784, 10])) # set weight
b = tf.Variable(tf.zeros([10])) # set bias





with tf.name_scope("Wx_b") as scope: # creating scope
    # Construct a linear model
    model = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax
    
# Use the attribute summary to collect data for weights and biases
w_h = tf.summary.histogram("weights", W)
b_h = tf.summary.histogram("biases", b)

# More name scopes will clean up graph representation
with tf.name_scope("cost_function") as scope: # scope
    # Minimize error using cross entropy
    # Cross Entropy is the statistical way of saying pass or fail. The image has the number or doesn't
    # Cross entropy
    cost_function = -tf.reduce_sum(y*tf.log(model))
    # Create a summary to monitor the cost function
    tf.summary.scalar("cost_function", cost_function)
    #visualization

with tf.name_scope("train") as scope: # scope
    # Gradient descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)# optimize the program during training
    #attribute from tensorflow
    # used to make program better
# Initializing the variables
init = tf.initialize_all_variables()

# Merge all summaries into a single operator
merged_summary_op = tf.summary.merge_all()
# this can be viewed later in TensorBoard






# Launch the graph and run for a test and training
with tf.Session() as sess:
    sess.run(init)

    
    
    # Change this to a location on your computer
    summary_writer = tf.summary.FileWriter('data/logs', graph_def=sess.graph_def)

    # Training cycle from scope
    for iteration in range(training_iteration):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # Compute the average loss
            avg_cost += sess.run(cost_function, feed_dict={x: batch_xs, y: batch_ys})/total_batch
            # Write logs for each iteration
            summary_str = sess.run(merged_summary_op, feed_dict={x: batch_xs, y: batch_ys})
            summary_writer.add_summary(summary_str, iteration*total_batch + i)
        # Display logs per iteration step
        if iteration % display_step == 0:
            print("Iteration:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(avg_cost))

    print("Tuning completed!")

    # Test the model
    predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))