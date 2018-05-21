import tensorflow as tf

graph = {}
graph['input'] = tf.Variable(3.0, dtype="float32")
graph['conv1_1'] = graph['input'] * tf.constant(3.0, dtype="float32")
graph['conv1_2'] = graph['conv1_1'] * graph['input'] + 5


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    val1 = sess.run(graph['conv1_1'])
    val2 = sess.run(graph['conv1_2'])

    # Update graph with new input
    sess.run(graph['input'].assign(1))
    # This will re-assign 3 to 'input'
    # sess.run(tf.global_variables_initializer())
    val3 = graph['conv1_1']
    val4 = sess.run(graph['conv1_1'])

    print("Val 1: ", val1)
    print("Val 2: ", val2)
    print("Val 3: ", val3)
    print("Val 4: ", val4)
