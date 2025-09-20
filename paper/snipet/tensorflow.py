import tensorflow as tf
x = tf.placeholder(dtype=tf.float32, shape=[None, 28,28])
y = tf.placeholder(dtype=tf.int32, shape=[None])

# Chuyển về vector 1 chiều
images_flat = tf.contrib.layers.flatten(x) 
# Lớp fully connected
logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits))
# Cấu hình hàm tối ưu
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss) 
correct_pred = tf.argmax(logits, 1)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# Huấn luyện với 1000 vòng lặp
for i in range(1000):
    print("EPOCH", i)
    _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x:images28, y: labels})
    print(accuracy_val)