# Commented out IPython magic to ensure Python compatibility.
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
import tensorflow_datasets as tfds

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://10.240.1.2:8470')
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)

def create_model():
  return tf.keras.Sequential(
      [tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
       tf.keras.layers.Flatten(),
       tf.keras.layers.Dense(128, activation='relu'),
       tf.keras.layers.Dense(10)])


def process(image, label):
  image = tf.cast(image, tf.float32)
  image = image/255.
  return image, label

def get_dataset(batch_size=2048):
  mnist = tfds.builder('mnist', data_dir='gs://arabert-mnist-gs/')
  mnist.download_and_prepare()
  mnist_train, mnist_test = mnist.as_dataset(split=['train', 'test'], as_supervised=True)
  print(mnist_train)

  train_dataset = mnist_train.map(process).shuffle(10000).batch(batch_size, drop_remainder=True)
  test_dataset = mnist_test.map(process).batch(batch_size, drop_remainder=True)

  return train_dataset, test_dataset

strategy = tf.distribute.experimental.TPUStrategy(resolver)
with strategy.scope():
  model = create_model()
  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['sparse_categorical_accuracy'])
  
train_dataset, test_dataset = get_dataset()

model.fit(train_dataset,
          epochs=5,
          validation_data=test_dataset)
