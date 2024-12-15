
import leaf_audio.frontend as frontend

import tensorflow as tf
import tensorflow_datasets as tfds

#set tf don't use all GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)



# Load the dataset
train_ds = tfds.load('speech_commands', split='train', shuffle_files=True)
val_ds = tfds.load('speech_commands', split='validation', shuffle_files=True)
test_ds = tfds.load('speech_commands', split='test', shuffle_files=True)

# preprocess the dataset
def preprocess_dataset(ds):
    def preprocess_fn(data):
        audio = data['audio']
        label = data['label']
        audio = tf.pad(audio, paddings=[[0, 16000-tf.shape(audio)[0]]], mode='CONSTANT')
        return audio, label
    return ds.map(preprocess_fn)

train_ds = preprocess_dataset(train_ds)
val_ds = preprocess_dataset(val_ds)
test_ds = preprocess_dataset(test_ds)


# Create the model by tf.keras.Sequential
# model = tf.keras.Sequential([
#     tf.keras.layers.Input((16000,)),
#     tf.keras.layers.Reshape((16000, 1)),
#     frontend.Leaf(),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dense(12, activation='softmax')

# ])

# cnn model
model= tf.keras.Sequential([
    tf.keras.layers.Input((16000,)),
    tf.keras.layers.Reshape((16000, 1)),
    frontend.SincNet(),
    tf.keras.layers.Conv1D(32, 3, activation='relu'),
    tf.keras.layers.Dropout(0.18),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(64, 3, activation='relu'),
    tf.keras.layers.Dropout(0.18),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(128, 3, activation='relu'),
    tf.keras.layers.Dropout(0.18),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(420, activation='relu'),
    tf.keras.layers.Dropout(0.18),
    tf.keras.layers.Dense(12, activation='softmax')
])

model.summary()


batch_size = 256
epochs=3

model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['sparse_categorical_accuracy'])

model.fit(train_ds.batch(batch_size), validation_data=val_ds.batch(batch_size), epochs=epochs)

model.evaluate(test_ds.batch(batch_size))

