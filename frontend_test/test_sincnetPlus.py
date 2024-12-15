import leaf_audio.frontend as frontend
import tensorflow as tf
import tensorflow_datasets as tfds

# Set a global seed for TensorFlow to ensure reproducibility
tf.random.set_seed(42)

# Set the GPU memory configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load the dataset with a seed for shuffling
train_ds = tfds.load('speech_commands', split='train', shuffle_files=True, as_supervised=True, read_config=tfds.ReadConfig(shuffle_seed=42))
val_ds = tfds.load('speech_commands', split='validation', shuffle_files=True, as_supervised=True, read_config=tfds.ReadConfig(shuffle_seed=42))
test_ds = tfds.load('speech_commands', split='test', shuffle_files=True, as_supervised=True, read_config=tfds.ReadConfig(shuffle_seed=42))

# Preprocess the dataset
def preprocess_dataset(ds):
    def preprocess_fn(audio, label):
        audio = tf.pad(audio, paddings=[[0, 16000 - tf.shape(audio)[0]]], mode='CONSTANT')
        return audio, label
    return ds.map(preprocess_fn)

train_ds = preprocess_dataset(train_ds)
val_ds = preprocess_dataset(val_ds)
test_ds = preprocess_dataset(test_ds)

# Neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Input((16000,)),
    tf.keras.layers.Reshape((16000, 1)),
    frontend.SincNetPlus(),
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

# Specify the batch size and number of epochs
batch_size = 256
epochs = 10

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

# Fit the model on the training data
model.fit(train_ds.batch(batch_size), validation_data=val_ds.batch(batch_size), epochs=epochs)

# Evaluate the model on the test data
model.evaluate(test_ds.batch(batch_size))