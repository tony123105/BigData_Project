import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras import layers
import librosa
from leaf_audio import frontend

# Add at the beginning of your file, after imports
def set_random_seed(seed=42):
    """Set random seed for reproducibility"""
    # TensorFlow seed
    tf.random.set_seed(seed)
    
    # Python random seed
    import random
    random.seed(seed)
    
    # NumPy seed
    np.random.seed(seed)
    
    # Set TensorFlow session behavior
    tf.keras.utils.set_random_seed(seed)
    
    # Configure GPU behavior for deterministic operations
    tf.config.experimental.enable_op_determinism()

# Call this function before any model or data operations
set_random_seed()

# Define constants
WORDS = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']
SILENCE = '_silence_'
UNKNOWN = '_unknown_'
BACKGROUND_NOISE = '_background_noise_'
SAMPLE_RATE = 16000

# 定义预处理方法
class AudioPreprocessing:
    def __init__(self, sample_rate=16000, frame_length=400, frame_step=160):
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.frame_step = frame_step
        
        # Initialize LEAF frontend components
        self.leaf = frontend.Leaf()
        
        self.melfbanks = frontend.MelFilterbanks()
        self.tfbanks = frontend.TimeDomainFilterbanks()
        self.sincnet = frontend.SincNet()
        self.sincnet_plus = frontend.SincNetPlus()

    def stft_preprocessing(self, audio):
        # STFT preprocessing
        stft = tf.signal.stft(
            audio,
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            fft_length=self.frame_length
        )
        # Use magnitude to avoid complex to float casting
        spectrogram = tf.abs(stft)
        return tf.math.log(spectrogram + 1e-6)

    def conv1d_preprocessing(self, audio):
        # con1d for preprocessing
        audio = tf.expand_dims(audio, axis=-1)
        conv_layer = layers.Conv1D(64, kernel_size=80, strides=4)
        return conv_layer(audio)

    def sinc_preprocessing(self, audio):
        # SincNet for preprocessing
        def sinc(x):
            return tf.where(
                tf.equal(x, 0),
                tf.ones_like(x),
                tf.sin(x) / x
            )

        t = tf.range(-(self.frame_length//2), self.frame_length//2, dtype=tf.float32)
        window = 0.54 - 0.46 * tf.cos(2 * np.pi * t / self.frame_length)
        
        # 1 set of frequency filters
        low_freqs = tf.range(1, 41) * 50.0  # 40 different frequency filters
        high_freqs = low_freqs + 50.0
        
        filters = []
        for low, high in zip(low_freqs, high_freqs):
            band_pass = 2 * (sinc(2 * np.pi * high * t) - sinc(2 * np.pi * low * t))
            band_pass = band_pass * window
            filters.append(band_pass)
            
        filters = tf.stack(filters)
        filters = tf.expand_dims(filters, axis=-1)
        
        audio = tf.expand_dims(audio, axis=-1)
        return tf.nn.conv1d(audio, filters, stride=4, padding='SAME')

    def gabor_preprocessing(self, audio):
        # Gabor preprocessing
        def gabor_kernel(kernel_size, sigma, frequency):
            t = tf.range(-(kernel_size//2), kernel_size//2, dtype=tf.float32)
            gaussian = tf.exp(-(t**2)/(2*sigma**2))
            sinusoid = tf.cos(2*np.pi*frequency*t)
            return gaussian * sinusoid

        kernels = []
        for freq in np.linspace(20, 4000, 40):  # 40 different frequency kernel
            kernel = gabor_kernel(self.frame_length, self.frame_length/6, freq/self.sample_rate)
            kernels.append(kernel)
            
        kernels = tf.stack(kernels)
        kernels = tf.expand_dims(kernels, axis=-1)
        
        audio = tf.expand_dims(audio, axis=-1)
        return tf.nn.conv1d(audio, kernels, stride=4, padding='SAME')

    def leaf_preprocessing(self, audio):
        # LEAF preprocessing
        audio = tf.expand_dims(audio, axis=0)  # Add batch dimension
        return self.leaf(audio)

    def melfilterbanks_preprocessing(self, audio):
        # MelFilterbanks preprocessing
        audio = tf.expand_dims(audio, axis=0)
        return self.melfbanks(audio)

    def timedomainfilterbanks_preprocessing(self, audio):
        # TimeDomainFilterbanks preprocessing
        audio = tf.expand_dims(audio, axis=0)
        return self.tfbanks(audio)

    def sincnet_preprocessing(self, audio):
        # SincNet preprocessing from LEAF
        audio = tf.expand_dims(audio, axis=0)
        return self.sincnet(audio)

    def sincnet_plus_preprocessing(self, audio):
        # SincNetPlus preprocessing
        audio = tf.expand_dims(audio, axis=0)
        return self.sincnet_plus(audio)

# Transformer model denfine
class TransformerModel(tf.keras.Model):
    def __init__(self, num_classes, d_model=256, num_heads=8, num_layers=4, dropout_rate=0.1):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        
        # Input embedding and normalization
        self.feature_layer = layers.Dense(d_model)
        self.input_dropout = layers.Dropout(dropout_rate)
        self.input_norm = layers.LayerNormalization(epsilon=1e-6)
        
        # Transformer encoder layers
        self.encoder_layers = []
        for _ in range(num_layers):
            self.encoder_layers.append({
                'attention': layers.MultiHeadAttention(
                    num_heads=num_heads, 
                    key_dim=d_model//num_heads,
                    dropout=dropout_rate
                ),
                'attention_norm': layers.LayerNormalization(epsilon=1e-6),
                'ffn_1': layers.Dense(d_model * 4, activation='ReLU'),
                'ffn_2': layers.Dense(d_model),
                'ffn_norm': layers.LayerNormalization(epsilon=1e-6),
                'dropout_1': layers.Dropout(dropout_rate),
                'dropout_2': layers.Dropout(dropout_rate)
            })
        
        # Output layers
        self.avg_pool = layers.GlobalAveragePooling1D()
        self.output_dropout = layers.Dropout(dropout_rate)
        self.output_layer = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        # Input embedding
        x = self.feature_layer(inputs)
        x = self.input_dropout(x, training=training)
        x = self.input_norm(x)
        
        # Position encoding
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        positions = tf.cast(positions, tf.float32)
        pos_encoding = self.positional_encoding(positions, self.d_model)
        x = x + pos_encoding
        
        # Transformer encoder blocks
        for encoder in self.encoder_layers:
            # Multi-head attention
            attn = encoder['attention'](x, x)
            attn = encoder['dropout_1'](attn, training=training)
            x = encoder['attention_norm'](x + attn)
            
            # Feed forward network
            ffn = encoder['ffn_1'](x)
            ffn = encoder['ffn_2'](ffn)
            ffn = encoder['dropout_2'](ffn, training=training)
            x = encoder['ffn_norm'](x + ffn)
        
        # Output
        x = self.avg_pool(x)
        x = self.output_dropout(x, training=training)
        return self.output_layer(x)
        
    def positional_encoding(self, position, d_model):
        # Convert to float32 before calculations
        position = tf.cast(position, tf.float32)
        d_model_float = tf.cast(d_model, tf.float32)
        
        # Create the angles for the positional encoding
        angles = tf.range(0, d_model, dtype=tf.float32)
        angles = angles / tf.pow(10000.0, (2 * (angles // 2)) / d_model_float)
        
        # Apply sin to even indices and cos to odd indices
        pos_encoding = tf.expand_dims(position, -1) * angles
        pos_encoding = tf.stack([
            tf.sin(pos_encoding[:, 0::2]),
            tf.cos(pos_encoding[:, 1::2])
        ], axis=-1)
        
        # Reshape and return
        pos_encoding = tf.reshape(pos_encoding, [tf.shape(position)[0], d_model])
        return tf.expand_dims(pos_encoding, 0)

# training
def train_model(model, train_dataset, validation_dataset, preprocessing_method, epochs=50):
    # Get preprocessing method name
    if hasattr(preprocessing_method, '__name__'):
        preprocess_name = preprocessing_method.__name__
    else:
        preprocess_name = preprocessing_method.__class__.__name__
    print(f"\nTraining with preprocessing method: {preprocess_name}")
    
    # Learning rate schedule
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate,
        first_decay_steps=1000,
        t_mul=2.0,
        m_mul=0.9,
        alpha=0.1
    )
    
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=0.01
    )
    
    # Use regular cross entropy without label smoothing
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    
    # Early stopping parameters
    patience = 10
    min_delta = 0.001
    patience_counter = 0
    best_val_accuracy = 0
    best_epoch = 0
    best_weights = None
    
    # Metrics
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            features = preprocessing_method(x)
            predictions = model(features, training=True)
            loss = loss_fn(y, predictions)
            
            # Add L2 regularization
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables 
                               if 'bias' not in v.name]) * 0.01
            loss += l2_loss
        
        gradients = tape.gradient(loss, model.trainable_variables)
        # Gradient clipping
        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        train_accuracy.update_state(y, predictions)
        train_loss.update_state(loss)
        return loss
    
    @tf.function
    def val_step(x, y):
        features = preprocessing_method(x)
        predictions = model(features, training=False)
        loss = loss_fn(y, predictions)
        
        val_accuracy.update_state(y, predictions)
        val_loss.update_state(loss)
    
    # Calculate total steps
    total_train_steps = sum(1 for _ in train_dataset)
    total_val_steps = sum(1 for _ in validation_dataset)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Reset metrics
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()
        
        # Training progress bar
        progbar = tf.keras.utils.Progbar(
            total_train_steps,
            stateful_metrics=['loss', 'accuracy']
        )
        
        # Training loop
        for i, (x_batch, y_batch) in enumerate(train_dataset):
            loss = train_step(x_batch, y_batch)
            # Update progress bar with metrics
            progbar.update(
                i + 1,
                values=[
                    ('loss', train_loss.result()),
                    ('accuracy', train_accuracy.result())
                ]
            )
        
        # Validation progress bar
        print("\nValidation:")
        val_progbar = tf.keras.utils.Progbar(
            total_val_steps,
            stateful_metrics=['val_loss', 'val_accuracy']
        )
        
        # Validation loop
        for i, (x_batch, y_batch) in enumerate(validation_dataset):
            val_step(x_batch, y_batch)
            # Update validation progress bar
            val_progbar.update(
                i + 1,
                values=[
                    ('val_loss', val_loss.result()),
                    ('val_accuracy', val_accuracy.result())
                ]
            )
        
        # Early stopping check
        current_val_accuracy = val_accuracy.result()
        if current_val_accuracy > best_val_accuracy + min_delta:
            best_val_accuracy = current_val_accuracy
            best_epoch = epoch + 1
            best_weights = model.get_weights()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after epoch {epoch + 1}")
            # Restore best weights
            model.set_weights(best_weights)
            break
    
    # Final summary and results
    print("\n" + "="*50)
    print(f"Training Summary for {preprocess_name}")
    print("="*50)
    print(f"Best validation accuracy: {best_val_accuracy:.4f} (Epoch {best_epoch})")
    print(f"Final training accuracy: {train_accuracy.result():.4f}")
    print(f"Final validation accuracy: {val_accuracy.result():.4f}")
    print("="*50)
    
    return {
        'preprocessing_method': preprocess_name,
        'best_val_accuracy': best_val_accuracy,
        'best_epoch': best_epoch,
        'final_train_accuracy': train_accuracy.result(),
        'final_val_accuracy': val_accuracy.result()
    }

def preprocess_dataset(ds, target_length=16000):
    def preprocess_fn(audio, label):
        # Convert audio to float32 and normalize
        audio = tf.cast(audio, tf.float32)
        audio = audio / tf.int16.max  # Normalize from int16 to float32 range
        
        # Pad or truncate to target length
        current_length = tf.shape(audio)[0]
        if current_length < target_length:
            # Pad with zeros if shorter
            padding = [[0, target_length - current_length]]
            audio = tf.pad(audio, padding, mode='CONSTANT')
        else:
            # Truncate if longer
            audio = audio[:target_length]
            
        return audio, label

    # Apply preprocessing
    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return ds

def prepare_dataset(dataset, batch_size=32, shuffle_buffer_size=10000):
    # First apply preprocessing to ensure consistent lengths
    dataset = preprocess_dataset(dataset)
    
    # Cache the dataset for better performance
    dataset = dataset.cache()
    
    # Use deterministic shuffling
    dataset = dataset.shuffle(shuffle_buffer_size, seed=42)
    
    # Batch the dataset
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# Download and prepare the dataset again
builder = tfds.builder('speech_commands')
builder.download_and_prepare()

# Load training data
train_ds = builder.as_dataset(split='train', as_supervised=True)
num_examples = builder.info.splits['train'].num_examples
print(f'Number of training examples: {num_examples}')

# Load test data
test_ds = builder.as_dataset(split='test', as_supervised=True)
num_examples = builder.info.splits['test'].num_examples
print(f'Number of test examples: {num_examples}')

# Load validation data
validation_ds = builder.as_dataset(split='validation', as_supervised=True)
num_examples = builder.info.splits['validation'].num_examples
print(f'Number of validation examples: {num_examples}')

# Prepare datasets
BATCH_SIZE = 256
train_ds = prepare_dataset(train_ds, BATCH_SIZE)
test_ds = prepare_dataset(test_ds, BATCH_SIZE)
validation_ds = prepare_dataset(validation_ds, BATCH_SIZE)

# Create preprocessor instance
audio_preprocessor = AudioPreprocessing()

# List of preprocessing methods to compare
preprocessing_methods = [
    audio_preprocessor.leaf,
    audio_preprocessor.melfbanks,
    audio_preprocessor.tfbanks,
    audio_preprocessor.sincnet,
    audio_preprocessor.sincnet_plus
]

# Run experiments with each method
results = []
for prep_method in preprocessing_methods:
    # Reset random seed before each experiment
    set_random_seed()
    
    # Reset model weights
    model = TransformerModel(
        num_classes=len(WORDS) + 2,
        d_model=256, 
        num_heads=8, 
        num_layers=4,
        dropout_rate=0.1
    )
    
    result = train_model(
        model=model,
        train_dataset=train_ds,
        validation_dataset=validation_ds,
        preprocessing_method=prep_method,
        epochs=20
    )
    results.append(result)

# Compare results
for result in results:
    print(f"\nPreprocessing method: {result['preprocessing_method']}")
    print(f"Best validation accuracy: {result['best_val_accuracy']:.4f}")
