from keras import applications
from keras.layers import Input
from keras.models import Model
from tensorflow.keras import layers, models, regularizers


xcep_base = applications.Xception(weights = 'imagenet', include_top = False, input_shape = (150, 150, 3))
xcep_base.trainable = False

inputs = Input(shape=(150, 150, 3))

x = xcep_base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(1024, activation = 'relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(2, activation = 'sigmoid')(x)
xcep_model = Model(inputs, outputs)


from tensorflow import keras
import matplotlib.pyplot as plt

# Compile the model
xcep_model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.CategoricalAccuracy()],
)




# Train the model and store the history
epochs = 10
history = xcep_model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

# Extract accuracy and validation accuracy from the history
acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']

# Plot the training and validation accuracy curves
plt.figure(figsize=(10, 5))
plt.plot(range(epochs), acc, label='Training Accuracy', color='blue')
plt.plot(range(epochs), val_acc, label='Validation Accuracy', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('training_validation_accuracy.png')
plt.show()

# Saving the model weights
xcep_model.save_weights('xcep_model_weights.h5')


