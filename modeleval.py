# Load the best model
model = tf.keras.models.load_model('sign_language_model.h5')

# Evaluate the model
scores = model.evaluate(validation_generator)
print(f"Validation Accuracy: {scores[1]*100:.2f}%")
