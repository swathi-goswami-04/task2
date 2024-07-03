import numpy as np
from tensorflow.keras.preprocessing import image

def predict_sign(image_path, model):
    img = image.load_img(image_path, target_size=(img_width, img_height))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0
    prediction = model.predict(img)
    class_idx = np.argmax(prediction, axis=1)[0]
    class_label = train_generator.class_indices
    class_label = {v: k for k, v in class_label.items()}
    return class_label[class_idx]

# Example Usage
result = predict_sign('path/to/sign_language_image.jpg', model)
print(f"Predicted Sign: {result}")

