import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import warnings
import numpy as np
warnings.filterwarnings('ignore')

BATCH_SIZE = 32
IMAGE_SIZE = 256

class_names = ['Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']


# Tải mô hình
loaded_model = tf.keras.models.load_model('/content/gdrive/MyDrive/Tomato/my_model.h5')

def predict_image(image_path, model):
	# Đọc hình ảnh từ file và chuyển đổi thành tensor
	img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
	img_array = tf.keras.preprocessing.image.img_to_array(img)
	img_tensor = tf.keras.preprocessing.image.smart_resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))
	img_tensor = tf.expand_dims(img_tensor, axis=0)
	# Dự đoán nhãn của hình ảnh
	prediction = model.predict(img_tensor)
	predicted_class = np.argmax(prediction[0])
	
	return class_names[predicted_class]

image_path = "/content/gdrive/MyDrive/Tomato/data/val/Tomato___Target_Spot/Tomato___Target_Spot_original_0e763e60-8821-4484-9793-f8ee4f515127___Com.G_TgS_FL 8135.JPG_d1afcf01-94b9-4536-a562-5ce58c5b5920.JPG"


print(predict_image(image_path, loaded_model))