from keras.models import model_from_json
from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

json_file = open("./model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("./20E.h5")

loaded_model.compile(loss='binary_crossentropy',
                     optimizer='rmsprop',
                     metrics=['accuracy'])

img = image.load_img('test/40.jpg', target_size=(150, 150))

x = img_to_array(img)
x = np.expand_dims(x, axis=0)
y_prop = loaded_model.predict(x)

print(y_prop)
