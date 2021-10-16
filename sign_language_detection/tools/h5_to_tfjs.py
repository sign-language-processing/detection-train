import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow.keras.models import load_model



model = load_model('../model.h5')
model.build(input_shape=(1, 1, 75))

model.layers[1].stateful = True

model.predict(np.random.randn(1, 1, 75))  # Set input shapes

model.save("stateful_model.h5")
