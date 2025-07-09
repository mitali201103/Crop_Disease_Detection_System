from src.data_preprocess import load_data
from src.model import build_model

train, val = load_data('data/', img_size=(128,128), batch_size=32)

model = build_model(input_shape=(128,128,3), num_classes=len(train.class_indices))

model.fit(train, validation_data=val, epochs=10)

model.save("plant_disease_model.h5")