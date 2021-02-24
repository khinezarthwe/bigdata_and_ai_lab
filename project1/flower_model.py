import tensorflow as tf

reload_model = tf.keras.models.load_model('project1/saved_flowers_model')
IMAGE_SIZE = (224, 224)
classes = {
    'roses': 2,
    'daisy': 0,
    'tulips': 4,
    'dandelion': 1,
    'sunflowers': 3}


def get_class_string_from_index(index):
   for class_string, class_index in classes.items():
      if class_index == index:
         return class_string
