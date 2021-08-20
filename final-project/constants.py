"""
    This file contains all the constants
"""

IMG_SIZE = 256
BATCH_SIZE = 45

NUM_OF_IMAGES_TRAIN = 1800
NUM_OF_IMAGES_VAL = 200
NUM_OF_IMAGES_TEST = 200

# We calculate the steps per epoch so that we use all the images
# we have in the folders
STEPS_PER_EPOCH = NUM_OF_IMAGES_TRAIN // BATCH_SIZE
VAL_STEPS = NUM_OF_IMAGES_VAL // BATCH_SIZE
