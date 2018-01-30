# The path where the original images together with their xml are stored
IMAGES_XML_PATH = "images/"

# The paths where the images, labels and npy files will be stored
IMAGES_PATH = "dataset/images"
LABELS_PATH = "dataset/labels"
NPY_PATH = "dataset/tiramisu"

# The size and channels of the images
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
CHANNELS = 3

# The size of the patches that are extracted from an image
PATCHSIZE = (128,128)
# The size of the steps for the patches (the smaller the steps the bigger the number of generated patches)
STEP = (128,128)

# Model to train and use. Currently the only two available models are Tiramisu and Unet
MODEL = "Tiramisu"

# Path where the weights of the model are saved
OUTPUT = "output/"

# Number of epochs to train the model
EPOCHS = 30000
# Number of epochs to save the weights
EPOCHS_SAVE = 5000

# Mean and std (this must be changed)
MU = 0.6019167833665388
STD = 0.2263702138716035
