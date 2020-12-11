from .config import *

## SampyoNet Network Model
def SampyoNet():

   ## Input Layer
   input = Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, 3))

   ## Convolutional Layer + MaxPooling
   _ = SeparableConv2D(filters=FILTER_SIZE, kernel_size=3, activation='relu')(input)
   _ = MaxPool2D()(_)

   ## Convolutional Layer + MaxPooling
   _ = SeparableConv2D(filters=FILTER_SIZE, kernel_size=3, activation='relu')(_)
   _ = MaxPool2D()(_)

   ## Convolutional Layer + MaxPooling
   _ = SeparableConv2D(filters=FILTER_SIZE*2, kernel_size=3, activation='relu')(_)
   _ = MaxPool2D()(_)

   ## Convolutional Layer + Normalization + MaxPooling
   _ = SeparableConv2D(filters=FILTER_SIZE*2, kernel_size=3, activation='relu')(_)
   _ = BatchNormalization()(_)
   _ = MaxPool2D()(_)

   ## Convolutional Layer + MaxPooling
   _ = SeparableConv2D(filters=FILTER_SIZE*4, kernel_size=3, activation='relu')(_)
   _ = MaxPool2D()(_)

   ## Convolutional Layer + MaxPooling
   _ = SeparableConv2D(filters=FILTER_SIZE*4, kernel_size=3, activation='relu')(_)
   _ = MaxPool2D()(_)

   ## MaxPooling + Dropout
   _ = GlobalMaxPool2D()(_)
   _ = Dropout(DROPOUT)(_)

   ## 2 Fully Connected Layers
   _ = Dense(units=DENSE_UNITS, activation='relu')(_)
   _ = Dense(units=DENSE_UNITS, activation='relu')(_)

   ## Output Layer
   output = Dense(units=OUTPUT_CLASS, activation='softmax', name='output')(_)

   # Compile and return
   model = Model(inputs=input, outputs=[output])
   return model
