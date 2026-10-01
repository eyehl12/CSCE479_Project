import tensorflow as tf


#define model class
class DenseModel(tf.keras.Model):
    def __init__(self, hidden_layers=(128, 128, 64, 32), activation=tf.nn.relu, output_size=10, dropout_rate=0.0, learning_rate=0.001, l2_rate=0.0):
        #initialize model settings
        super(DenseModel, self).__init__()
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.l2_rate = l2_rate

    def build(self, input_shape):
        #avoid rebuilding an existing model and resetting its weights
        if self.built:
            return

        #define input preprocessing inside the model
        self.network = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1.0 / 255),
            tf.keras.layers.Flatten(),
        ])

        #use L2 on hidden-layer weights when value is positive
        if self.l2_rate > 0:
            regularizer = tf.keras.regularizers.L2(self.l2_rate) 
        else:
            regularizer = None

        #define the hidden layers and optional dropout
        for units in self.hidden_layers:
            self.network.add(tf.keras.layers.Dense(
                units,
                activation=self.activation,
                kernel_regularizer=regularizer,
            ))
            if self.dropout_rate > 0:
                self.network.add(tf.keras.layers.Dropout(self.dropout_rate))

        #define the output layer and create the weights
        self.network.add(tf.keras.layers.Dense(self.output_size, activation='softmax'))
        self.network.build(input_shape)
        super(DenseModel, self).build(input_shape)

        #configure the optimizer, loss, and metrics for training
        self.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )

    def call(self, x, training=False):
        #forward pass; dropout is active only during training
        return self.network(x, training=training)
