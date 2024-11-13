import tensorflow as tf
from tensorflow.keras import layers

# Feature extraction, downsampling
class _EncodeBlock(tf.keras.Model):
    """
    Parameters:
    - filters: List of 2D convolution filters
    - stage: Integer, current stage label used for generating layer names
    - data_format: Specify the channel location in the input image ('channels_first' or 'channels_last')
    """
    def __init__(self, filters, stage, data_format):
        super(_EncodeBlock, self).__init__(name='encode_block_{}'.format(stage))
        
        filters1, filters2 = filters
        conv_name_base = 'encode' + str(stage) + '_conv'
        bn_name_base = 'encode' + str(stage) + '_bn'
        pool_name = 'encode' + str(stage) + '_pool'
        bn_axis = 1 if data_format == 'channels_first' else 3

        self.conv2a = layers.Conv2D(
            filters1, (3, 3),
            padding='same',
            data_format=data_format,
            name=conv_name_base + '2a'
        )
        self.bn2a = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2a'
        )

        self.conv2b = layers.Conv2D(
            filters2, (3, 3),
            padding='same',
            data_format=data_format,
            name=conv_name_base + '2b'
        )
        self.bn2b = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2b'
        )

        self.pool = layers.MaxPooling2D(data_format=data_format, name=pool_name)

    def call(self, input_tensor, training=True):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = layers.Activation('relu')(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = layers.Activation('relu')(x)

        poolx = self.pool(x)
        return [x, poolx]

# Upsampling
class _DecodeBlock(tf.keras.Model):
    """
    Args:
    - filters: List of 2D convolution filters
    - stage: Integer, current stage label used for generating layer names
    - data_format: Specify the channel location in the input image ('channels_first' or 'channels_last')
    - transpose_conv: Bool, whether to use transposed convolution for upsampling
    """
    def __init__(self, filters, stage, data_format, transpose_conv=False):
        super(_DecodeBlock, self).__init__(name='decode_block_{}'.format(stage))
        
        filters1, filters2, filters3 = filters
        self.transpose_conv = transpose_conv

        conv_name_base = 'decode' + str(stage) + '_conv'
        bn_name_base = 'decode' + str(stage) + '_bn'
        up_name_base = 'decode' + str(stage) + '_'
        bn_axis = 1 if data_format == 'channels_first' else 3

        self.conva = layers.Conv2D(
            filters1, (3, 3),
            padding='same',
            data_format=data_format,
            name=conv_name_base + '2a'
        )
        self.bn2a = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2a'
        )

        self.conv2b = layers.Conv2D(
            filters2, (3, 3),
            padding='same',
            data_format=data_format,
            name=conv_name_base + '2b'
        )
        self.bn2b = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2b'
        )

        if self.transpose_conv:
            self.conv2c_transpose = layers.Conv2DTranspose(
                filters3, (3, 3),
                strides=(2, 2),
                padding='same',
                data_format=data_format,
                name=up_name_base + '2c_transpose'
            )
        else:
            self.upsample = layers.UpSampling2D(
                size=(2, 2),
                data_format=data_format,
                name=up_name_base + 'upsample'
            )
            self.conv2c = layers.Conv2D(
                filters3, (1, 1), data_format=data_format, name=conv_name_base + '2c'
            )
        self.bn2c = layers.BatchNormalization(
            axis=bn_axis, name=bn_name_base + '2b'
        )

    def call(self, input_tensor, training=True):
        x = self.conva(input_tensor)
        x = self.bn2a(x, training=training)
        x = layers.Activation('relu')(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = layers.Activation('relu')(x)

        if self.transpose_conv:
            x = self.conv2c_transpose(x)
        else:
            x = self.upsample(x)
            x = self.conv2c(x)
        x = self.bn2c(x, training=training)
        return layers.Activation('relu')(x)

# Construct U-Net Model
class Unet(tf.keras.Model):
    """
    Construct U-Net model
    
    Parameters:
    - data_format: Specify 'channels_first' or 'channels_last'
    - classes: Int, pixel categories
    - transpose_conv: Bool, specify whether to use transposed convolution for upsampling
    - name: Prefix for variable names created in the model
    """
    def __init__(self, data_format, classes, transpose_conv=False, name=''):
        super(Unet, self).__init__(name=name)
        valid_channel_values = ('channels_first', 'channels_last')
        if data_format not in valid_channel_values:
            raise ValueError('Unknown data_format: %s. Valid values: %s' % (data_format, valid_channel_values))

        self.concat_axis = 3 if data_format == 'channels_last' else 1

        def encode_block(filters, stage):
            return _EncodeBlock(filters, stage=stage, data_format=data_format)

        def decode_block(filters, stage):
            return _DecodeBlock(
                filters,
                stage=stage,
                data_format=data_format,
                transpose_conv=transpose_conv
            )

        self.e1 = encode_block([32, 32], stage=1)
        self.e2 = encode_block([64, 64], stage=2)
        self.e3 = encode_block([128, 128], stage=3)
        self.e4 = encode_block([256, 256], stage=4)

        self.d4 = decode_block([512, 512, 256], stage=4)
        self.d3 = decode_block([256, 256, 128], stage=3)
        self.d2 = decode_block([128, 128, 64], stage=2)
        self.d1 = decode_block([64, 64, 32], stage=1)

        self.output_block = encode_block([64, 64], stage=5)
        self.conv_output = layers.Conv2D(
            classes, (1, 1), data_format=data_format, name='conv_output'
        )

    def build_call(self, inputs, training=True):
        """
        Construct the model
        Args:
        - inputs: Images to be segmented
        - training: Whether it is in training mode

        Returns:
        - Tensor: shape=[n, h, w, classes]
        """
        e1x, x = self.e1(inputs, training=training)
        e2x, x = self.e2(x, training=training)
        e3x, x = self.e3(x, training=training)
        e4x, x = self.e4(x, training=training)
        x = self.d4(x, training=training)

        x = layers.concatenate([x, e4x], axis=self.concat_axis)
        x = self.d3(x, training=training)

        x = layers.concatenate([x, e3x], axis=self.concat_axis)
        x = self.d2(x, training=training)

        x = layers.concatenate([x, e2x], axis=self.concat_axis)
        x = self.d1(x, training=training)

        x = layers.concatenate([x, e1x], axis=self.concat_axis)
        x, _ = self.output_block(x, training=training)
        x = self.conv_output(x)
        return x

    def call(self, inputs, training=True):
        """
        Model unit
        Args:
        - inputs: Image feature vectors
        - training: Whether it is in training mode
        
        Returns:
        - Tensor: shape=[n, h, w, classes]
        """
        return self.build_call(inputs, training)
