from tensorflow.keras import layers
import keras_cv


class DenseLayer(layers.Layer):
    def __init__(self, units, dropout_rate=0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config

    def build(self, input_shape):
        self.Dense = layers.Dense(units=self.units, use_bias=False)
        self.BatchNormalization = layers.BatchNormalization()
        self.Activation = layers.Activation("relu")
        self.Dropout = layers.Dropout(self.dropout_rate)

        super().build(input_shape)

    def call(self, x):
        x = self.Dense(x)
        x = self.BatchNormalization(x)
        x = self.Activation(x)
        x = self.Dropout(x)

        return x

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class PassThroughLayer(layers.Layer):
    """Đơn giản là placeholdout layer, không biến đổi gì cả"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):

        super().build(input_shape)

    def call(self, x):

        return x

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ImageDataAugmentation(layers.Layer):
    def __init__(
        self,
        rotation_factor,
        zoom_factor,
        bright_factor,
        blur_kernel_size,
        blur_factor,
        saturation_factor,
        contrast_factor,
        contrast_value_range,
        hue_factor,
        hue_value_range,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.rotation_factor = rotation_factor
        self.zoom_factor = zoom_factor
        self.bright_factor = bright_factor
        self.blur_kernel_size = blur_kernel_size
        self.blur_factor = blur_factor
        self.saturation_factor = saturation_factor
        self.contrast_factor = contrast_factor
        self.contrast_value_range = contrast_value_range
        self.hue_factor = hue_factor
        self.hue_value_range = hue_value_range

    def get_config(self):
        # Trả về cấu hình của lớp tùy chỉnh
        config = super().get_config()
        config.update(
            {
                "rotation_factor": self.rotation_factor,
                "zoom_factor": self.zoom_factor,
                "bright_factor": self.bright_factor,
                "blur_kernel_size": self.blur_kernel_size,
                "blur_factor": self.blur_factor,
                "saturation_factor": self.saturation_factor,
                "contrast_factor": self.contrast_factor,
                "contrast_value_range": self.contrast_value_range,
                "hue_factor": self.hue_factor,
                "hue_value_range": self.hue_value_range,
            }
        )
        return config

    def build(self, input_shape):
        self.RandomFlip = keras_cv.layers.RandomFlip(mode="horizontal_and_vertical")
        self.RandomRotation = keras_cv.layers.RandomRotation(
            factor=self.rotation_factor
        )
        self.RandomZoom = keras_cv.layers.RandomZoom(height_factor=self.zoom_factor)
        self.RandomBrightness = keras_cv.layers.RandomBrightness(
            factor=self.bright_factor
        )
        self.RandomGaussianBlur = keras_cv.layers.RandomGaussianBlur(
            kernel_size=self.blur_kernel_size, factor=self.blur_factor
        )
        self.RandomSaturation = keras_cv.layers.RandomSaturation(
            factor=self.saturation_factor,
        )
        self.RandomContrast = keras_cv.layers.RandomContrast(
            factor=self.contrast_factor, value_range=self.contrast_value_range
        )
        self.RandomHue = keras_cv.layers.RandomHue(
            factor=self.hue_factor, value_range=self.hue_value_range
        )

        super().build(input_shape)

    def call(self, x):
        # Xử lí x
        x = self.RandomFlip(x)
        x = self.RandomRotation(x)
        x = self.RandomZoom(x)
        x = self.RandomBrightness(x)
        x = self.RandomGaussianBlur(x)
        x = self.RandomSaturation(x)
        x = self.RandomContrast(x)
        x = self.RandomHue(x)

        return x

    @classmethod
    def from_config(cls, config):
        # Giải mã lại lớp từ cấu hình
        return cls(**config)


class ImageDataAugmentation1(layers.Layer):
    def __init__(self, rotation_factor, zoom_factor, **kwargs):
        super().__init__(**kwargs)
        self.rotation_factor = rotation_factor
        self.zoom_factor = zoom_factor

    def get_config(self):
        # Trả về cấu hình của lớp tùy chỉnh
        config = super().get_config()
        config.update(
            {
                "rotation_factor": self.rotation_factor,
                "zoom_factor": self.zoom_factor,
            }
        )
        return config

    def build(self, input_shape):
        self.RandomFlip = layers.RandomFlip(mode="horizontal_and_vertical")
        self.RandomRotation = layers.RandomRotation(factor=self.rotation_factor)
        self.RandomZoom = layers.RandomZoom(height_factor=self.zoom_factor)

        super().build(input_shape)

    def call(self, x):
        x = self.RandomFlip(x)
        x = self.RandomRotation(x)
        x = self.RandomZoom(x)

        return x

    @classmethod
    def from_config(cls, config):
        # Giải mã lại lớp từ cấu hình
        return cls(**config)


class SeparableConv2DBlock(layers.Layer):
    def __init__(self, filters, name=None, **kwargs):
        # super(ConvNetBlock_XceptionVersion, self).__init__()
        super().__init__(name=name, **kwargs)
        self.filters = filters

    def get_config(self):
        # Trả về cấu hình của lớp tùy chỉnh
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
            }
        )
        return config

    def build(self, input_shape):
        self.BatchNormalization1 = layers.BatchNormalization()
        self.Activation1 = layers.Activation("relu")
        self.SeparableConv2D1 = layers.SeparableConv2D(
            filters=self.filters,
            kernel_size=3,
            padding="same",
            use_bias=False,
        )

        self.BatchNormalization2 = layers.BatchNormalization()
        self.Activation2 = layers.Activation("relu")
        self.SeparableConv2D2 = layers.SeparableConv2D(
            filters=self.filters,
            kernel_size=3,
            padding="same",
            use_bias=False,
        )

        self.MaxPooling2D = layers.MaxPooling2D(pool_size=2, strides=2, padding="same")

        self.Conv2DForResidual = layers.Conv2D(
            filters=self.filters,
            kernel_size=1,
            strides=2,
            padding="same",
            use_bias=False,
        )

        super().build(input_shape)

    def call(self, x):
        residual = x

        x = self.BatchNormalization1(x)
        x = self.Activation1(x)
        x = self.SeparableConv2D1(x)
        x = self.BatchNormalization2(x)
        x = self.Activation2(x)
        x = self.SeparableConv2D2(x)
        x = self.MaxPooling2D(x)

        residual = self.Conv2DForResidual(residual)
        x = layers.add([x, residual])

        return x

    @classmethod
    def from_config(cls, config):
        # Giải mã lại lớp từ cấu hình
        return cls(**config)


class Conv2DBlock(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

    def get_config(self):
        # Trả về cấu hình của lớp tùy chỉnh
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
            }
        )
        return config

    def build(self, input_shape):
        self.BatchNormalization1 = layers.BatchNormalization()
        self.Activation1 = layers.Activation("relu")
        self.Conv2D1 = layers.Conv2D(
            filters=self.filters,
            kernel_size=3,
            padding="same",
            use_bias=False,
        )

        self.BatchNormalization2 = layers.BatchNormalization()
        self.Activation2 = layers.Activation("relu")
        self.Conv2D2 = layers.Conv2D(
            filters=self.filters,
            kernel_size=3,
            padding="same",
            use_bias=False,
        )

        self.MaxPooling2D = layers.MaxPooling2D(
            pool_size=2,
            strides=2,
            padding="same",
        )

        self.Conv2DForResidual = layers.Conv2D(
            filters=self.filters,
            kernel_size=1,
            strides=2,
            padding="same",
            use_bias=False,
        )

        super().build(input_shape)

    def call(self, x):
        residual = x

        x = self.BatchNormalization1(x)
        x = self.Activation1(x)
        x = self.Conv2D1(x)
        x = self.BatchNormalization2(x)
        x = self.Activation2(x)
        x = self.Conv2D2(x)
        x = self.MaxPooling2D(x)

        residual = self.Conv2DForResidual(residual)
        x = layers.add([x, residual])

        return x

    @classmethod
    def from_config(cls, config):
        # Giải mã lại lớp từ cấu hình
        return cls(**config)


class Conv2DBlockNoMaxPooling(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

    def get_config(self):
        # Trả về cấu hình của lớp tùy chỉnh
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
            }
        )
        return config

    def build(self, input_shape):
        self.bn = layers.BatchNormalization()
        self.activation = layers.Activation("relu")
        self.conv2d = layers.Conv2D(
            filters=self.filters,
            kernel_size=3,
            padding="same",
            use_bias=False,
        )

        self.Conv2DForResidual = layers.Conv2D(filters=self.filters, kernel_size=1)

        super().build(input_shape)

    def call(self, x):
        residual = x

        x = self.bn(x)
        x = self.activation(x)
        x = self.conv2d(x)

        residual = self.Conv2DForResidual(residual)
        x = layers.add([x, residual])

        return x

    @classmethod
    def from_config(cls, config):
        # Giải mã lại lớp từ cấu hình
        return cls(**config)


class DenseLayerList(layers.Layer):
    def __init__(self, dropout_rate, list_units, do_have_last_layer, **kwargs):
        super().__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.list_units = list_units
        self.do_have_last_layer = do_have_last_layer

    def get_config(self):
        # Trả về cấu hình của lớp tùy chỉnh
        config = super().get_config()
        config.update(
            {
                "dropout_rate": self.dropout_rate,
                "list_units": self.list_units,
                "do_have_last_layer": self.do_have_last_layer,
            }
        )
        return config

    def build(self, input_shape):
        self.DenseLayers = (
            [
                DenseLayer(units=units, dropout_rate=self.dropout_rate)
                for units in self.list_units
            ]
            if self.list_units is not None
            else PassThroughLayer()
        )
        self.lastDenseLayer = (
            DenseLayer(units=self.list_units[-1], dropout_rate=0)
            if self.do_have_last_layer
            else PassThroughLayer()
        )

        super().build(input_shape)

    def call(self, x):
        # Xử lí x
        for layer in self.DenseLayers:
            x = layer(x)

        x = self.lastDenseLayer(x)

        return x

    @classmethod
    def from_config(cls, config):
        # Giải mã lại lớp từ cấu hình
        return cls(**config)


class Conv2DBlockList(layers.Layer):
    def __init__(self, list_filters, layer_name, do_have_last_layer, **kwargs):
        super().__init__(**kwargs)
        self.list_filters = list_filters
        self.layer_name = layer_name
        self.do_have_last_layer = do_have_last_layer

    def get_config(self):
        # Trả về cấu hình của lớp tùy chỉnh
        config = super().get_config()
        config.update(
            {
                "list_filters": self.list_filters,
                "layer_name": self.layer_name,
                "do_have_last_layer": self.do_have_last_layer,
            }
        )
        return config

    def build(self, input_shape):
        ConvName = globals()[self.layer_name]

        self.Conv2DBlocks = [
            ConvName(
                filters=filters,
            )
            for filters in self.list_filters
        ]

        self.lastConv2DBlock = (
            Conv2DBlockNoMaxPooling(filters=self.list_filters[-1])
            if self.do_have_last_layer
            else PassThroughLayer()
        )

        super().build(input_shape)

    def call(self, x):
        for layer in self.Conv2DBlocks:
            x = layer(x)

        x = self.lastConv2DBlock(x)

        return x

    @classmethod
    def from_config(cls, config):
        # Giải mã lại lớp từ cấu hình
        return cls(**config)
