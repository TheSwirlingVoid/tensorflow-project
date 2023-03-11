from tensorflow import keras, zeros, data
from keras import layers, utils, Sequential, losses, optimizers, callbacks
import pickle

class Net():

    def __init__(self, input_shape) -> None:
        self.BATCH_SIZE = 32
        self.INPUT_SHAPE = input_shape
        self.LEARNING_RATE = 0.001

        self.__setup_model()

    def get_batch_size(self) -> str:
        return self.BATCH_SIZE

    def get_learning_rate(self) -> float:
        return self.LEARNING_RATE
        
    def __setup_model(self) -> None:
        self.model = Sequential()

        self.model.add(layers.ZeroPadding2D(
                padding=((1,1), (1,1)),
                input_shape=self.INPUT_SHAPE
            )) # output 225 x 225 x 3

        self.model.add(layers.Conv2D(
                32,
                11,
                strides=5,
                activation="relu",
            )) # output 44 x 44 x 32

        self.model.add(layers.MaxPool2D(
                pool_size=2
            )) # output 22 x 22 x 32

        self.model.add(layers.Conv2D(
            32,
            2,
            strides=1,
            activation="relu"
        )) # output 21 x 21 x 32

        self.model.add(layers.ZeroPadding2D(
            padding=((1,0), (1,0)),
            input_shape=self.INPUT_SHAPE
        )) # output 22 x 22 x 32

        self.model.add(layers.MaxPool2D(
            pool_size=2
        )) # 11 x 11 x 32

        self.model.add(layers.Flatten())

        self.model.add(layers.Dense(
                4096,
                activation="relu"
            ))

        self.model.add(layers.Dense(
                1024,
                activation="relu"
            ))

        self.model.add(layers.Dense(
                450,
                activation="relu"
            ))
        # self.model.add(layers.Dense(
        #         20,
        #         activation="softmax"
        #     ))

        self.loss = losses.CategoricalCrossentropy()
        self.optimizer = optimizers.SGD(learning_rate=self.LEARNING_RATE)

        self.model.compile(
                loss=self.loss,
                optimizer=self.optimizer,
                metrics=["accuracy"],
                )

        def __str__(self) -> str:
            self.model.summary()
            return ""

def main() -> None:
    input_shape = (224, 224, 3)
    net = Net(input_shape)
    print(net)

    train = utils.image_dataset_from_directory(
                "archive/train",
                label_mode="categorical",
                batch_size=net.get_batch_size(),
                image_size=input_shape[:2], 
            ) 

    test = utils.image_dataset_from_directory(
                "archive/valid",
                label_mode="categorical",
                batch_size=net.get_batch_size(),
                image_size=input_shape[:2], 
            )
    
    # with open("checkpoints/450classes/class_names.data", "wb") as f:
    #     pickle.dump(train.class_names, f)

    rotation = Sequential([
        layers.RandomRotation(0.125, fill_mode="constant", input_shape=input_shape)
    ])

    rotated = train.map(lambda train, sol: (rotation(train), sol))

    train.concatenate(rotated)

    train = train.cache().prefetch(buffer_size=data.AUTOTUNE)
    test = test.cache().prefetch(buffer_size=data.AUTOTUNE)

    # net.model.save("temp/model")
    net.model.fit(
            train,
            batch_size=net.get_batch_size(),
            epochs=300,
            verbose=1,
            validation_data=test,
            validation_batch_size=net.get_batch_size(),
            callbacks=[
                    callbacks.ModelCheckpoint(
                        "checkpoints/higherdenselayers/checkpoints_{epoch:02f}",
                        verbose=2,
                        save_freq="epoch"
                    )
            ]
            )

main()
