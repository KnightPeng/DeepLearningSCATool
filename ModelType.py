from enum import Enum


class ModelType(Enum):
    MLP = "MLP"
    CNN = "CNN"
    CNN_custom = "CNN_custom"
    RNN = "RNN"
    ASCAD2 = "ASCAD2"