import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
from redes_neuronales_v11 import *

# Optimizadores y Schedulers
import torch.optim as optim

class EfficientNetB3():
    def __init__(self, out_features=20):
        if out_features==20:
            self.net = EfficientNetB3_20_classes()
        else: 
            self.net = models.efficientnet_b3()
        self.letter_variant = 'A'
        self.lr = 1e-4
        self.optimizer = optim.RMSprop(self.net.parameters(), lr=self.lr, momentum=0.9, alpha=0.9, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.97)

    def get_net(self):
        return self.net

    def get_scheduler(self):
        return self.scheduler

    def get_optimizer(self):
        return self.optimizer

    def get_lr(self):
        return self.lr

    def get_checkpoint(self):
        if self.letter_variant == 'B':
            return f'New_Checkpoints/B/effnetb3/B_effnetb3_checkpoint'
        if self.letter_variant == 'C':
            return f'New_Checkpoints/C/effnetb3/C_effnetb3_checkpoint'
        if self.letter_variant == 'D':
            return f'New_Checkpoints/D/effnetb3/D_effnetb3_checkpoint'
        if self.letter_variant == 'E':
            return f'New_Checkpoints/E/effnetb3/E_effnetb3_checkpoint'
        return f'F_Checkpoints/effnetb3/effnetb3_checkpoint'

    def get_checkpoint_acc(self):
        if self.letter_variant == 'B':
            return 66.0
        if self.letter_variant == 'C':
            return 68.0
        if self.letter_variant == 'D':
            return 66.0
        if self.letter_variant == 'E':
            return 70.8
        return 68.0

    def get_checkpoint_lr(self):
        if self.letter_variant == 'B':
            return 8.079828447811299e-05
        if self.letter_variant == 'C':
            return 8.587340257e-05
        if self.letter_variant == 'D':
            return 8.587340257e-05
        if self.letter_variant == 'E':
            return 8.587340257e-05
        return 8.329720049289999e-05

    def update(self, letter):
        self.letter_variant = letter

class Resnet152():
    def __init__(self, out_features=20):
        if out_features==20:
            self.net = Resnet152_20_classes()
        else: 
            self.net = models.resnet152()
        self.letter_variant = 'A'
        self.lr = 0.01
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    def get_net(self):
        return self.net

    def get_scheduler(self):
        return self.scheduler

    def get_optimizer(self):
        return self.optimizer

    def get_lr(self):
        return self.lr

    def get_checkpoint(self):
        if self.letter_variant == 'B':
            return f'New_Checkpoints/B/resnet152/B_resnet152_checkpoint'
        if self.letter_variant == 'C':
            return f'New_Checkpoints/C/resnet152/C_resnet152_checkpoint'
        if self.letter_variant == 'D':
            return f'New_Checkpoints/D/resnet152/D_resnet152_checkpoint'
        if self.letter_variant == 'E':
            return f'New_Checkpoints/E/resnet152/E_resnet152_checkpoint'
        return f'F_Checkpoints/resnet152/resnet152_checkpoint'

    def get_checkpoint_acc(self):
        if self.letter_variant == 'B':
            return 59.0
        if self.letter_variant == 'C':
            return 58.199999999999996
        if self.letter_variant == 'D':
            return 59.199999999999996
        if self.letter_variant == 'E':
            return 61.4
        return 61.6

    def get_checkpoint_lr(self):
        if self.letter_variant == 'B':
            return 0.001
        if self.letter_variant == 'C':
            return 0.001
        if self.letter_variant == 'D':
            return 0.001
        if self.letter_variant == 'E':
            return 0.001
        return 0.001

    def update(self, letter):
        self.letter_variant = letter

class Squeezenet():
    def __init__(self, out_features=20):
        if out_features==20:
            self.net = SqueezeNet_20_classes()
        else: 
            self.net = models.squeezenet1_0()
        self.letter_variant = 'A'
        self.lr = 0.001
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0002) 
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    def get_net(self):
        return self.net

    def get_scheduler(self):
        return self.scheduler

    def get_optimizer(self):
        return self.optimizer

    def get_lr(self):
        return self.lr

    def get_checkpoint(self):
        if self.letter_variant == 'B':
            return f'New_Checkpoints/B/squeezenet/B_squeezenet_checkpoint'
        if self.letter_variant == 'C':
            return f'New_Checkpoints/C/squeezenet/C_squeezenet_checkpoint'
        if self.letter_variant == 'D':
            return f'New_Checkpoints/D/squeezenet/D_squeezenet_checkpoint'
        if self.letter_variant == 'E':
            return f'New_Checkpoints/E/squeezenet/E_squeezenet_checkpoint'
        return f'F_Checkpoints/squeezenet/squeezenet_checkpoint'

    def get_checkpoint_acc(self):
        if self.letter_variant == 'B':
            return 39.2
        if self.letter_variant == 'C':
            return 39.4
        if self.letter_variant == 'D':
            return 39.2
        if self.letter_variant == 'E':
            return 39.2
        return 39.800000000000004

    def get_checkpoint_lr(self):
        if self.letter_variant == 'B':
            return 0.0001
        if self.letter_variant == 'C':
            return 0.0001
        if self.letter_variant == 'D':
            return 0.0001
        if self.letter_variant == 'E':
            return 0.0001
        return 1e-05

    def update(self, letter):
        self.letter_variant = letter

def instantiate_network(selection):
    if selection=="effnetb3":
        return EfficientNetB3()
    if selection=="squeezenet":
        return Squeezenet()
    if selection=="resnet152":
        return Resnet152()