from model import *
from parameters import *
tem = Model(parameters())
import torch.utils.bottleneck as bottleneck
bottleneck.run(tem)
