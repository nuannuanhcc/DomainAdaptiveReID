from __future__ import absolute_import
import warnings

from .cuhk01 import CUHK01
from .cuhk03 import CUHK03
from .dukemtmc import DukeMTMC
from .market1501 import Market1501
from .viper import VIPeR
from .veri776 import Veri776
from .vehicleid import VehicleID


__factory = {
    'market1501': Market1501,
    'cuhk03': CUHK03,
    'dukemtmc': DukeMTMC,
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
