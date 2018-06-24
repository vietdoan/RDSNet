import json

from .camvid_loader import CamvidLoader
from .cityscapes_loader import CityscapesLoader
from .sunrgb_loader import SunRGBLoader

def get_loader(name):
    """get_loader
    :param name:
    """
    return {
        'camvid': CamvidLoader,
        'cityscapes': CityscapesLoader,
        'sunrgb': SunRGBLoader
    }[name]