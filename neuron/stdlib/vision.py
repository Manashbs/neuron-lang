"""Neuron Vision Library — Computer Vision utilities"""
import numpy as np
from neuron.environment import NeuronModule
from neuron.interpreter import BuiltinFunction
from neuron.stdlib.tensor import NeuronTensor

def create_vision_module():
    bf = BuiltinFunction
    attrs = {
        'load_image': bf('load_image', _load_image),
        'save_image': bf('save_image', _save_image),
        'resize': bf('resize', _resize),
        'to_grayscale': bf('to_grayscale', _to_grayscale),
        'normalize': bf('normalize', lambda a, **kw: NeuronTensor(a[0].data / 255.0 if isinstance(a[0], NeuronTensor) else np.array(a[0]) / 255.0)),
        'flip_horizontal': bf('flip_horizontal', lambda a, **kw: NeuronTensor(np.flip(a[0].data if isinstance(a[0], NeuronTensor) else a[0], axis=1))),
        'flip_vertical': bf('flip_vertical', lambda a, **kw: NeuronTensor(np.flip(a[0].data if isinstance(a[0], NeuronTensor) else a[0], axis=0))),
        'rotate90': bf('rotate90', lambda a, **kw: NeuronTensor(np.rot90(a[0].data if isinstance(a[0], NeuronTensor) else a[0]))),
        'flatten': bf('flatten', lambda a, **kw: NeuronTensor((a[0].data if isinstance(a[0], NeuronTensor) else np.array(a[0])).flatten())),
        'random_crop': bf('random_crop', _random_crop),
    }
    return NeuronModule("vision", attrs)

def _load_image(args, **kwargs):
    try:
        from PIL import Image
        img = Image.open(args[0])
        return NeuronTensor(np.array(img, dtype=np.float64))
    except ImportError:
        import struct, zlib
        return NeuronTensor(np.random.rand(28, 28))

def _save_image(args, **kwargs):
    try:
        from PIL import Image
        data = args[0].data if isinstance(args[0], NeuronTensor) else np.array(args[0])
        if data.max() <= 1.0: data = (data * 255).astype(np.uint8)
        else: data = data.astype(np.uint8)
        Image.fromarray(data).save(args[1])
    except ImportError:
        pass

def _resize(args, **kwargs):
    try:
        from PIL import Image
        data = args[0].data if isinstance(args[0], NeuronTensor) else np.array(args[0])
        if data.max() <= 1.0: data = (data * 255).astype(np.uint8)
        else: data = data.astype(np.uint8)
        img = Image.fromarray(data).resize((int(args[1]), int(args[2])))
        return NeuronTensor(np.array(img, dtype=np.float64))
    except ImportError:
        return args[0]

def _to_grayscale(args, **kwargs):
    data = args[0].data if isinstance(args[0], NeuronTensor) else np.array(args[0])
    if data.ndim == 3 and data.shape[2] == 3:
        gray = 0.299 * data[:,:,0] + 0.587 * data[:,:,1] + 0.114 * data[:,:,2]
        return NeuronTensor(gray)
    return NeuronTensor(data)

def _random_crop(args, **kwargs):
    data = args[0].data if isinstance(args[0], NeuronTensor) else np.array(args[0])
    h, w = int(args[1]), int(args[2])
    max_y = data.shape[0] - h
    max_x = data.shape[1] - w
    y = np.random.randint(0, max(1, max_y))
    x = np.random.randint(0, max(1, max_x))
    return NeuronTensor(data[y:y+h, x:x+w])
