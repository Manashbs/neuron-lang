"""Neuron Plot/Visualization Library"""
from neuron.environment import NeuronModule
from neuron.interpreter import BuiltinFunction

def _get_plt():
    import matplotlib.pyplot as plt
    return plt

def create_plot_module():
    bf = BuiltinFunction
    attrs = {
        'plot': bf('plot', lambda a, **kw: (_get_plt().plot(a[0], a[1] if len(a) > 1 else None, **kw), None)[-1]),
        'scatter': bf('scatter', lambda a, **kw: (_get_plt().scatter(a[0], a[1], **kw), None)[-1]),
        'bar': bf('bar', lambda a, **kw: (_get_plt().bar(a[0], a[1], **kw), None)[-1]),
        'hist': bf('hist', lambda a, **kw: (_get_plt().hist(a[0], bins=kw.get('bins', 30), **{k:v for k,v in kw.items() if k!='bins'}), None)[-1]),
        'title': bf('title', lambda a, **kw: (_get_plt().title(a[0]), None)[-1]),
        'xlabel': bf('xlabel', lambda a, **kw: (_get_plt().xlabel(a[0]), None)[-1]),
        'ylabel': bf('ylabel', lambda a, **kw: (_get_plt().ylabel(a[0]), None)[-1]),
        'legend': bf('legend', lambda a, **kw: (_get_plt().legend(), None)[-1]),
        'grid': bf('grid', lambda a, **kw: (_get_plt().grid(True), None)[-1]),
        'display': bf('display', lambda a, **kw: (_get_plt().show(), None)[-1]),
        'save': bf('save', lambda a, **kw: (_get_plt().savefig(a[0], dpi=kw.get('dpi', 150), bbox_inches='tight'), None)[-1]),
        'figure': bf('figure', lambda a, **kw: (_get_plt().figure(figsize=tuple(a[0]) if a else (10,6)), None)[-1]),
        'subplot': bf('subplot', lambda a, **kw: (_get_plt().subplot(a[0], a[1], a[2]), None)[-1]),
        'training_plot': bf('training_plot', _training_plot),
    }
    return NeuronModule("plot", attrs)

def _training_plot(args, **kwargs):
    plt = _get_plt()
    losses = args[0]
    plt.figure(figsize=(10, 5))
    plt.plot(losses, 'b-', linewidth=2)
    plt.title('Training Loss', fontsize=14)
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    if len(args) > 1:
        plt.savefig(args[1], dpi=150, bbox_inches='tight')
    else:
        plt.show()
