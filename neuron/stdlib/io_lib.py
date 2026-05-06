"""Neuron I/O Library"""
import json, csv, os
from neuron.environment import NeuronModule
from neuron.interpreter import BuiltinFunction

def create_io_module():
    bf = BuiltinFunction
    attrs = {
        'read': bf('read', lambda a, **kw: open(a[0], 'r', encoding='utf-8').read()),
        'write': bf('write', lambda a, **kw: _write_file(a[0], a[1])),
        'append': bf('append', lambda a, **kw: _append_file(a[0], a[1])),
        'read_json': bf('read_json', lambda a, **kw: json.loads(open(a[0], 'r').read())),
        'write_json': bf('write_json', lambda a, **kw: _write_file(a[0], json.dumps(a[1], indent=2))),
        'read_csv': bf('read_csv', lambda a, **kw: list(csv.DictReader(open(a[0], 'r')))),
        'exists': bf('exists', lambda a, **kw: os.path.exists(a[0])),
        'listdir': bf('listdir', lambda a, **kw: os.listdir(a[0] if a else '.')),
        'mkdir': bf('mkdir', lambda a, **kw: os.makedirs(a[0], exist_ok=True)),
        'remove': bf('remove', lambda a, **kw: os.remove(a[0])),
        'join_path': bf('join_path', lambda a, **kw: os.path.join(*a)),
        'basename': bf('basename', lambda a, **kw: os.path.basename(a[0])),
        'dirname': bf('dirname', lambda a, **kw: os.path.dirname(a[0])),
        'abspath': bf('abspath', lambda a, **kw: os.path.abspath(a[0])),
        'read_lines': bf('read_lines', lambda a, **kw: open(a[0], 'r').read().splitlines()),
    }
    return NeuronModule("io", attrs)

def _write_file(path, content):
    with open(path, 'w', encoding='utf-8') as f: f.write(str(content))
def _append_file(path, content):
    with open(path, 'a', encoding='utf-8') as f: f.write(str(content))
