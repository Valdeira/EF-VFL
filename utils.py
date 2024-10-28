from importlib import import_module


def load_module(module_path, module_name):
    spec = import_module(module_path)
    return getattr(spec, module_name)
