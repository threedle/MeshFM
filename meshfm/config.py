"""Model config loading."""

import yaml


class DotDict(dict):
    """dict with attribute access; missing keys read back as ``None``."""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def _to_dot_dict(obj):
    if isinstance(obj, dict):
        dd = DotDict()
        for k, v in obj.items():
            dd[k] = _to_dot_dict(v)
        return dd
    return obj


def load_config(cfg_path):
    """Load a model YAML for attribute access."""
    with open(cfg_path, 'r') as f:
        data = yaml.safe_load(f)
    return _to_dot_dict(data)
