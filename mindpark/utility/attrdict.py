class AttrDict(dict):

    def __getattr__(self, key):
        if key not in self:
            raise AttributeError("unknown key '{}'".format(key))
        return self[key]

    def __setattr__(self, key, value):
        if key not in self:
            raise AttributeError("unknown key '{}'".format(key))
        self[key] = value


def use_attrdicts(obj):
    if isinstance(obj, dict):
        return AttrDict({k: use_attrdicts(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [use_attrdicts(x) for x in obj]
    return obj
