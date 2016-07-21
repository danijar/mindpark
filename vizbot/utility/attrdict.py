class AttrDict(dict):

    def __getattr__(self, key):
        if key not in self:
            raise AttributeError
        return self[key]

    def __setattr__(self, key, value):
        if key not in self:
            raise AttributeError
        self[key] = value


def use_attrdicts(obj):
    if isinstance(obj, dict):
        obj = {k: use_attrdicts(v) for k, v in obj.items()}
        return AttrDict(obj)
    elif isinstance(obj, list):
        return [use_attrdicts(x) for x in obj]
    return obj
