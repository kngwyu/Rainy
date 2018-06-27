def has_members(*args):
    def decorator(Class):
        for mem in args:
            if hasattr(Class, mem):
                continue
            raise AttributeError("The class {} must have {}".format(Class, mem))
        return Class
    return decorator

