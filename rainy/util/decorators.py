
def dict_init(*args):
    def decorator(class_):
        def dict_init_(self, d: dict):
            for member in args:
                if member in d:
                    setattr(self, member, d[member])
            if hasattr(self, "lazy_init"):
                self.lazy_init()
        setattr(class_, "dict_init", dict_init_)
        return class_
    return decorator

