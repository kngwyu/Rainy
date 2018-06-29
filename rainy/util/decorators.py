

def has_members(*args):
    """Returns a class decorator to check if it has all given members or not,
       when __init__ is caled.
    """
    def decorator(class_):
        def init(self, *init_args):
            self.__init__(init_args)
            for arg in args:
                if hasattr(self, arg):
                    continue
                raise ValueError("This class must have {}".format(arg))
        setattr(class_, "__init__", init)
        return class_
    return decorator
