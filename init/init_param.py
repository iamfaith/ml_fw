

@staticmethod
def wandb(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        print('call %s():' % func)
        return func(*args, **kw)
    return wrapper



