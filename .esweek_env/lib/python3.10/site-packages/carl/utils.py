'''Utilities for use alongside `carl`.'''
from wrapt import decorator


@decorator
def empty_args(f, _, args, kwargs):
    '''Make a decorator that takes parameters be callable with none.'''
    if len(args) == 1 and not kwargs and callable(args[0]):
        return f()(args[0])
    return f(*args, **kwargs)


@decorator
def printer(f, _, args, kwargs):
    '''Print the result of a function. For use as a wrapper.'''
    print(f(*args, **kwargs))
