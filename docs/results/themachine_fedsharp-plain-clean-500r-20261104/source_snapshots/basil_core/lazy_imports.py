"""Small dependency loaders: no numerical libraries are imported here.

Python caches normal imports. Function-local imports are preferred; a module
proxy is only needed for runtime annotations or definition-time dependencies.
Import errors are propagated unchanged on first use, never swallowed.
"""
from importlib import import_module
from types import ModuleType
from threading import RLock


class LazyModule(ModuleType):
    """Resolve a real module on first attribute access, with import-lock safety."""

    def __init__(self, name, *, backend=None, style=None):
        super().__init__(name)
        self.__dict__['_target_name'] = name
        self.__dict__['_target'] = None
        self.__dict__['_lock'] = RLock()
        self.__dict__['_backend'] = backend
        self.__dict__['_style'] = style

    def _resolve(self):
        target = self.__dict__['_target']
        if target is not None:
            return target
        with self.__dict__['_lock']:
            if self.__dict__['_target'] is None:
                if self.__dict__['_backend'] is not None:
                    import_module('matplotlib').use(self.__dict__['_backend'])
                target = import_module(self.__dict__['_target_name'])
                if self.__dict__['_style'] is not None:
                    target.style.use(self.__dict__['_style'])
                self.__dict__['_target'] = target
            return self.__dict__['_target']

    def __getattr__(self, name):
        return getattr(self._resolve(), name)

    def __dir__(self):
        return dir(self._resolve())


def lazy_module(name):
    return LazyModule(name)


def lazy_pyplot(*, backend=None, style=None):
    return LazyModule('matplotlib.pyplot', backend=backend, style=style)


def pyplot(backend=None):
    """Configure the requested backend before loading pyplot, not at startup."""
    matplotlib = import_module('matplotlib')
    if backend is not None:
        matplotlib.use(backend)
    return import_module('matplotlib.pyplot')


def lazy_tf_function(**options):
    """Delay tf.function construction; cache one compiled function per wrapper.

    The same TensorFlow tracing options and original Python function are used.
    Attribute access (e.g. get_concrete_function) resolves the real function.
    """
    from functools import update_wrapper

    def decorate(function):
        class DeferredFunction:
            def __init__(self):
                self._compiled = None
                self._lock = RLock()
                update_wrapper(self, function)

            def _resolve(self):
                if self._compiled is not None:
                    return self._compiled
                with self._lock:
                    if self._compiled is None:
                        self._compiled = import_module('tensorflow').function(function, **options)
                    return self._compiled

            def __call__(self, *args, **kwargs):
                return self._resolve()(*args, **kwargs)

            def __getattr__(self, name):
                return getattr(self._resolve(), name)

        return DeferredFunction()
    return decorate
