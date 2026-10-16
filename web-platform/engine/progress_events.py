"""Request-scoped progress; no private reasoning or credentials are emitted."""
from contextvars import ContextVar
from contextlib import contextmanager
import time

class ClientDisconnected(RuntimeError):pass
_scope=ContextVar('mfl_progress',default=None)

@contextmanager
def reporting(callback):
    token=_scope.set((callback,time.monotonic()))
    try:yield
    finally:_scope.reset(token)

def notify(stage,**fields):
    scope=_scope.get()
    if scope is not None:
        callback,began=scope
        callback(dict(stage=stage,elapsed_s=round(time.monotonic()-began,2),**fields))
