import logging
from .main import NDimInv

all = [NDimInv, ]

logging.getLogger(__name__).addHandler(logging.NullHandler())
