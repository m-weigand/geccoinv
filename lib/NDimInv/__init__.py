import logging
from .main import NDimInv as Inversion

all = [Inversion, ]

logging.getLogger(__name__).addHandler(logging.NullHandler())
