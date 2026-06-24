from ordinalclip.utils import Registry

GRAPH_ENCODERS = Registry("graph_encoders")

from . import gin, gcn  # noqa: F401
