from importlib.metadata import version

from graphnamlss.local_shape import CenteredLocalShapeFunctions, LocalShapeResult

__version__ = version("graphnamlss")

__all__ = ["CenteredLocalShapeFunctions", "LocalShapeResult", "__version__"]
