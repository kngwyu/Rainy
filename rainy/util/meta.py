try:
    from typing import GenericMeta

    class GenericNamedMeta(GenericMeta, NamedTupleMeta):
        pass
except ImportError:
    from typing import NamedTupleMeta  # type: ignore
    GenericNamedMeta = NamedTupleMeta  # type: ignore
