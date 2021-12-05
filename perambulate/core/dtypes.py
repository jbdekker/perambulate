from typing import cast
from typing import Type
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from perambulate import Condition


def create_abc_type(name, attr, comp):

    # https://github.com/python/mypy/issues/1006
    # error: 'classmethod' used with a non-method
    @classmethod  # type: ignore[misc]
    def _check(cls, inst) -> bool:
        return getattr(inst, attr, "_typ") in comp

    dct = {"__instancecheck__": _check, "__subclasscheck__": _check}
    meta = type("ABCBase", (type,), dct)
    return meta(name, (), dct)


ABCCondition = cast(
    "Type[Condition]", create_abc_type("ABCCondition", "_typ", ("condition",))
)
