
from typing import Any


class Session:
    python: str

    def run(self, *args: Any, **kwargs: Any) -> Any: ...

    def install(self, *args: Any, **kwargs: Any) -> Any: ...

    def skip(self, reason: str) -> None: ...

    def notify(self, target: str) -> None: ...
