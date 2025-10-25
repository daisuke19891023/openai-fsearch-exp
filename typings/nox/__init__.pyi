
from typing import Any
from collections.abc import Callable, Iterable

from .sessions import Session


class _Options:
    default_venv_backend: str
    reuse_existing_virtualenvs: bool


options: _Options


def session(
    *,
    python: Iterable[str] | None = ...,
    tags: Iterable[str] | None = ...,
    venv_backend: str | None = ...,
) -> Callable[[Callable[[Session], Any]], Callable[[Session], Any]]: ...
