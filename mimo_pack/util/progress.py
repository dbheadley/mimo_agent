"""Stacked progress bars for nested loops.

``Progress`` is a drop-in replacement for ``tqdm`` (same arguments, including
manual ``total=`` / ``update()`` use and context-manager use). A bar created
while any other ``tqdm`` bar is active (including plain ``tqdm`` bars in a
notebook) becomes a nested bar: it is drawn below its parent and removed when
finished, so only the top-level bar remains.
Bars nested deeper than ``MAX_DEPTH`` are silently disabled.
"""

from tqdm.auto import tqdm

MAX_DEPTH = 3


class Progress(tqdm):
    """``tqdm.auto`` bar that stacks automatically when nested.

    Parameters
    ----------
    *args, **kwargs
        Passed to ``tqdm.auto.tqdm``. ``leave`` defaults to ``True`` for the
        top-level bar and ``False`` for nested bars; pass it explicitly to
        override.

    Notes
    -----
    Set ``mimo_pack.util.progress.MAX_DEPTH`` to change how many levels are
    displayed (default 3). Works in terminals (bars stack by position) and in
    Jupyter (each bar is its own widget that disappears on completion).

    Examples
    --------
    >>> from mimo_pack.util.progress import Progress
    >>> for unit in Progress(range(10), desc='Units'):
    ...     for trial in Progress(range(100), desc='Trials'):
    ...         pass
    """

    def __init__(self, *args, **kwargs):
        # tqdm registers every live bar, including plain ``tqdm`` bars created
        # outside this module (e.g. a notebook's top-level loop); exclude self.
        depth = len(tqdm._instances) - 1
        if depth >= MAX_DEPTH:
            kwargs['disable'] = True
        kwargs.setdefault('leave', depth == 0)
        super().__init__(*args, **kwargs)
