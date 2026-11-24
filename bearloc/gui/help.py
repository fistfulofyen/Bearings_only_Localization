"""In-program help: "?" icons with hover tooltips, and the Help window.

* :class:`HelpIcon` -- a small round "?" placed next to an input or result.
  Hovering shows the topic's short ``tip`` in a :class:`Tooltip`; clicking
  opens the :class:`HelpWindow` on that topic.
* :class:`HelpWindow` -- one per main window: a topic tree on the left and
  the full text on the right.  Also opened with F1 / Help > Help topics.
* :func:`titled_frame` -- a ``ttk.LabelFrame`` whose title carries a "?".

The texts themselves live in :mod:`bearloc.gui.help_text`.
"""

from __future__ import annotations

import base64
import io
import tkinter as tk
from tkinter import ttk

from .help_text import HelpTopic

#: Colours of the "?" icon (normal, mouse over).
_ICON_COLOURS = ((52, 101, 164), (25, 64, 120))


def _scale(widget: tk.Misc) -> float:
    """Screen scaling factor relative to 96 dpi."""
    return widget.winfo_fpixels("1i") / 96.0


def _icon_images(widget: tk.Misc) -> tuple[tk.PhotoImage, tk.PhotoImage]:
    """The (normal, hover) "?" images, drawn once per application and cached.

    Drawn with Pillow at 4x size and scaled down for smooth edges; the
    transparent background blends with any widget colour.
    """
    root = widget.winfo_toplevel()
    cached = getattr(root, "_help_icon_images", None)
    if cached is not None:
        return cached
    from PIL import Image, ImageDraw, ImageFont

    size = max(14, round(16 * _scale(root)))
    big = 4 * size
    try:
        font = ImageFont.truetype("segoeuib.ttf", int(big * 0.72))
    except OSError:
        try:
            font = ImageFont.truetype("arialbd.ttf", int(big * 0.72))
        except OSError:
            font = ImageFont.load_default()
    images = []
    for colour in _ICON_COLOURS:
        img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.ellipse([2, 2, big - 3, big - 3], fill=(*colour, 255))
        d.text((big / 2, big / 2 + big * 0.02), "?", font=font, fill="white", anchor="mm")
        img = img.resize((size, size), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        images.append(tk.PhotoImage(master=root, data=base64.b64encode(buf.getvalue())))
    root._help_icon_images = tuple(images)  # keep references alive
    return root._help_icon_images


class Tooltip:
    """A small yellow note that appears while the mouse rests on ``widget``.

    Parameters
    ----------
    widget : tkinter widget
        Widget that triggers the tooltip.
    text : str
        Text to show (wrapped automatically).
    delay_ms : int
        Hover time before the note appears.
    """

    def __init__(self, widget: tk.Widget, text: str, delay_ms: int = 350):
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self.window: tk.Toplevel | None = None
        self._after: str | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self.hide, add="+")
        widget.bind("<ButtonPress>", self.hide, add="+")

    def _schedule(self, _event=None) -> None:
        self._cancel()
        self._after = self.widget.after(self.delay_ms, self.show)

    def _cancel(self) -> None:
        if self._after is not None:
            self.widget.after_cancel(self._after)
            self._after = None

    def show(self) -> None:
        """Display the tooltip next to the mouse pointer (kept on screen)."""
        self._after = None
        if self.window is not None or not self.widget.winfo_exists():
            return
        scale = _scale(self.widget)
        win = tk.Toplevel(self.widget)
        win.wm_overrideredirect(True)
        try:
            win.wm_attributes("-topmost", True)
        except tk.TclError:
            pass
        tk.Label(win, text=self.text, justify="left", wraplength=int(380 * scale),
                 background="#fffde7", foreground="#222", relief="solid", borderwidth=1,
                 font=("Segoe UI", 9), padx=8, pady=5).pack()
        win.update_idletasks()
        x = self.widget.winfo_pointerx() + int(14 * scale)
        y = self.widget.winfo_pointery() + int(18 * scale)
        x = min(x, self.widget.winfo_screenwidth() - win.winfo_reqwidth() - 4)
        y = min(y, self.widget.winfo_screenheight() - win.winfo_reqheight() - 4)
        win.wm_geometry(f"+{max(x, 0)}+{max(y, 0)}")
        self.window = win

    def hide(self, _event=None) -> None:
        """Remove the tooltip (and cancel a pending one)."""
        self._cancel()
        if self.window is not None:
            self.window.destroy()
            self.window = None


class HelpWindow:
    """Help browser: topic tree on the left, explanation on the right.

    Use :func:`open_help` rather than creating it directly, so that each main
    window has a single Help window that is re-used.
    """

    def __init__(self, master: tk.Misc, topics: dict[str, HelpTopic], title: str):
        self.topics = topics
        scale = _scale(master)
        self.win = tk.Toplevel(master)
        self.win.title(title)
        self.win.geometry(f"{int(860 * scale)}x{int(540 * scale)}")
        self.win.minsize(int(560 * scale), int(320 * scale))
        self.win.bind("<Escape>", lambda _e: self.close())
        self.win.protocol("WM_DELETE_WINDOW", self.close)

        bottom = ttk.Frame(self.win, padding=(8, 4, 8, 8))
        bottom.pack(side="bottom", fill="x")
        ttk.Button(bottom, text="Close", command=self.close).pack(side="right")

        paned = ttk.PanedWindow(self.win, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=8, pady=(8, 0))
        self.tree = ttk.Treeview(paned, show="tree", selectmode="browse")
        self.tree.column("#0", width=int(230 * scale))
        groups: dict[str, str] = {}
        for key, topic in topics.items():
            if topic.group not in groups:
                groups[topic.group] = self.tree.insert("", "end", text=topic.group, open=True)
            self.tree.insert(groups[topic.group], "end", iid=key, text=topic.title)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        frame = ttk.Frame(paned)
        self.text = tk.Text(frame, wrap="word", relief="flat", padx=14, pady=10,
                            font=("Segoe UI", 10), spacing2=2, cursor="arrow")
        ys = ttk.Scrollbar(frame, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=ys.set)
        self.text.pack(side="left", fill="both", expand=True)
        ys.pack(side="right", fill="y")
        self.text.tag_configure("title", font=("Segoe UI", 13, "bold"), spacing3=8)
        self.text.tag_configure("tip", font=("Segoe UI", 10, "italic"), foreground="#444", spacing3=10)
        paned.add(self.tree, weight=0)
        paned.add(frame, weight=1)
        self.key: str | None = None

    def show(self, key: str) -> None:
        """Display topic ``key`` and bring the window to the front."""
        topic = self.topics[key]
        self.key = key
        if tuple(self.tree.selection()) != (key,):
            self.tree.selection_set(key)
        self.tree.see(key)
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("end", topic.title + "\n", "title")
        self.text.insert("end", topic.tip + "\n", "tip")
        self.text.insert("end", topic.body)
        self.text.configure(state="disabled")
        self.win.deiconify()
        self.win.lift()
        self.win.focus_set()

    def _on_select(self, _event=None) -> None:
        sel = self.tree.selection()
        if sel and sel[0] in self.topics and sel[0] != self.key:
            self.show(sel[0])

    def exists(self) -> bool:
        """Whether the window is still open."""
        try:
            return bool(self.win.winfo_exists())
        except tk.TclError:
            return False

    def close(self) -> None:
        """Close the window."""
        if self.exists():
            self.win.destroy()


def open_help(widget: tk.Misc, topics: dict[str, HelpTopic], key: str | None = None,
              title: str = "Help") -> HelpWindow:
    """Show topic ``key`` (default: the first topic) in the main window's Help window.

    The Help window is created on first use and re-used afterwards.
    """
    root = widget.winfo_toplevel()
    window = getattr(root, "_help_window", None)
    if window is None or not window.exists() or window.topics is not topics:
        window = HelpWindow(root, topics, title)
        root._help_window = window
    window.show(key if key is not None else next(iter(topics)))
    return window


class HelpIcon(ttk.Label):
    """A round "?" that shows a tooltip on hover and opens the Help window on click.

    Parameters
    ----------
    parent : tkinter widget
        Parent widget.
    topics : dict
        Topic dictionary (e.g. ``BEAR_TOPICS``).
    key : str
        Topic to explain; must be a key of ``topics``.
    title : str
        Title of the Help window.

    Raises
    ------
    KeyError
        If ``key`` is not a known topic (caught early, when the window is built).
    """

    def __init__(self, parent: tk.Misc, topics: dict[str, HelpTopic], key: str, title: str = "Help"):
        topic = topics[key]
        self._images = _icon_images(parent)
        super().__init__(parent, image=self._images[0], cursor="hand2", takefocus=False)
        self.topics, self.key, self.help_title = topics, key, title
        self.tooltip = Tooltip(self, f"{topic.tip}\n\nClick for details.")
        self.bind("<Enter>", lambda _e: self.configure(image=self._images[1]), add="+")
        self.bind("<Leave>", lambda _e: self.configure(image=self._images[0]), add="+")
        self.bind("<Button-1>", self.open, add="+")

    def open(self, _event=None) -> HelpWindow:
        """Open the Help window on this icon's topic."""
        return open_help(self, self.topics, self.key, self.help_title)


def titled_frame(parent: tk.Misc, text: str, topics: dict[str, HelpTopic], key: str,
                 title: str = "Help") -> ttk.LabelFrame:
    """A ``ttk.LabelFrame`` whose title is ``text`` followed by a "?" icon."""
    frame = ttk.LabelFrame(parent)
    header = ttk.Frame(frame)
    ttk.Label(header, text=text).pack(side="left")
    HelpIcon(header, topics, key, title).pack(side="left", padx=(5, 0))
    frame.configure(labelwidget=header)
    return frame
