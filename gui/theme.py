from __future__ import annotations

from tkinter import ttk


COLORS = {
    "bg": "#e8edf5",
    "surface": "#f7f9fc",
    "surface_alt": "#ffffff",
    "surface_hover": "#e6eafb",
    "input": "#f3f5fa",
    "border": "#c8d0df",
    "border_strong": "#aeb9cc",
    "text": "#1d2738",
    "muted": "#66738a",
    "accent": "#5b5bd6",
    "accent_hover": "#4848bd",
    "selection": "#d9ddff",
    "success": "#16845b",
    "warning": "#b36b12",
    "danger": "#c53d50",
    "info": "#2374ab",
    "disabled_bg": "#e1e6ef",
    "disabled_text": "#929db0",
}

SPACE = {"xs": 4, "sm": 8, "md": 16, "lg": 24, "xl": 32}


def apply_theme(root) -> None:
    """Apply the light slate theme, including native combobox popdown colors."""
    root.configure(bg=COLORS["bg"])
    style = ttk.Style(root)
    style.theme_use("clam")

    style.configure(
        ".",
        background=COLORS["bg"],
        foreground=COLORS["text"],
        font=("TkDefaultFont", 11),
        bordercolor=COLORS["border"],
        lightcolor=COLORS["border"],
        darkcolor=COLORS["border"],
        focuscolor=COLORS["accent"],
    )
    style.configure("TFrame", background=COLORS["bg"])
    style.configure("Surface.TFrame", background=COLORS["surface"])
    style.configure("Card.TFrame", background=COLORS["surface_alt"], relief="flat")
    style.configure("TLabel", background=COLORS["bg"], foreground=COLORS["text"])
    style.configure("Surface.TLabel", background=COLORS["surface"], foreground=COLORS["text"])
    style.configure("SurfaceMuted.TLabel", background=COLORS["surface"], foreground=COLORS["muted"])
    style.configure("Title.TLabel", font=("TkDefaultFont", 20, "bold"))
    style.configure(
        "Section.TLabel",
        font=("TkDefaultFont", 13, "bold"),
        background=COLORS["surface_alt"],
    )
    style.configure("Muted.TLabel", foreground=COLORS["muted"])
    style.configure("Card.TLabel", background=COLORS["surface_alt"], foreground=COLORS["text"])

    style.configure(
        "TButton",
        background=COLORS["surface_alt"],
        foreground=COLORS["text"],
        padding=(12, 8),
        borderwidth=1,
        relief="flat",
    )
    style.map(
        "TButton",
        background=[
            ("disabled", COLORS["disabled_bg"]),
            ("pressed", COLORS["selection"]),
            ("active", COLORS["surface_hover"]),
        ],
        foreground=[("disabled", COLORS["disabled_text"])],
        bordercolor=[("focus", COLORS["accent"])],
    )
    style.configure(
        "Nav.TButton",
        anchor="w",
        padding=(16, 12),
        background=COLORS["surface"],
        borderwidth=0,
    )
    style.map(
        "Nav.TButton",
        background=[("pressed", COLORS["selection"]), ("active", COLORS["surface_hover"])],
        foreground=[("pressed", COLORS["accent_hover"])],
    )
    style.configure(
        "Accent.TButton",
        background=COLORS["accent"],
        foreground="#ffffff",
        padding=(14, 8),
    )
    style.map(
        "Accent.TButton",
        background=[
            ("disabled", COLORS["disabled_bg"]),
            ("pressed", COLORS["accent_hover"]),
            ("active", COLORS["accent_hover"]),
        ],
        foreground=[("disabled", COLORS["disabled_text"]), ("!disabled", "#ffffff")],
    )
    style.configure("Danger.TButton", background=COLORS["danger"], foreground="#ffffff")
    style.map(
        "Danger.TButton",
        background=[("disabled", COLORS["disabled_bg"]), ("active", "#a82f41")],
        foreground=[("disabled", COLORS["disabled_text"]), ("!disabled", "#ffffff")],
    )

    # Explicit field colors prevent platform themes from producing white-on-white text.
    style.configure(
        "TEntry",
        padding=7,
        fieldbackground=COLORS["input"],
        foreground=COLORS["text"],
        insertcolor=COLORS["text"],
        bordercolor=COLORS["border_strong"],
    )
    style.map(
        "TEntry",
        fieldbackground=[("disabled", COLORS["disabled_bg"]), ("readonly", COLORS["input"])],
        foreground=[("disabled", COLORS["disabled_text"]), ("readonly", COLORS["text"])],
        bordercolor=[("focus", COLORS["accent"])],
    )
    style.configure(
        "TCombobox",
        padding=6,
        fieldbackground=COLORS["input"],
        background=COLORS["input"],
        foreground=COLORS["text"],
        arrowcolor=COLORS["text"],
        bordercolor=COLORS["border_strong"],
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", COLORS["input"]), ("disabled", COLORS["disabled_bg"])],
        background=[("readonly", COLORS["input"]), ("active", COLORS["surface_hover"])],
        foreground=[("readonly", COLORS["text"]), ("disabled", COLORS["disabled_text"])],
        selectbackground=[("readonly", COLORS["input"])],
        selectforeground=[("readonly", COLORS["text"])],
        bordercolor=[("focus", COLORS["accent"])],
        arrowcolor=[("disabled", COLORS["disabled_text"]), ("!disabled", COLORS["text"])],
    )
    style.configure("TCheckbutton", background=COLORS["surface_alt"], foreground=COLORS["text"])
    style.map(
        "TCheckbutton",
        background=[("active", COLORS["surface_alt"])],
        foreground=[("disabled", COLORS["disabled_text"])],
        indicatorcolor=[("selected", COLORS["accent"]), ("!selected", COLORS["input"])],
    )

    style.configure(
        "Treeview",
        rowheight=30,
        background=COLORS["surface_alt"],
        fieldbackground=COLORS["surface_alt"],
        foreground=COLORS["text"],
        bordercolor=COLORS["border"],
    )
    style.map(
        "Treeview",
        background=[("selected", COLORS["selection"])],
        foreground=[("selected", COLORS["text"])],
    )
    style.configure(
        "Treeview.Heading",
        background=COLORS["surface"],
        foreground=COLORS["text"],
        padding=(8, 7),
    )
    style.map("Treeview.Heading", background=[("active", COLORS["surface_hover"])])
    style.configure("TScrollbar", background=COLORS["border"], troughcolor=COLORS["surface"])
    style.configure("TPanedwindow", background=COLORS["bg"])

    root.option_add("*TCombobox*Listbox.background", COLORS["surface_alt"])
    root.option_add("*TCombobox*Listbox.foreground", COLORS["text"])
    root.option_add("*TCombobox*Listbox.selectBackground", COLORS["selection"])
    root.option_add("*TCombobox*Listbox.selectForeground", COLORS["text"])
    root.option_add("*TCombobox*Listbox.font", "TkDefaultFont 11")
    root.option_add("*Listbox.background", COLORS["surface_alt"])
    root.option_add("*Listbox.foreground", COLORS["text"])
    root.option_add("*Listbox.selectBackground", COLORS["selection"])
    root.option_add("*Listbox.selectForeground", COLORS["text"])
