"""Installed and ``python -m sr_tool`` entry point."""

import tkinter as tk

from sr_tool.gui.app import Application


def main() -> None:
    root = tk.Tk()
    Application(root)
    root.mainloop()


if __name__ == "__main__":
    main()
