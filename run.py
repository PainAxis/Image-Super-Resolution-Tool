"""Launch script for FSR 1.0 Image Super-Resolution tool."""

import sys
import os

# Ensure fsr_tool is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fsr_tool.gui.app import Application
import tkinter as tk


def main():
    root = tk.Tk()
    Application(root)
    root.mainloop()


if __name__ == "__main__":
    main()
