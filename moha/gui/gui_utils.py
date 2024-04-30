import tkinter as tk
from tkinter import ttk

__all__ = [
    "destroy_widgets",
    "enable_dropdown_on_click",
    "set_prompt",
    "make_title"
]

format_options = {
    "title_font" : ("Arial", 12, "bold"),
}

def destroy_widgets(widgets):
    '''
    Destroy list of widgets

    Parameters
    ----------
    widgets: list
        list of widgets to be destroyed
    '''
    # Destroy old form widgets if they exist
    for widget in widgets:
        widget.destroy()
    widgets = []

def enable_dropdown_on_click(dropdown, prompt_text):
    '''
    Enable dropdown and remove promt text on click

    Parameters
    ----------
    dropdown: tk.Dropdown
        dropdown object to be enabled
    prompt_text: str
        initial prompt in the dropdown
    '''
    if dropdown.get() == prompt_text:
        dropdown.state(["!disabled"])
        dropdown.set("")

def set_prompt(dropdown, prompt_text):
    '''
    Initialize dropdown text to default prompt

    Parameters
    ----------
    dropdown: tk.Dropdown
        dropdown object to be enabled
    prompt_text: str
        initial prompt in the dropdown
    '''
    # Initialize dropdown text to default prompt
    dropdown.state(["disabled"])
    dropdown.set(prompt_text)
    dropdown.state(["readonly"])
    dropdown.bind("<Button-1>", (lambda event,
                                 prompt_text = prompt_text,
                                 dropdown = dropdown:
                                 enable_dropdown_on_click(dropdown, prompt_text)))

def make_title(frame, title, pady=(0,0), side=tk.TOP):
    '''
    Make and display title in frame

    Parameters
    ----------
    frame: tk.Frame
        frame to make title within
    title: str
        the title to be displayed
    pady: tuple
        vertical padding of the title
    side: one of tk.TOP, tk.BOTTOM, tk.LEFT, tk.RIGHT
        the side which the title will be packed
    '''
    title_frame = tk.Frame(frame)
    title_frame.pack(side=side, fill=tk.X, pady=pady)
    title_label = ttk.Label(title_frame,
                            text=title,
                            font=format_options["title_font"])
    title_label.pack(side=tk.LEFT)

    return title_frame
