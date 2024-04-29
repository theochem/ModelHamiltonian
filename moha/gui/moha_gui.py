import tkinter as tk
from tkinter import ttk
import tomllib
from PIL import ImageTk
from rdkit import Chem
from rdkit.Chem import Draw

# import 'Sun Valley' theme
# created by rdbende at https://github.com/rdbende/Sun-Valley-ttk-theme
import sv_ttk

import sys
sys.path.append("../../.")
from moha.toml_tools import *

from gui_utils import *

# Suppress RDKit messages
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Examples 
# CID 42626654
# smiles_entry = C1=CC=C2C=CC=CC2=C1
# molfile_path = "../../examples/mol/Fe8S7.mol"

#----- MOLECULE -----#

def build_molecule(entries):
    # Get moltype and field entry
    mol_field = entries[0][0]
    mol_entry = entries[0][1].get()
    # Print message if mol field is empty
    if mol_entry == "":
        print("Need to specify molecule first!")
        return
    # Initialize mol object from SMILES
    elif mol_field == "SMILES":
        print("Created molecule from SMILES: " + mol_entry)
        state_data["system"]["moltype"] = "smiles"
        state_data["system"]["smiles"] = mol_entry
        mol = Chem.MolFromSmiles(mol_entry)
    # Initialize mol object from Molfile
    elif mol_field == "Molfile":
        print("Created molecule from molfile: " + mol_entry)
        state_data["system"]["moltype"] = "molfile"
        state_data["system"]["molfile"] = mol_entry
        mol = Chem.MolFromMolFile(mol_entry)
    # Error if dropdown option is not implemented
    else:
        raise ValueError("Invalid moltype option")

    if mol:
        # Convert the molecule to an editable form
        new_mol = Chem.RWMol(mol)

        # Identify and set symbolic bonds
        symbolic_bonds = [bond.GetIdx() for bond in new_mol.GetBonds() 
                          if bond.GetBondTypeAsDouble() == 0]
        for bond_idx in symbolic_bonds:
            new_mol.GetBondWithIdx(bond_idx).SetBondType(Chem.BondType.ZERO)

        # Convert the editable molecule back to a standard molecule
        new_mol = new_mol.GetMol()

        # Get the size of the mol frame
        mol_frame_width = mol_frame.winfo_width()-14
        mol_frame_height = mol_frame.winfo_height()

        # Create mol image
        img = Draw.MolToImage(new_mol, size=(mol_frame_width, mol_frame_height))
        img_tk = ImageTk.PhotoImage(img)

        # Clear any previous content from the mol_frame
        destroy_widgets(mol_frame.winfo_children())

        # Create a canvas in the mol_frame and display mol
        canvas = tk.Canvas(mol_frame, width=mol_frame_width, height=mol_frame_height)
        canvas.img = img_tk
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        canvas.pack()
    else:
        print("Invalid Entry: " + mol_entry + 
              " for moltype: " + mol_field)

def on_moltype_dropdown_select(event, ents, selected):
    moltype = selected.get()
    modified_ent = list(ents[0])
    modified_ent[0] = moltype
    ents[0] = modified_ent
    state_data["system"]["moltype"] = moltype

#----- MODEL -----#

def make_model_form(root, widgets, fields=[], ents=[]):
    destroy_widgets(widgets)
    ents.clear()
    for field in fields:
        # Create a frame for other fields
        row = ttk.Frame(root)
        row.pack(pady=5, fill=tk.X)
        widgets.append(row)

        # Create a label for the field
        lab = ttk.Label(row, width=15, text=field)
        lab.pack(side=tk.LEFT)
        widgets.append(lab)

        # Create an Entry widget
        ent = ttk.Entry(row)
        ent.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
        widgets.append(ent)

        # Append the entry to entries list
        ents.append((field, ent))

    return ents

def set_model(entries):
    for entry in entries:
        field = entry[0]
        value = entry[1].get()
        if value == '':
            state_data["model"][field] = 0
        else:
            state_data["model"][field] = value

def on_ham_dropdown_select(event, root, widgets, ents):
    # Get the selected value from the Combobox
    selected_ham_value = ham_dropdown.get()

    # Select the appropriate fields based on the selected value
    ppp_fields = ["alpha", "beta", "charge"]
    hubbard_fields = ["alpha", "beta", "u_onsite"]
    huckel_fields = ["alpha", "beta"]
    heisenberg_fields = ["J_eq", "J_ax", "mu"]
    ising_fields = ["J_ax", "mu"]
    rg_fields = ["J_eq", "mu"]

    fields = []
    if selected_ham_value == "PPP":
        fields = ppp_fields
    elif selected_ham_value == "Hubbard":
        fields = hubbard_fields
    elif selected_ham_value == "Huckel":
        fields = huckel_fields
    elif selected_ham_value == "Heisenberg":
        fields = heisenberg_fields
    elif selected_ham_value == "Ising":
        fields = ising_fields
    elif selected_ham_value == "RG":
        fields = rg_fields
    # Add other supported hamiltonians as needed

    ents = make_model_form(root, widgets, fields, ents)
    ents.append(("hamiltonian", ham_dropdown))

#----- CONTROL -----#

def make_control_form(root):
    entries = []
    fields = ["outdir", "prefix"]
    for field in fields:
        if field == "outdir":
            y_offset = 60
        else:
            y_offset = 5
        row = ttk.Frame(root)
        lab = ttk.Label(row, width=15, text=field, anchor='w')
        ent = ttk.Entry(row)
        row.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, y_offset))
        lab.pack(side=tk.LEFT)
        ent.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
        entries.append((field, ent))
    return entries

def set_control(entries):
    for entry in entries:
        field = entry[0]
        value = entry[1].get()
        if value == "":
            continue
        else:
            state_data["control"][field.lower()] = value

def on_save_format_dropdown_select(event, ents):
    modified_ent = list(ents[0])
    modified_ent[0] = selected_save_format.get()
    state_data["control"]["integral_format"] = selected_save_format.get().lower()
    set_control(ents)

def save_integrals(entries):
    state_data["control"]["save_integrals"] = True
    set_model(model_ents)
    set_control(control_ents)
    print(state_data)
    if state_data["system"]["moltype"] == "molfile":
        dict_to_ham(state_data)
        print("Integrals have been saved!")
    else:
        print("Need to build molecule first!")

#----- SAVE/QUIT BUTTONS -----#
def make_save_quit_buttons(frame):
    # Create a frame for the buttons at the bottom right corner
    right_button_frame = tk.Frame(frame)
    right_button_frame.place(relx=1, rely=1, anchor='se')

    # Position the "Quit" button
    quit_button = ttk.Button(right_button_frame, 
                             text='Quit', 
                             command=root.quit)
    quit_button.pack(side=tk.RIGHT, padx=5)

    # Position the "Save Integrals" button
    save_integrals_button = ttk.Button(right_button_frame,
                                       text='Save Integrals',
                                       command=(lambda e=model_ents: save_integrals(e)))
    save_integrals_button.pack(side=tk.RIGHT, padx=5)


if __name__ == '__main__':
    required_default_paramfile = "../../moha/toml_tools/defaults.toml"
    state_data = tomllib.load(open(required_default_paramfile, "rb"))

    format_options = {
        "title_font" : ("Arial", 12, "bold"),
    }

    root = tk.Tk()
    root.title("ModelHamiltonian GUI")

    # comment/uncomment to disable/enable window resizing
    #root.resizable(width=False, height=False)

    # Set the window width and height
    root.geometry("1100x500")

    # Create a box to the left of the form
    mol_frame = tk.Frame(root, background="white", borderwidth=5, relief=tk.SOLID, width=640)
    mol_frame.pack(side=tk.LEFT, fill=tk.BOTH, pady=(30, 30), padx=(30,30))

    # Create a red border for the left box
    mol_frame.config(highlightbackground="black", highlightcolor="black",)

    # Create a frame for the title and fields on the right side
    right_frame = tk.Frame(root, width=200)
    right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,30), pady=(30,30))

    #-------- MOLECULE FORM --------#

    # Add a title above the fields
    mol_title = "Molecule"
    mol_title_frame = make_title(right_frame, mol_title)

    # Create a frame for the molecule form
    mol_form_frame = tk.Frame(right_frame)
    mol_form_frame.pack(fill=tk.X)

    mol_widgets = []
    mol_fields = ["Molfile", "SMILES"]
    mol_ents = []

    row = ttk.Frame(mol_form_frame)
    row.pack(fill=tk.X, pady=(5,0))

    # Create a Combobox widget for mol input
    selected_moltype = tk.StringVar(value=mol_fields[0])
    moltype_dropdown = ttk.Combobox(row, values=mol_fields, textvariable=selected_moltype, width=6)

    moltype_dropdown.pack(side=tk.LEFT, padx=(0,20))

    ent = ttk.Entry(row)
    ent.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
    mol_ents.append((mol_fields[0], ent))

    moltype_dropdown.bind("<<ComboboxSelected>>", (lambda event, 
                                                   ents=mol_ents,
                                                   selected=selected_moltype:
                                                   on_moltype_dropdown_select(event, ents, selected)))

    # Create the "Build Molecule" button
    build_mol_button = ttk.Button(mol_title_frame, 
                                  text='Build Molecule', 
                                  command=(lambda e=mol_ents: build_molecule(e)))
    build_mol_button.pack(side=tk.RIGHT)

    #-------- MODEL FORM --------#

    # Add a title above the fields
    model_title = "Model"
    model_title_frame = make_title(right_frame, model_title, pady=(20,0))

    # Initialize model entries
    model_widgets = []
    model_ents = make_model_form(right_frame, model_widgets)

    # Create a frame for the dropdown
    ham_dropdown_frame = ttk.Frame(model_title_frame)
    ham_dropdown_frame.pack(side=tk.RIGHT)

    # Create a Combobox widget for hamiltonian
    selected_item = tk.StringVar()
    ham_dropdown = ttk.Combobox(ham_dropdown_frame, textvariable=selected_item, width=20)
    ham_dropdown['values'] = ["PPP", "Hubbard", "Huckel", "Heisenberg", "Ising", "RG"]

    # Set the prompt text
    ham_dropdown_prompt_text = "Select Hamiltonian"
    set_prompt(ham_dropdown, ham_dropdown_prompt_text)

    ham_dropdown.pack(side=tk.RIGHT)

    ham_dropdown.bind("<<ComboboxSelected>>", (lambda event, 
                                               root=right_frame, 
                                               widgets=model_widgets, 
                                               ents=model_ents: 
                                               on_ham_dropdown_select(event, root, widgets, ents)))

    #-------- CONTROL FORM --------#

    # Create control fields starting from bottom of right_frame
    control_ents = make_control_form(right_frame)

    # Add a title above the fields
    control_title = "Output"
    control_title_frame = make_title(right_frame, control_title, side=tk.BOTTOM)

    # Create a frame for the dropdown
    save_format_dropdown_frame = ttk.Frame(control_title_frame)
    save_format_dropdown_frame.pack(side=tk.RIGHT)

    # Create a Combobox widget for the save format
    selected_save_format = tk.StringVar()
    save_format_dropdown = ttk.Combobox(save_format_dropdown_frame, 
                                        textvariable=selected_save_format, 
                                        width=20)
    save_format_dropdown['values'] = ["FCIDUMP", "npz"]

    # Set the prompt text
    save_format_prompt_text = "Select Save Format"
    set_prompt(save_format_dropdown, save_format_prompt_text)

    save_format_dropdown.pack(side=tk.RIGHT)

    save_format_dropdown.bind("<<ComboboxSelected>>", 
                              (lambda event,
                               ents=control_ents:
                               on_save_format_dropdown_select(event, ents)))

    #-------- SAVE/QUIT BUTTONS --------#

    make_save_quit_buttons(right_frame)

    #-------- KEYBINDS --------#
    root.bind('<Return>', (lambda event, e=mol_ents: build_molecule(e)))
    root.bind('<Escape>', (lambda event, : root.quit()))

    # DEBUG BINDINGS
    #root.bind('<1>', (lambda event, : print(model_ents)))

    #-------- THEME --------#
    sv_ttk.set_theme("dark")

    #--------
    root.mainloop()
