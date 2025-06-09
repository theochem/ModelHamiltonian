r"""Model Hamiltonian GUI."""

import tkinter as tk
import toml
import sv_ttk
from tkinter import ttk
from PIL import ImageTk
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDLogger
from moha.toml_tools import *
from gui_utils import *
from pathlib import Path
# Uses 'Sun Valley' theme from sv_ttk
# Created by rdbende at https://github.com/rdbende/Sun-Valley-ttk-theme

# Suppress RDKit messages
RDLogger.DisableLog('rdApp.*')

# ----- MOLECULE -----#


def build_molecule(ents):
    r"""
    Build molecule from one of molfile or smiles moltyes.

    Parameters
    ----------
    ents: list
        list of molecule form entries

    Notes
    -----
    smiles_entry: C1=CC=C2C=CC=CC2=C1
    molfile_path: "../../examples/mol/Fe8S7.mol"
    """
    # Get moltype and field entry
    mol_field = ents[0][0]
    mol_entry = ents[0][1].get()
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
        mol_frame_width = mol_frame.winfo_width() - 14
        mol_frame_height = mol_frame.winfo_height()

        # Create mol image
        img = Draw.MolToImage(new_mol,
                              size=(mol_frame_width, mol_frame_height))
        img_tk = ImageTk.PhotoImage(img)

        # Clear any previous content from the mol_frame
        destroy_widgets(mol_frame.winfo_children())

        # Create a canvas in the mol_frame and display mol
        canvas = tk.Canvas(mol_frame,
                           width=mol_frame_width,
                           height=mol_frame_height)
        canvas.img = img_tk
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        canvas.pack()
    else:
        print("Invalid Entry: " + mol_entry + " for moltype: " + mol_field)


def make_moltype_fields(frame, fields, ents):
    r"""
    Make fields relevant to moltype in molecule form.

    Parameters
    ----------
    frame: tk.Frame
        the frame to display the fields within
    fields: list
        the fields relevant to moltype
    ents: list
        list of molecule form entries
    """
    mol_form_frame = tk.Frame(frame)
    mol_form_frame.pack(fill=tk.X)

    row = ttk.Frame(mol_form_frame)
    row.pack(fill=tk.X, pady=(5, 0))

    ent = ttk.Entry(row)
    ent.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
    ents.append((fields[0], ent))

    # Create a Combobox widget for mol input
    selected_moltype = tk.StringVar(value=fields[0])
    moltype_dropdown = ttk.Combobox(row,
                                    values=fields,
                                    textvariable=selected_moltype,
                                    width=6)
    moltype_dropdown.pack(side=tk.LEFT, padx=(0, 20))

    # Add dropdown entry to ents
    ents.append(("moltype", moltype_dropdown))

    moltype_dropdown.bind("<<ComboboxSelected>>",
                          (lambda event,
                           ents=ents,
                           selected=selected_moltype:
                           on_moltype_dropdown_select(ents, selected)))


def on_moltype_dropdown_select(ents, selected):
    r"""
    Build molecule from one of molfile or smiles moltyes.

    Parameters
    ----------
    ents: list
        list of molecule form entries
    """
    moltype = selected.get()
    modified_ent = list(ents[0])
    modified_ent[0] = moltype
    ents[0] = modified_ent
    state_data["system"]["moltype"] = moltype


def make_molecule_form(frame, fields, ents):
    r"""
    Make form for molecule entries.

    Parameters
    ----------
    frame: tk.Frame
        the frame to display the form within
    fields: list
        the fields relevant to moltype
    ents: list
        list of molecule form entries
    """
    # Add a title above the fields
    mol_title = "Molecule"
    mol_title_frame = make_title(frame, mol_title)

    # Create the "Build Molecule" button
    build_mol_button = ttk.Button(mol_title_frame,
                                  text='Build Molecule',
                                  command=(lambda ents=ents:
                                           build_molecule(ents)))
    build_mol_button.pack(side=tk.RIGHT)

    make_moltype_fields(frame, fields, ents)

# ----- MODEL -----#


def make_model_fields(frame, widgets, fields=[], ents=[]):
    r"""
    Make fields relevant to selected hamiltonian in model form.

    Parameters
    ----------
    frame: tk.Frame
        the frame to display the fields within
    widgets: list
        widgets associated with all hamiltonian field elements
    fields: list
        the fields relevant to the hamiltonian
    ents: list
        list of model form entries
    """
    destroy_widgets(widgets)
    if ents != []:
        ham_ent = ents[0]
        ents.clear()
        ents.append(ham_ent)
    for field in fields:
        # Create a frame for other fields
        row = ttk.Frame(frame)
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


def set_model(ents):
    r"""
    Set the model elements of state_data.

    Parameters
    ----------
    ents: list
        list of model form entries
    """
    for ent in ents:
        field = ent[0]
        value = ent[1].get()
        if value == '':
            state_data["model"][field] = 0
        else:
            state_data["model"][field] = value


def on_ham_dropdown_select(frame, widgets, ents, selected):
    r"""
    Reset hamiltonian fields on dropdown selection.

    Parameters
    ----------
    frame: tk.Frame
        the frame to display the fields within
    widgets: list
        widgets associated with all hamiltonian field elements
    ents: list
        list of model form entries
    selected: tk.StringVar
        selected value of the hamiltonian dropdown
    """
    # Get the selected value from the Combobox
    selected_ham_value = selected.get()

    # Select the appropriate fields based on the selected value
    ppp_fields = ["alpha", "beta", "charge", "u_onsite"]
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

    make_model_fields(frame, widgets, fields, ents)


def make_model_form(frame, widgets, ents):
    r"""
    Make form for model entries.

    Parameters
    ----------
    frame: tk.Frame
        the frame to display the form within
    widgets: list
        widgets associated with all hamiltonian field elements
    ents: list
        list of model form entries
    """
    # Add a title above the fields
    model_title = "Model"
    model_title_frame = make_title(frame, model_title, pady=(20, 0))

    # Initialize model entries
    make_model_fields(frame, widgets, ents)

    # Create a frame for the dropdown
    ham_dropdown_frame = ttk.Frame(model_title_frame)
    ham_dropdown_frame.pack(side=tk.RIGHT)

    # Create a Combobox widget for hamiltonian
    selected_item = tk.StringVar()
    ham_dropdown = ttk.Combobox(ham_dropdown_frame,
                                textvariable=selected_item,
                                width=20)
    ham_dropdown['values'] = [
        "PPP",
        "Hubbard",
        "Huckel",
        "Heisenberg",
        "Ising",
        "RG"
    ]

    # Set the prompt text
    ham_dropdown_prompt_text = "Select Hamiltonian"
    set_prompt(ham_dropdown, ham_dropdown_prompt_text)

    ham_dropdown.pack(side=tk.RIGHT)

    ham_dropdown.bind("<<ComboboxSelected>>",
                      (lambda event,
                       frame=frame,
                       widgets=widgets,
                       ents=ents,
                       selected=selected_item:
                       on_ham_dropdown_select(frame,
                                              widgets,
                                              ents,
                                              selected)))

    ents.append(("hamiltonian", ham_dropdown))

# ----- CONTROL -----#


def make_control_fields(frame, ents):
    r"""
    Make fields relevant to the control/output form.

    Parameters
    ----------
    frame: tk.Frame
        the frame to display the fields within
    ents: list
        list of control form entries
    """
    fields = ["outdir", "prefix"]
    for field in fields:
        if field == fields[0]:
            y_offset = 60
        else:
            y_offset = 5
        row = ttk.Frame(frame)
        row.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, y_offset))

        lab = ttk.Label(row, width=15, text=field, anchor='w')
        lab.pack(side=tk.LEFT)

        ent = ttk.Entry(row)
        ent.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)

        ents.append((field, ent))


def set_control(ents):
    r"""
    Set the control elements of state_data.

    Parameters
    ----------
    ents: list
    list of control form entries
    """
    for entry in ents:
        field = entry[0]
        value = entry[1].get()
        if value == "":
            continue
        else:
            state_data["control"][field] = value.lower()


def make_control_form(frame, ents):
    r"""
    Make form for control/output entries.

    Parameters
    ----------
    frame: tk.Frame
        the frame to display the form within
    ents: list
        list of control form entries
    """
    # Create control fields starting from bottom of field_frame
    make_control_fields(frame, ents)

    # Add a title above the fields
    control_title = "Output"
    control_title_frame = make_title(frame, control_title, side=tk.BOTTOM)

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
                               ents=ents,
                               selected=selected_save_format:
                               set_control(ents)))

    ents.append(("integral_format", save_format_dropdown))


def save_integrals():
    r"""Save integrals to file specified in control fields."""
    state_data["control"]["save_integrals"] = True
    set_model(model_ents)
    set_control(control_ents)
    if state_data["system"]["moltype"] == "molfile":
        dict_to_ham(state_data)
        print("Integrals have been saved!")
    elif state_data["system"]["moltype"] == "smiles":
        dict_to_ham(state_data)
        print("Integrals have been saved!")
    else:
        print("Need to build molecule first!")

# ----- SAVE/QUIT BUTTONS -----#


def make_save_quit_buttons(frame):
    r"""
    Make buttons for save integrals and quit.

    Parameters
    ----------
    frame: tk.Frame
        the frame to display the buttons within
    """
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
                                       command=save_integrals)
    save_integrals_button.pack(side=tk.RIGHT, padx=5)


if __name__ == '__main__':
    required_default_paramfile = Path(__file__).parent.\
        parent / "toml_tools" / "defaults.toml"
    state_data = toml.load(required_default_paramfile)

    root = tk.Tk()
    root.title("ModelHamiltonian GUI")

    # comment/uncomment to disable/enable window resizing
    # root.resizable(width=False, height=False)

    # Set the window width and height
    root.geometry("1100x520")

    # Make left frame to display molecule with rdkit
    mol_frame = tk.Frame(root,
                         background="white",
                         borderwidth=5,
                         relief=tk.SOLID,
                         width=640)
    mol_frame.pack(side=tk.LEFT,
                   fill=tk.BOTH,
                   pady=(30, 30),
                   padx=(30, 30))

    # Make right frame to contain input fields
    field_frame = tk.Frame(root, width=200)
    field_frame.pack(side=tk.LEFT,
                     fill=tk.BOTH,
                     expand=True,
                     padx=(0, 30),
                     pady=(30, 30))

    # Make Molecule form

    mol_fields = ["Molfile", "SMILES"]
    mol_ents = []

    make_molecule_form(field_frame, mol_fields, mol_ents)

    # Make Model form

    model_ents = []
    model_widgets = []
    make_model_form(field_frame, model_widgets, model_ents)

    # Make Output/Control form

    control_ents = []
    make_control_form(field_frame, control_ents)

    # Make Save/Quit buttons

    make_save_quit_buttons(field_frame)

    # Set keybinds
    root.bind('<Return>', (lambda event, ents=mol_ents: build_molecule(ents)))
    root.bind('<Escape>', (lambda event, : root.quit()))

    # Debug bindings
    # root.bind('<1>', (lambda event, : print(control_ents)))

    # Set "Sun Valley" theme
    sv_ttk.set_theme("dark")

    # Start tkinter event loop
    root.mainloop()
