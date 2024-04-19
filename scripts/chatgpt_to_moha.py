from toml_to_moha import *
import sys
sys.path.insert(0, '../')

import json
from openai import OpenAI

# uncomment to set your API key from text file
# with open('path/to/api_key.txt', 'r') as file:
#     api_key = file.read().replace('\n', '')

# Choose ChatGPT model version and setup client
GPT_MODEL = "gpt-3.5-turbo"
client = OpenAI()

# Define model parameters and common descriptions
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_system",
            "description": "Get the system parameters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "moltype": {
                        "type": "string",
                        "enum": ["1d"],
                        "description": "The site structure on which the model is defined.",
                    },
                    "bc": {
                        "type": "string",
                        "enum": ["open", "periodic"],
                        "description": "The boundary conditions. Prefer open if not specified.",
                    },
                    "norb": {
                        "type": "integer",
                        "description": "The number of spin-orbitals.",
                    },
                    "nelec": {
                        "type": "integer",
                        "description": "The number of electrons occupying some of the spin-orbitals.",
                    },
                    "symmetry": {
                        "type": "integer",
                        "enum": [1, 2, 4, 8],
                        "description": "Symmetry of the two-electron integrals. Prefer 4 if not specified.",
                    }
                },
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_model_hamiltonian",
            "description": "Get the model hamiltonian",
            "parameters": {
                "type": "object",
                "properties": {
                    "hamiltonian": {
                        "type": "string",
                        "enum": ["PPP", "Hubbard", "Huckel", "Heisenberg", "Ising", "RG"],
                        "description": "The name of the model, e.g. Hubbard.",
                    },
                    "basis": {
                        "type": "string",
                        "enum": ["spatial basis", "spinorbital basis"],
                        "description": "The basis type. Prefer spatial basis if not specified.",
                    },
                    "alpha": {
                        "type": "number",
                        "description": "The on-site interaction strength. \
                                        Assume all sites are equivalent if not specified. \
                                        Relevant to PPP, Hubbard, Huckel.",
                    },
                    "beta": {
                        "type": "number",
                        "description": "The resonance energy, hopping amplitude. \
                                        Assume all bonds are equivalent if not specified. \
                                        Relevant to PPP, Hubbard, Huckel.",
                    },
                    "u_onsite": {
                        "type": "number",
                        "description": "The on-site Coulomb interaction. \
                                        Assume all sites are equivalent if not specified. \
                                        Relevant to PPP, Hubbard.",
                    },
                    "gamma": {
                        "type": "number",
                        "description": "Parameter that specifies long-range Coulomb interaction. \
                                        Relevant to PPP.",
                    },
                    "mu": {
                        "type": "number",
                        "description": "Zeeman term or external magnetic field. \
                                        Relevant to Ising, RG, Heisenberg.",
                    },
                    "J_eq": {
                        "type": "number",
                        "description": "Coupling between neighbouring S_x and S_y spin components. \
                                        It enters the hamiltonian as J_eq*(S^x_iS^x_j + S^y_iS^y_j). \
                                        Relevant to RG, Heisenberg.",
                    },
                    "J_ax": {
                        "type": "number",
                        "description": "Coupling between neighbouring S_z spin components. \
                                        It enters the hamiltonian as J_ax*(S^z_iS^z_j). \
                                        Relevant to Ising, Heisenberg.",
                    },
                },
                "required": ["hamiltonian"],
            },
        }
    },
]

def chat_completion_request(messages, tools=None, tool_choice=None, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

def chatgpt_to_ham(input_string):
    '''
    Function for generating hamiltonian from toml file.
    Prints integrals to output file if specified in toml_file.

    Parameters
        ----------
        input_string: str
            user input ChatGPT command

    Returns
        -------
        moha.Ham
    '''

    messages = []
    messages.append({"role": "system", "content":
                     "Set all parameters relevant to specified model. Do not describe the model."})
    messages.append({"role": "user", "content": input_string})
    chat_response = chat_completion_request(
        messages, tools=tools
    )
    assistant_message = chat_response.choices[0].message
    messages.append(assistant_message)

    data = {}
    if assistant_message.tool_calls[0].function.name == "get_model_hamiltonian":
        data["model"] = json.loads(assistant_message.tool_calls[0].function.arguments)
    else:
        results = f"Error: function {assistant_message.tool_calls[0].function.name} does not exist"

    ham = dict_to_ham(data)
    print(ham)
    print(data["model"])

    return ham

if __name__ == '__main__':
    input_string = input("Tell us to generate a Hamiltonian: ") 
    ham = chatgpt_to_ham(input_string)
