"""GPT-3 based Hamiltonian generator."""

import json
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
import toml
from moha.toml_tools.tools import dict_to_ham
import json
from pathlib import Path


def map_to_toml(funcs):
    """Functin that maps the dictionary to the toml-like dictionary.

    Parmaeters:
    -----------
    funcs: dict
        The dictionary that contains the function arguments

    Returns
    -------
    toml_dict: dict
        The dictionary that contains the toml-like arguments

    """
    toml_dict = {}
    toml_dict["control"] = {}
    toml_dict["system"] = {}
    toml_dict["model"] = {}

    # check if values under norb and nelec are non-zero
    if not funcs.get("norb", None):
        funcs["norb"] = funcs["nelec"]
    if not funcs.get("nelec", None):
        funcs["nelec"] = funcs["norb"]

    # loop over the dictionary and map it to the toml file
    for key, value in funcs.items():
        if key in ["save_integrals",
                   "integral_format",
                   "outdir",
                   "prefix"]:
            toml_dict["control"][key] = value
            if key == "save_integrals":
                toml_dict["control"][key] = True if value == "true" else False

        elif key in ["bc",
                     "norb",
                     "nelec",
                     "symmetry",
                     "moltype",
                     "Lx", "Ly"]:
            if key == 'symmetry':
                value = int(value)
            toml_dict["system"][key] = value
        else:
            toml_dict["model"][key] = value

    # specify prefix if not specified
    toml_dict['control']['prefix'] = toml_dict['control'].get('prefix',
                                                              toml_dict['model']['hamiltonian']  # noqa E128
    )
    return toml_dict


def load_config():
    """Load OpenAI API key and model type from config.txt file.

    Returns
    -------
    OPENAI_API_KEY : str
        OpenAI API key
    GPT_MODEL : str
        GPT model type

    """
    config_path = Path(__file__).parent / "config.toml"
    config = toml.load(config_path)
    OPENAI_API_KEY = config["key"]
    GPT_MODEL = config["model"]

    return OPENAI_API_KEY, GPT_MODEL


@retry(wait=wait_random_exponential(multiplier=1, max=100),
       stop=stop_after_attempt(3))
def chat_completion_request(messages, model, client,
                            tools=None,
                            tool_choice=None):
    """Request ChatCompletion from OpenAI API.

    Parameters
    ----------
    messages : list
        List of messages
    model : str
        GPT model type
    client : OpenAI
        OpenAI client
    tools : dict
        Tools dictionary
    tool_choice : dict
        Tool choice dictionary

    Returns
    -------
    response : dict
        ChatCompletion response

    """
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


def read_promt():
    """Read prompt from user.

    Returns
    -------
    prompt : str
        User input prompt

    """
    prompt = input("Describe Hamiltonian that needs to be generated: ")
    return prompt


def generate_ham(prompt):
    """Generate Hamiltonian from prompt.

    Parameters
    ----------
    prompt : str
        User input prompt

    Returns
    -------
    ham : dict
        Model Hamiltonian

    """
    # get api key and model type
    OPENAI_API_KEY, GPT_MODEL = load_config()
    # load tools from tools.json file
    tools_path = Path(__file__).parent / "tools.json"
    tools = json.load(open(tools_path, "rb"))
    # initialize model
    client = OpenAI(api_key=OPENAI_API_KEY)
    # initialize messages
    messages = []
    messages.append({"role": "system",
                    "content":
                    "You will be given a description of Hamiltonian.\
                     You need to map it to the correct keywords and describe the generated system."})  # noqa E128
    messages.append({"role": "system",
                    "content":
                    "Don't make assumptions about what values to plug into functions."})  # noqa E128
    tmp = {"role": "user"}

    # add prompt to messages
    tmp["content"] = prompt
    messages.append(tmp)

    # get chat completion response
    chat_response = chat_completion_request(model=GPT_MODEL, client=client,
                                            messages=messages, tools=tools)

    funcs = chat_response.choices[0].\
        message.to_dict()['tool_calls'][0]['function']['arguments']
    dct = json.loads(funcs)
    try:
        print(dct['explanation'])
    # if explanation is not present, raise KeyError
    except KeyError:
        funcs = chat_response.choices[0].\
            message.to_dict()['tool_calls'][0]['function']['arguments']
        dct = json.loads(funcs)
        try:
            print(dct['explanation'])
        except KeyError:
            raise ValueError("Unexpected error occurred. Please try again.")

    # convert chat response to Hamiltonian
    toml_dict = map_to_toml(dct)
    ham = dict_to_ham(toml_dict)
    return ham


if __name__ == "__main__":
    prompt = read_promt()
    ham = generate_ham(prompt)
    print(ham)
    print("Hamiltonian generated successfully")
