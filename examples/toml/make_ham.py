import moha
from moha.toml_tools import from_toml

ham = from_toml("hubbard.toml")
zero_body = ham.generate_zero_body_integral()
one_body = ham.generate_one_body_integral(basis='spatial basis', dense=True)
two_body = ham.generate_two_body_integral(basis='spatial basis', dense=True)

print(zero_body, one_body.shape, two_body.shape)