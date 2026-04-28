from moha.hamiltonians import AlternativeSpinHamiltonian

def main():
    print("Testing AlternativeSpinHamiltonian integration...")
    H = AlternativeSpinHamiltonian(num_spins=3, coupling_constants=1.0)

    print("\nOne-Body Integrals:")
    one_body = H.generate_one_body_integral(dense=True)
    print(one_body)

    print("\nTwo-Body Integrals (Shape):")
    two_body = H.generate_two_body_integral(dense=True)
    print(two_body.shape)

if __name__ == "__main__":
    main()