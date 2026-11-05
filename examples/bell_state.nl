## Compile this source to OpenQASM 3 with:
## neuron compile examples/bell_state.nl --target qasm3
quantum BellState:
    qubit alice
    qubit bob
    h(alice)
    cx(alice, bob)
    measure(alice)
    measure(bob)
