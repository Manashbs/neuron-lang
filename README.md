# Neuron Language 

> The AI-First Programming Language. Built for humans, designed for Machine Learning.

Neuron (`.nl`) is a high-level, Python-like, OOP-based programming language built from the ground up for Artificial Intelligence training, tensors, and neural networks. It features a clean, English-like syntax that replaces traditional programming keywords with intuitive verbs.

---

## Installation

You can install Neuron directly from PyPI on any Windows, Mac, or Linux machine:

```bash
pip install neuron-lang
```

---

## Running Code

Once installed, you can use the `neuron` command to run files, compile them, or start the interactive REPL.

```bash
# Start the interactive coding environment
neuron

# Run a Neuron script
neuron run my_script.nl

# Compile for faster future execution
neuron compile my_script.nl
```

---

## Package Manager (Auto-Install!)

Neuron has built-in integration with the entire Python ecosystem. Any Python library (NumPy, Pandas, TensorFlow, PyTorch) can be used in Neuron!

If you `grab` a library that isn't installed, **Neuron will automatically install it for you in the background!**

```neuron
## If pandas isn't installed, Neuron auto-installs it and imports it!
grab pandas

## You can also explicitly manage packages via the CLI:
# neuron get scikit-learn
# neuron packages
# neuron remove pandas
```

---

## Language Syntax & Features

Neuron uses natural language keywords to make coding more intuitive.

### Variables & Printing
```neuron
## Variables don't need type declarations
name = "AI Builder"
version = 1.0

## String interpolation uses {brackets}
show("Welcome {name} to Neuron v{version}!")

## Ask for user input
user = ask("What is your name? ")
```

### Functions
Use `do` to define a function, and `give` to return a value.

```neuron
do add_numbers(x, y):
    result = x + y
    give result

show(add_numbers(10, 5))  ## Output: 15
```

### Conditions
Neuron replaces `if/elif/else` with `when/orwhen/otherwise`.

```neuron
age = 20

when age < 18:
    show("Too young!")
orwhen age == 18:
    show("Just right!")
otherwise:
    show("Welcome!")
```

### Loops
Neuron features a powerful `each ... within` loop construct.

```neuron
## Loop through a list
each item within [1, 2, 3]:
    show(item)

## Loop through a range using upto()
each i within upto(5):
    show(i)  ## Outputs 0, 1, 2, 3, 4
```

### The Pipeline Operator `|>`
Easily chain function calls together reading left-to-right!

```neuron
do double(x):
    give x * 2

do add_ten(x):
    give x + 10

## Passes 5 into double(), then passes the result into add_ten()
result = 5 |> double |> add_ten
show(result)  ## Output: 20
```

---

## Object-Oriented Programming (OOP)

Neuron uses `entity` instead of `class`, and `me` instead of `self` or `this`.

```neuron
entity Animal:
    do create(me, name):
        me.name = name

    do speak(me):
        show("{me.name} makes a sound.")

## Inheritance using 'from'
entity Dog from Animal:
    do speak(me):
        show("{me.name} says Woof!")

buddy = Dog("Buddy")
buddy.speak()  ## Output: Buddy says Woof!
```

---

## AI-Native Features (`brain`)

Neuron has a special class type called `brain` specifically for AI models. Brains automatically have `train()`, `predict()`, and `save()` functionalities built-in.

```neuron
grab neural

brain MyNet:
    do create(me):
        me.add_layer(neural.Dense(128))
        me.add_layer(neural.ReLU())
        me.add_layer(neural.Dense(10))

model = MyNet()
show(model.status)  ## Output: NOT TRAINED
model.train(data)
```

---

## Tensors and Deep Learning

Neuron ships with a built-in standard library for tensors (`tensor`) with automatic differentiation (Autograd) for building neural networks from scratch.

### Compile-Time Shape Checks

The compiler and interpreter check statically known tensor shapes before running a program. Use a typed tensor declaration to declare an expected shape:

```neuron
grab tensor

tensor[100, 784] x = tensor.random(100, 784)
tensor[784, 10] w = tensor.random(784, 10)
tensor[100, 10] y = x.matmul(w)
```

The current analysis infers shapes from `tensor.random`, `tensor.rand`, `tensor.zeros`, and `tensor.ones`, propagates them through assignments, and checks rank-2 `matmul` and elementwise broadcasting. It reports a compile error when a known result conflicts with a declaration or matrix dimensions do not align. Shapes that cannot be inferred statically are left for runtime validation.

### Linear Regression Example
```neuron
grab tensor

## Generate Data
x = tensor.random(100, 1)
y = x * 3.0 + 2.0

## Weights
w = tensor.random(1, 1)
b = tensor.zeros(1, 1)
lr = 0.01

## Training Loop
each epoch within upto(100):
    ## Forward pass
    y_pred = x.matmul(w) + b
    
    ## MSE Loss
    diff = y_pred - y
    loss = (diff * diff).mean()
    
    ## Manual Gradients
    grad_w = x.transpose().matmul(diff) * 0.02
    grad_b = diff.mean() * 2.0
    
    ## Update
    w = w - grad_w * lr
    b = b - grad_b * lr

show("Learned w: {w}, b: {b}")
```

---

## Built-In Standard Libraries

Neuron comes batteries-included with AI primitives:
* `tensor` - Tensors and math operations
* `neural` - Neural network layers (Dense, Conv2d, ReLU, Softmax)
* `optim` - Optimizers (SGD, Adam)
* `data` - Data loading and batching
* `vision` / `nlp` - Image processing and text tokenization
* `metrics` - Accuracy, MSE, MAE
* `plot` - Easy data visualization

## Compiler and Runtime Status

Neuron currently uses a Python interpreter and a Python bytecode compiler. `.nlc` files are Python-pickled bytecode chunks; they are not yet a standalone executable format or a native VM. The shape-analysis pass is an initial compiler foundation, not a native tensor type system.

Native CPU/CUDA memory management and GIL-free execution, a linear quantum type checker, hybrid parameter-shift training, OpenQASM 3 code generation, and IBM Quantum Runtime submission are not implemented yet. They require separate compiler/runtime modules and explicit circuit syntax and ownership rules; no claim of hardware support is made by this release.

---
*Created by the Neuron Developer Community*
