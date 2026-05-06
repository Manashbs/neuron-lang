## Hello World in Neuron!
show("🧠 Welcome to Neuron!")
show("The AI-First Programming Language")
show("")

## Variables
name = ask("What is your name? ")
show("Hello, {name}! Let's build some AI together!")

## Quick math
x = 42
y = 8
show("The answer to everything: {x}")
show("{x} + {y} = {x + y}")

## A simple function
do greet(person):
    show("Hey {person}, welcome to the future of AI programming! 🚀")

greet(name)

## Using Neuron's unique keywords
do fizzbuzz(n):
    each i within upto(1, n + 1):
        when i % 15 == 0:
            show("FizzBuzz")
        orwhen i % 3 == 0:
            show("Fizz")
        orwhen i % 5 == 0:
            show("Buzz")
        otherwise:
            show(i)

show("\nFizzBuzz 1-15:")
fizzbuzz(15)
