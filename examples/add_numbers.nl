## A simple program to ask for two numbers and add them

show("--- Addition Calculator ---")

## Ask the user for input
num1_str = ask("Enter the first number: ")
num2_str = ask("Enter the second number: ")

## Convert the text inputs into numbers (floats)
num1 = float(num1_str)
num2 = float(num2_str)

## Calculate the sum
result = num1 + num2

## Show the result using string interpolation
show("The sum of {num1} and {num2} is: {result}")
