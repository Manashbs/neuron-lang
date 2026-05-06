## OOP Demo in Neuron

show("Neuron OOP Demo")
show("=" * 40)

## --- Entity (Class) ---
entity Animal:
    do create(me, name, sound):
        me.name = name
        me.sound = sound

    do speak(me):
        show("{me.name} says {me.sound}!")

    do describe(me):
        show("I am {me.name}")

## --- Inheritance ---
entity Dog from Animal:
    do create(me, name):
        me.name = name
        me.sound = "Woof"
        me.tricks = []

    do learn(me, trick):
        me.tricks.append(trick)
        show("{me.name} learned {trick}!")

    do perform(me):
        when me.tricks.length > 0:
            show("{me.name}'s tricks:")
            each trick within me.tricks:
                show("  > {trick}")
        otherwise:
            show("{me.name} doesn't know any tricks yet")

## Create instances
dog = Dog("Buddy")
dog.speak()
dog.learn("sit")
dog.learn("roll over")
dog.learn("shake")
dog.perform()

show("")

## --- Brain (AI Model) ---
entity Layer:
    do create(me, name):
        me.name = name
    do forward(me, x):
        give x

brain SimpleModel:
    do create(me, name):
        me.name = name
        me.layers = []
        me.trained = no

    do add_layer(me, layer):
        me.layers.append(layer)
        show("  Added layer: {layer.name}")

    do forward(me, x):
        result = x
        each layer within me.layers:
            result = layer.forward(result)
        give result

    do train(me):
        me.trained = yes
        show("  Model '{me.name}' trained!")

    do status(me):
        when me.trained:
            show("  Model '{me.name}': TRAINED")
        otherwise:
            show("  Model '{me.name}': NOT TRAINED")

show("Building AI Model...")
model = SimpleModel("MyNet")
model.add_layer(Layer("Dense-128"))
model.add_layer(Layer("ReLU"))
model.add_layer(Layer("Dense-10"))

show("")
model.status()
model.train()
model.status()

## Pipeline demo
show("")
show("--- Pipeline Operator ---")

do double(x):
    give x * 2

do add_ten(x):
    give x + 10

result = 5 |> double |> add_ten |> double
show("5 |> double |> add_ten |> double = {result}")
