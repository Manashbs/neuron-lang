## Test Python interop -- grab real Python libraries!

grab numpy
grab os

## Use numpy directly
arr = numpy.array([1, 2, 3, 4, 5])
show(arr)
show("Sum: {numpy.sum(arr)}")
show("Mean: {numpy.mean(arr)}")

## Create a matrix
mat = numpy.zeros([3, 3])
show("Zero matrix:")
show(mat)

## Random numbers
rand = numpy.random.rand(5)
show("Random: {rand}")

## OS info
show("Current dir: {os.getcwd()}")
show("Python interop works!")
