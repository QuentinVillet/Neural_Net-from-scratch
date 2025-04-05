import math

softmax_output = [0.7, 0.1, 0.2] #example of an output of the softmax and calculate a loss on that
target_output = [1, 0, 0]

loss = -(math.log(softmax_output[0])*target_output[0] +
            math.log(softmax_output[1])*target_output[1] +
            math.log(softmax_output[2])*target_output[2]
            )

# same as doing loss = -math.log(0.7)*1
# when confidence is higher, loss is lower and vice versa. loss belongs (-inf,0]
# confidence element of [0,1]
