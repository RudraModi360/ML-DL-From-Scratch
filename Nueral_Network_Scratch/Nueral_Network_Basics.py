import numpy as np

weigths = [
    [0.1, 0.2, 0.3],
    [0.4, 0.6, 0.8],
    [0.5, 0.7, 0.9],
]
inputs=[1,2,3]

print(np.dot(weigths,inputs)) ## Note always do W*X+b not X*W+b both are resulting diff things


