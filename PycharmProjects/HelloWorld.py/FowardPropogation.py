import numpy as np
input_data = np.array([2,3])

weights = { 'node_0': np.array([2,4]),
            'node_1': np.array([-5,4]),
            'output': np.array([2,7])
            }
node_0_value = (input_data * weights['node_0']).sum()
node_1_value = (input_data * weights['node_1']).sum()


print(node_0_value)
print(node_1_value)
hidden_layer_values = np.array([node_0_value,node_1_value])
print(hidden_layer_values)

output_layer_values = (hidden_layer_values*weights['output']).sum()
print(output_layer_values)




