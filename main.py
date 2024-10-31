from src.generator import *

 
# Generation Params
max_num_children = 3
max_depth = 4
min_children = 2
num_graphs = 1

# Debugging
visualization = True
print_out = False  

# Dataset data.txt file - 100% better way to do this.
path = 'models\disjoint_graphs'

Gen = FaultTreeDatasetGenerator(max_num_children, min_children, max_depth, num_graphs, path, visualization, print_out)


