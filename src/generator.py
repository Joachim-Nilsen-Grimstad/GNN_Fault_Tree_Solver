'''
title      :   generator.py
version    :   0.0.1
date       :   08.10.2024 15:05:39
fileName   :   generator.py
author     :   Joachim Nilsen Grimstad
contact    :   Joachim.Grimstad@ias.uni-stuttgart.de

description:   Generates fault trees as graphs, then saves them.

license    :   This tool is licensed under Creative commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
                For license details, see https://creativecommons.org/licenses/by-nc-sa/4.0/  

disclaimer :   Author takes no responsibility for any use.
'''

# Imports
import random
import networkx as nx
import matplotlib.pyplot as plt
from datetime import date
import csv
from networkx.readwrite import json_graph
import json
import pydot
from networkx.drawing.nx_pydot import graphviz_layout


class FaultTreeDatasetGenerator():
    '''Fault Tree Generator Object'''    
    def __init__(self, max_num_children, min_children,  max_depth, num_graphs, path, visualization, print_out):
        # Instantiate Generator
        self.max_num_children = max_num_children
        self.max_depth = max_depth
        self.min_children = min_children
        self.num_graphs = num_graphs
        self.path = path
        self.visualization = visualization
        self.print_out = print_out

        for _ in range(0, self.num_graphs):
            self.reset()
            self.generate_subgraph()

    # Methods   
    def reset(self):        
        self.graph = nx.DiGraph()  # Instantiate an empty directed graph.  
        data = self.read_data_config()  
        self.node_counter = data['num_nodes'] # Instantiate a node counter for the graph.
        self.model_counter = data['num_models'] # keeps track of the number of generated models.

    def generate_subgraph(self):
        '''Generates a graph'''
        top_event_node_id = self.create_node(0) # Generate top event node
        self.create_tree(top_event_node_id, 0) # recursive generation from top event node (depth for this node is 0)
        self.add_meta_data()
        self.calculate_node_features()
        self.save_model()

    def create_node(self, node_type, gate_type=None, node_probability=-1):
        '''Creates a node; node_type = 0, 1, 2 for Top event, Intermediate node, and leaf node respectively,
        gate_type = 0, 1, 2 for dummy value, and-gate, or-gate; Leaf_probability -1 is a dummy value for 
        intermediate and top events.'''
        
        # Logic for selecting gate type.
        if gate_type is None:  # If default None gate type.
            if node_type == 2:  # If node_type is leaf, then select dummy gate_type.
                gate_type = 0  # Assign dummy gate type for leaf node
            else:  # if not a leaf node, then we need to pick gate.
                gate_type = random.choice([1, 2])  # Randomly choose gate type

        # Logic for assigning leaf probabilities
        if node_type == 2:
            node_probability = self.biased_random_low_number()  # Assign a random probability between 0 and 1
        node_id = self.node_counter + 1      
        node = self.graph.add_node(node_id, node_type=node_type, gate_type=gate_type, node_probability=node_probability)
        self.node_counter += 1  # Update node counter
        return node_id
    
    def create_edge(self, child_node_id, parent_node_id):
        '''Creates a directed edge between a child and parent node.'''
        self.graph.add_edge(child_node_id, parent_node_id)

    def create_tree(self, parent_node_id, current_depth):
        '''Recursively generates nodes, where each node has an independent chance to be intermediate or leaf.'''
        
        # Stop recursion at max depth
        if current_depth >= self.max_depth:
            return

        # Determine number of child nodes
        num_children = random.randint(self.min_children, self.max_num_children)
        
        # Generate each child independently
        for _ in range(num_children):
            # Set probability for being an intermediate node to zero if at max depth
            if current_depth == self.max_depth - 1:  # max_depth - 1 because depth starts from 0
                is_intermediate = False
            else:
                is_intermediate = random.random() < (1 - current_depth / self.max_depth)

            node_type = 1 if is_intermediate else 2  # Intermediate or Leaf
            
            # Create child node
            child_node_id = self.create_node(node_type, current_depth + 1)
            self.create_edge(child_node_id, parent_node_id)
            
            # recursive only if this node is intermediate
            if node_type == 1:
                self.create_tree(child_node_id, current_depth + 1)

    def print_nodes(self):
        '''Prints all nodes in the graph along with their attributes.'''
        for node in self.graph.nodes(data=True):
            print(f'Node ID: {node[0]}, Attributes: {node[1]}')

    def add_meta_data(self):
        self.graph.graph['model_id'] = self.model_counter + 1
        self.graph.graph['creation_date'] = date.today().isoformat()
        self.graph.graph['num_nodes'] = self.node_counter
        node_counts, gate_counts = self.count_nodes_and_gates()
        self.graph.graph['num_top_nodes'] = node_counts[0]
        self.graph.graph['num_intermediate_nodes'] = node_counts[1]
        self.graph.graph['num_leaf_nodes'] = node_counts[2]
        self.graph.graph['num_and_gates'] = gate_counts[1]
        self.graph.graph['num_or_gates'] = gate_counts[2]

    def count_nodes_and_gates(self):
        '''Counts the number of each type of node and gate in the graph and returns dictionaries with the counts.'''
        node_counts = {0: 0, 1: 0, 2: 0}  # Initialize counts for each node type
        gate_counts = {0: 0, 1: 0, 2: 0}  # Initialize counts for each gate type

        for _, attributes in self.graph.nodes(data=True):
            node_type = attributes.get('node_type')
            gate_type = attributes.get('gate_type')

            if node_type in node_counts:
                node_counts[node_type] += 1
            if gate_type in gate_counts:
                gate_counts[gate_type] += 1

        return node_counts, gate_counts


    def read_data_config(self):
        '''Reads a file with key-value pairs and returns a dictionary.'''
        file_path = self.path + '\data_config.txt'
        
        try:
            with open(file_path, 'r') as file:
                content = file.read().strip()  # Read and strip whitespace

            # Split by ';' and then by '=' to create key-value pairs
            data = {}
            for pair in content.split(';'):
                try:
                    key, value = pair.split('=')
                    data[key.strip()] = int(value.strip())  # Convert value to integer
                except ValueError as ve:
                    print(f"Value error while processing pair '{pair}': {ve}")
                except Exception as e:
                    print(f"Unexpected error while processing pair '{pair}': {e}")

            return data

        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found. Creating a new file with default values.")
            # Create the file with default data
            with open(file_path, 'w') as file:
                file.write("num_models=0;num_nodes=0")
            return {'num_models': 0, 'num_nodes': 0}  # Return default values

    def write_data_config(self):
        '''Writes a dictionary to a file in key=value; format.'''
        file_path = self.path + '\data_config.txt'
        data = {}
        data['num_models'] = self.model_counter + 1
        data['num_nodes'] = self.node_counter

        with open(file_path, 'w') as file:
            # Create key=value pairs and join them with ';'
            content = ';'.join(f"{key}={value}" for key, value in data.items())
            file.write(content)

    def save_model(self):
        # Save the model
        self.write_data_config()
        model_number = self.model_counter + 1
        model_filename = f"{self.path}/model_{model_number}.json"  # Corrected filename formatting
        data = json_graph.node_link_data(self.graph)  # Use node_link_data for saving the graph

        # Write to a JSON file
        with open(model_filename, 'w') as json_file:
            json.dump(data, json_file, indent=4)  # Directly dump the data

        if self.visualization:
            self.visualize_graph(model_number)

        if self.print_out:
            self.print_nodes()


    def visualize_graph(self, model_number):
        '''Visualizes the fault tree graph using Matplotlib with Graphviz layout.'''
        
        color_map = {0: 'red', 1: 'orange', 2: 'green'}  # Node colors

        # Draw the nodes using Graphviz layout
        pos = graphviz_layout(self.graph, prog='dot')  # Use 'twopi' layout from Graphviz

        # Create new positions with inverted y-coordinates
        pos_flipped = {node: (x, -y) for node, (x, y) in pos.items()}  # Flip y-coordinates

        # Draw nodes with specific colors
        for node, attrs in self.graph.nodes(data=True):
            node_color = color_map[attrs['node_type']]
            nx.draw_networkx_nodes(self.graph, pos_flipped, nodelist=[node], node_color=node_color)

        # Draw edges and labels
        nx.draw_networkx_edges(self.graph, pos_flipped, arrowstyle='->', arrowsize=10, edge_color='grey')
        nx.draw_networkx_labels(self.graph, pos_flipped, font_size=8)

        plt.title('Fault Tree Visualization')
        plt.axis('off')  # Turn off the axis
        
        # Save the figure to a file
        filename = f"{self.path}/visualizations/model_{model_number}.png"
        plt.savefig(filename, format='png', bbox_inches='tight')  # Save as PNG file
        plt.close()  

    def biased_random_low_number(self):
        '''Needs to be replaced with something that samples from a distribution that is biased towards lower probabilities.'''
        number = random.uniform(0, 1)
        power = 10
        return number**power

### Node Features

    def calculate_node_features(self):
        self.calculate_in_and_out_degrees()

    def calculate_in_and_out_degrees(self):
        for node in self.graph.nodes():
            in_degree = self.graph.in_degree(node)
            out_degree = self.graph.out_degree(node)
            self.graph.nodes[node]['in_degree'] = in_degree
            self.graph.nodes[node]['out_degree'] = out_degree