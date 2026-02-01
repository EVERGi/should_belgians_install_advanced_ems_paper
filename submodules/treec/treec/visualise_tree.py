from treec.tree import BinaryTree
from treec.utils import denormalise_input

from graphviz import Source
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from collections import Counter


def binarytree_to_dot(binarytree, title="", leafs_visited=None):
    if leafs_visited is not None:
        leafs_count = Counter(leafs_visited)
    dot_format = "digraph BST {\n"
    dot_format += '    node [fontname="Arial" style=filled colorscheme=paired12];\n'

    node_description = ""
    parent_links = ""
    for i, node in enumerate(binarytree.node_stack):
        value = node.value
        round_value = "{:.2f}".format(round(value, 2))
        feature_name = node.feature
        if not node.is_leaf():
            feature_value = round_value

            color = hash(feature_name) % 12 + 1
            label = feature_name + ":\n" + str(feature_value)
            node_description += (
                "    "
                + str(i)
                + ' [ label = "'
                + label
                + '" fillcolor='
                + str(color)
                + "];\n"
            )
        else:

            action_name = feature_name
            action_value = round_value
            base_label = action_name + str(action_value)

            if leafs_visited is None:
                label = base_label
            else:
                cur_index = binarytree.node_stack.index(node)

                label = base_label + "\nVisit count: " + str(leafs_count[cur_index])
            node_description += (
                "    " + str(i) + ' [ label = "' + label + '" fillcolor=white];\n'
            )
        left_node = node.left_node
        right_node = node.right_node
        if left_node is not None:
            left_index = binarytree.node_stack.index(left_node)
            right_index = binarytree.node_stack.index(right_node)
            parent_links += (
                "    " + str(i) + "  -> " + str(left_index) + '[ label = "<"];\n'
            )
            parent_links += (
                "    " + str(i) + "  -> " + str(right_index) + '[ label = ">="];\n'
            )

    dot_format += node_description + "\n" + parent_links + "\n"
    dot_format += '    labelloc="t";\n'
    dot_format += '    label="' + title + '";\n'
    dot_format += "}"
    return dot_format


def display_dot_matplotlib(dot_string):
    plt.figure()
    temp_file = "temp"
    s = Source(dot_string, filename=temp_file, format="png")
    s.render()
    img = mpimg.imread(temp_file + ".png")
    plt.imshow(img)

    os.remove(temp_file)
    os.remove(temp_file + ".png")


def display_binarytree(binarytree, title="", leafs_visited=None):
    bt_dot = binarytree_to_dot(binarytree, title, leafs_visited)
    display_dot_matplotlib(bt_dot)


if __name__ == "__main__":
    example_value_dict = {
        "feature": "irradiation",
        "value": 0.5,
        "left_node": {
            "feature": "temperature",
            "value": 0.3,
        },
        "right_node": {
            "feature": "temperature",
            "value": 0.7,
            "left_node": {
                "feature": "discrete action",
                "value": 0,
            },
            "right_node": {
                "feature": "discrete action",
                "value": 1,
            },
        },
    }
    example_tree = BinaryTree(example_value_dict)
    a = BinaryTree(example_tree)

    a_dot = binarytree_to_dot(a)
    print(a_dot)

    display_dot_matplotlib(a_dot)
    plt.show()
