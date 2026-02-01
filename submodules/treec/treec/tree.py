import ast
from typing import List, Union
import abc
from treec.utils import normalise_input, denormalise_input


class NodeBinary:
    def __init__(self, feature: str, value: float, depth: int):
        self.feature = feature
        self.value = value
        self.depth = depth

        self.left_node = None
        self.right_node = None

    def is_leaf(self):
        empty_left = self.left_node is None
        empty_right = self.right_node is None
        return empty_left and empty_right


class BinaryTree:
    def __init__(
        self,
        value_dict: dict,
    ):
        self.node_stack = list()
        self.root_node = self.populate_node_stack(value_dict)

    def populate_node_stack(self, value_dict: dict, depth=0):
        feature = value_dict["feature"]
        value = value_dict["value"]
        node = NodeBinary(feature, value, depth)

        self.node_stack.append(node)
        if "left_node" in value_dict.keys():
            node.left_node = self.populate_node_stack(
                value_dict["left_node"], depth + 1
            )
        if "right_node" in value_dict.keys():
            node.right_node = self.populate_node_stack(
                value_dict["right_node"], depth + 1
            )
        return node

    def get_action(self, feature_values: dict):
        cur_node = self.root_node

        while not cur_node.is_leaf():
            val = feature_values[cur_node.feature]
            if val < cur_node.value:
                cur_node = cur_node.left_node
            else:
                cur_node = cur_node.right_node
        return cur_node

    def get_parent(self, node):
        for cur_node in self.node_stack:
            if cur_node.left_node == node or cur_node.right_node == node:
                return cur_node
        return None

    def __str__(self):
        tree_struct = ""
        for node in self.node_stack:
            if node.feature == "discrete action":
                feature = ""
            else:
                feature = str(node.feature) + ":"
            tree_struct += node.depth * " " + feature + str(node.value) + "\n"

        return tree_struct

    def recalculate_node_stack(self):
        new_stack = list()
        sort_stack(new_stack, self.root_node)
        self.node_stack = new_stack

    def recalculate_depth(self):
        new_stack = list()
        sort_stack(new_stack, self.root_node)
        self.node_stack = new_stack

    def get_value_dict(self):
        value_dict = self.recur_value_dict(self.root_node)
        return value_dict

    def recur_value_dict(self, node):
        node_dict = dict()
        node_dict["feature"] = node.feature
        node_dict["value"] = node.value

        if node.left_node is not None:
            node_dict["left_node"] = self.recur_value_dict(node.left_node)
        if node.right_node is not None:
            node_dict["right_node"] = self.recur_value_dict(node.right_node)

        return node_dict

    def save_tree(self, filepath):
        value_dict = self.get_value_dict()

        file = open(filepath, "w+")
        file.write(str(value_dict))
        file.close()


def construc_tree_from_file(filepath):
    file = open(filepath, "r+")
    content = file.read()
    file.close()

    value_dict = ast.literal_eval(content)

    return BinaryTree(value_dict)


def norm_array_to_complete_tree(
    norm_array: List[float], feature_bounds: dict, action_info: dict
):
    if (len(norm_array) - 1) % 3 != 0:
        message = """The list should have a length of 3n+1 with n a positive integer 
        where the first n elements are the features of the tree, 
        the next n are the split values 
        and the next n+1 the values of the leafs"""
        raise (ValueError(message))

    # Seperate already in feature, feature value and leaf and then write in free format
    num_splits = int((len(norm_array) - 1) / 3)

    name_features = list(feature_bounds.keys())
    name_actions = list(action_info.keys())

    num_features = len(name_features)
    num_actions = len(name_actions)

    norm_feature = norm_array[:num_splits]
    norm_split = norm_array[num_splits : 2 * num_splits]
    norm_leaf = norm_array[2 * num_splits :]

    index_features = [int(i * (num_features)) for i in norm_feature]
    value_features = [name_features[i] for i in index_features]
    value_splits = []
    for i, name_feature in enumerate(value_features):
        low_bound = feature_bounds[name_feature][0]
        up_bound = feature_bounds[name_feature][1]
        norm_value = norm_split[i]
        value = denormalise_input(norm_value, low_bound, up_bound)
        value_splits.append(value)

    feature_leafs = []
    value_leafs = []

    for i, norm_val_leaf in enumerate(norm_leaf):
        index_action = int(norm_val_leaf * num_actions)
        action_name = name_actions[index_action]
        cur_action = action_info[action_name]
        if "bounds" not in cur_action or cur_action["bounds"] is None:
            feature_leaf = "discrete action"
            value_leaf = name_actions[index_action]
        else:
            low_bound = cur_action["bounds"][0]
            up_bound = cur_action["bounds"][1]
            norm_section = norm_val_leaf * num_actions - index_action
            feature_leaf = name_actions[index_action]
            value_leaf = denormalise_input(norm_section, low_bound, up_bound)

        feature_leafs.append(feature_leaf)
        value_leafs.append(value_leaf)

    all_features = value_features + feature_leafs
    all_values = value_splits + value_leafs

    uncompleted_stack = list()
    for i, feature in enumerate(all_features):
        node = dict()
        node["feature"] = feature
        node["value"] = all_values[i]

        if i == 0:
            value_dict = node
        elif "left_node" not in node_to_complete.keys():
            node_to_complete["left_node"] = node
        elif "right_node" not in node_to_complete.keys():
            node_to_complete["right_node"] = node
            uncompleted_stack.remove(node_to_complete)

        uncompleted_stack.append(node)
        node_to_complete = uncompleted_stack[0]

    tree = BinaryTree(value_dict)
    return tree


def sort_stack(new_stack, cur_node, depth=0):
    cur_node.depth = depth
    new_stack.append(cur_node)
    if cur_node.is_leaf():
        return
    else:
        sort_stack(new_stack, cur_node.left_node, depth + 1)
        sort_stack(new_stack, cur_node.right_node, depth + 1)


if __name__ == "__main__":

    example = {
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
    example_tree = BinaryTree(example)
    print(example_tree)
    """
    a = BinaryTreeFixed(
        [0.4, 0.8, 0.5, 0.4, 0, 0.4, 0.8],
        ["Feat 0", "Feat 1", "Feat 2"],
        ["Action 0", "Action 1", "Action 2"],
    )
    """

    """
    b = BinaryTreeFree(
        [0.6, 0.8, 0.1, 0.1, 0.1, 0.5, 0.4, 0.4, 0.8, 0.1],
        ["Feat 0", "Feat 1", "Feat 2"],
        ["Action 0", "Action 1", "Action 2"],
    )
    """
    # a = BinaryTreeFixedCont([0.4,0.8,0.5,0.4,0.1,0.4,0.8], ["Feat 0", "Feat 1", "Feat 2"])
    # b = BinaryTreeFreeCont([0.6,0.8,0.1,0.1,0.1,0.5,0.4,0.4,0.8,0.1], ["Feat 0", "Feat 1", "Feat 2"])

    # print(a)
    # print(b)

    # print(a.get_action([0.2, 0.6, 0.6]))
    # print(b.get_action([0.2, 0.3, 0.3]))
