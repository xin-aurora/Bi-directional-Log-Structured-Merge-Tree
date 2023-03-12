# Red Black Tree implementation in Python 2.7
# Author: Algorithm Tutor
# Tutorial URL: https://algorithmtutor.com/Data-Structures/Tree/Red-Black-Trees/
import random
import sys


# data structure that represents a node in the tree
from queue import Queue


class Node():
    def __init__(self, data):
        self.data = data  # holds the key
        self.parent = None  # pointer to the parent
        self.left = None  # pointer to left child
        self.right = None  # pointer to right child
        self.color = 1  # 1 . Red, 0 . Black

class IntervalNode(Node):
    def __init__(self, min, max):
        super().__init__(min)
        self.node_min = min
        self.node_max = max
        self.subtree_min = min
        self.subtree_max = max
        self.overlap_count = 0
        self.update_count = 0
        self.create_count = None
        self.create = False
        self.frequency = 1
        self.save_frequency = 0
        self.target_frequency = -1
        self.q_write = 0

    def update_subtree_min(self):
        if self.left is not None:
            if self.subtree_min > self.left.subtree_min:
                self.subtree_min = self.left.subtree_min

    def update_subtree_max(self):
        if self.right is not None:
            if self.subtree_max < self.right.subtree_max:
                self.subtree_max = self.right.subtree_max
        if self.left is not None:
            if self.subtree_max < self.left.subtree_max:
                self.subtree_max = self.left.subtree_max

# class RedBlackTree implements the operations in Red Black Tree
class RedBlackTree():
    def __init__(self):
        self.TNULL = Node(0)
        self.TNULL.color = 0
        self.TNULL.left = None
        self.TNULL.right = None
        self.root = self.TNULL

    def __pre_order_helper(self, node):
        if node != self.TNULL:
            sys.stdout.write(node.data + " ")
            self.__pre_order_helper(node.left)
            self.__pre_order_helper(node.right)

    def __in_order_helper(self, node):
        if node != self.TNULL:
            self.__in_order_helper(node.left)
            sys.stdout.write(node.data + " ")
            self.__in_order_helper(node.right)

    def __post_order_helper(self, node):
        if node != self.TNULL:
            self.__post_order_helper(node.left)
            self.__post_order_helper(node.right)
            sys.stdout.write(node.data + " ")

    def __search_tree_helper(self, node, key):
        if node == self.TNULL or key == node.data:
            return node

        if key < node.data:
            return self.__search_tree_helper(node.left, key)
        return self.__search_tree_helper(node.right, key)

    # fix the rb tree modified by the delete operation
    def __fix_delete(self, x):
        while x != self.root and x.color == 0:
            if x == x.parent.left:
                s = x.parent.right
                if s.color == 1:
                    # case 3.1
                    s.color = 0
                    x.parent.color = 1
                    self.left_rotate(x.parent)
                    s = x.parent.right

                if s.left.color == 0 and s.right.color == 0:
                    # case 3.2
                    s.color = 1
                    x = x.parent
                else:
                    if s.right.color == 0:
                        # case 3.3
                        s.left.color = 0
                        s.color = 1
                        self.right_rotate(s)
                        s = x.parent.right

                    # case 3.4
                    s.color = x.parent.color
                    x.parent.color = 0
                    s.right.color = 0
                    self.left_rotate(x.parent)
                    x = self.root
            else:
                s = x.parent.left
                if s.color == 1:
                    # case 3.1
                    s.color = 0
                    x.parent.color = 1
                    self.right_rotate(x.parent)
                    s = x.parent.left

                if s.left.color == 0 and s.right.color == 0:
                    # case 3.2
                    s.color = 1
                    x = x.parent
                else:
                    if s.left.color == 0:
                        # case 3.3
                        s.right.color = 0
                        s.color = 1
                        self.left_rotate(s)
                        s = x.parent.left

                        # case 3.4
                    s.color = x.parent.color
                    x.parent.color = 0
                    s.left.color = 0
                    self.right_rotate(x.parent)
                    x = self.root
        x.color = 0

    def __rb_transplant(self, u, v):
        if u.parent == None:
            self.root = v
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent

    def __delete_node_helper(self, node, key):
        # find the node containing key
        z = self.TNULL
        while node != self.TNULL:
            if node.data == key:
                z = node

            if node.data <= key:
                node = node.right
            else:
                node = node.left

        if z == self.TNULL:
            print
            "Couldn't find key in the tree"
            return

        y = z
        y_original_color = y.color
        if z.left == self.TNULL:
            x = z.right
            self.__rb_transplant(z, z.right)
        elif (z.right == self.TNULL):
            x = z.left
            self.__rb_transplant(z, z.left)
        else:
            y = self.minimum(z.right)
            y_original_color = y.color
            x = y.right
            if y.parent == z:
                x.parent = y
            else:
                self.__rb_transplant(y, y.right)
                y.right = z.right
                y.right.parent = y

            self.__rb_transplant(z, y)
            y.left = z.left
            y.left.parent = y
            y.color = z.color
        if y_original_color == 0:
            self.__fix_delete(x)

    # fix the red-black tree
    def __fix_insert(self, k):
        while k.parent.color == 1:
            if k.parent == k.parent.parent.right:
                u = k.parent.parent.left  # uncle
                if u.color == 1:
                    # case 3.1
                    u.color = 0
                    k.parent.color = 0
                    k.parent.parent.color = 1
                    k = k.parent.parent
                else:
                    if k == k.parent.left:
                        # case 3.2.2
                        k = k.parent
                        self.right_rotate(k)
                    # case 3.2.1
                    k.parent.color = 0
                    k.parent.parent.color = 1
                    self.left_rotate(k.parent.parent)
            else:
                u = k.parent.parent.right  # uncle

                if u.color == 1:
                    # mirror case 3.1
                    u.color = 0
                    k.parent.color = 0
                    k.parent.parent.color = 1
                    k = k.parent.parent
                else:
                    if k == k.parent.right:
                        # mirror case 3.2.2
                        k = k.parent
                        self.left_rotate(k)
                    # mirror case 3.2.1
                    k.parent.color = 0
                    k.parent.parent.color = 1
                    self.right_rotate(k.parent.parent)
            if k == self.root:
                break
        self.root.color = 0

    def __print_helper(self, node, indent, last):
        # print the tree structure on the screen
        if node != self.TNULL:
            sys.stdout.write(indent)
            if last:
                sys.stdout.write("R----")
                indent += "     "
            else:
                sys.stdout.write("L----")
                indent += "|    "

            s_color = "RED" if node.color == 1 else "BLACK"
            print(str(node.data) + "(" + s_color + ")")
            self.__print_helper(node.left, indent, False)
            self.__print_helper(node.right, indent, True)

    # Pre-Order traversal
    # Node.Left Subtree.Right Subtree
    def preorder(self):
        self.__pre_order_helper(self.root)

    # In-Order traversal
    # left Subtree . Node . Right Subtree
    def inorder(self):
        self.__in_order_helper(self.root)

    # Post-Order traversal
    # Left Subtree . Right Subtree . Node
    def postorder(self):
        self.__post_order_helper(self.root)

    # search the tree for the key k
    # and return the corresponding node
    def searchTree(self, k):
        return self.__search_tree_helper(self.root, k)

    # find the node with the minimum key
    def minimum(self, node):
        while node.left != self.TNULL:
            node = node.left
        return node

    # find the node with the maximum key
    def maximum(self, node):
        while node.right != self.TNULL:
            node = node.right
        return node

    # find the successor of a given node
    def successor(self, x):
        # if the right subtree is not None,
        # the successor is the leftmost node in the
        # right subtree
        if x.right != self.TNULL:
            return self.minimum(x.right)

        # else it is the lowest ancestor of x whose
        # left child is also an ancestor of x.
        y = x.parent
        while y != self.TNULL and x == y.right:
            x = y
            y = y.parent
        return y

    # find the predecessor of a given node
    def predecessor(self, x):
        # if the left subtree is not None,
        # the predecessor is the rightmost node in the
        # left subtree
        if (x.left != self.TNULL):
            return self.maximum(x.left)

        y = x.parent
        while y != self.TNULL and x == y.left:
            x = y
            y = y.parent

        return y

    # rotate left at node x
    def left_rotate(self, x):
        y = x.right
        x.right = y.left
        if y.left != self.TNULL:
            y.left.parent = x

        y.parent = x.parent
        if x.parent == None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    # rotate right at node x
    def right_rotate(self, x):
        y = x.left
        x.left = y.right
        if y.right != self.TNULL:
            y.right.parent = x

        y.parent = x.parent
        if x.parent == None:
            self.root = y
        elif x == x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y

    # insert the key to the tree in its appropriate position
    # and fix the tree
    def insert(self, key):
        # Ordinary Binary Search Insertion
        node = Node(key)
        node.parent = None
        node.data = key
        node.left = self.TNULL
        node.right = self.TNULL
        node.color = 1  # new node must be red

        y = None
        x = self.root

        while x != self.TNULL:
            y = x
            if node.data < x.data:
                x = x.left
            else:
                x = x.right

        # y is parent of x
        node.parent = y
        if y == None:
            self.root = node
        elif node.data < y.data:
            y.left = node
        else:
            y.right = node

        # if new node is a root node, simply return
        if node.parent == None:
            node.color = 0
            return

        # if the grandparent is None, simply return
        if node.parent.parent == None:
            return

        # Fix the tree
        self.__fix_insert(node)

    def get_root(self):
        return self.root

    # delete the node from the tree
    def delete_node(self, data):
        self.__delete_node_helper(self.root, data)

    # print the tree structure on the screen
    def pretty_print(self):
        self.__print_helper(self.root, "", True)



class IntervalStatisticTree(RedBlackTree):
    # def __init__(self, frozen_threshold: int, frequent_threshold: float, rebuild_threshold: int, similar_threshold: float):
    #     super().__init__()
    #     self.query_size = 0
    #     self.frozen_threshold = frozen_threshold
    #     self.frequent_threshold = frequent_threshold
    #     self.rebuild_threshold = rebuild_threshold
    #     self.similar_threshold = similar_threshold
    #     self.diff_threshold = 1.0 - similar_threshold
    def __init__(self, max_window_size: int, frozen_threshold: int, similar_threshold: float, do_rebuilt):
        super().__init__()
        self.max_window_size = max_window_size
        self.query_size = 0
        self.frozen_threshold = frozen_threshold
        self.similar_threshold = similar_threshold
        self.diff_threshold = 1.0 - similar_threshold
        self.do_rebuilt = do_rebuilt

    def create_new_node(self, q_min, q_max):
        new_node = IntervalNode(q_min, q_max)
        if self.root == self.TNULL:
            return new_node
        queue = Queue()
        queue.put(self.root)
        while queue.qsize() > 0:
            tree_node: IntervalNode = queue.get()
            # update overlap_count
            if q_min <= tree_node.node_min <= q_max or q_min <= tree_node.node_max <= q_max or \
                    tree_node.node_min <= q_min <= tree_node.node_max or tree_node.node_min <= q_max <= tree_node.node_max:
                tree_node.overlap_count += 1
                new_node.overlap_count += 1
            # if subtree range of the tree node overlap with query
            # check its children
            if q_min <= tree_node.subtree_min <= q_max or q_min <= tree_node.subtree_max <= q_max or \
                    tree_node.subtree_min <= q_min <= tree_node.subtree_max or tree_node.subtree_min <= q_max <= tree_node.subtree_max:
                if tree_node.left != self.TNULL:
                    queue.put(tree_node.left)
                if tree_node.right != self.TNULL:
                    queue.put(tree_node.right)
        return new_node

    def create_new_node_with_similar_t(self, q_min, q_max, q_length):
        is_new_range = True
        # q_length = q_max - q_min
        new_node = IntervalNode(q_min, q_max)
        similar_node = None
        if self.root == self.TNULL:
            return new_node, is_new_range
        queue = Queue()
        queue.put(self.root)
        while queue.qsize() > 0:
            tree_node: IntervalNode = queue.get()

            # update overlap_count
            if q_min <= tree_node.node_min <= q_max or q_min <= tree_node.node_max <= q_max or \
                    tree_node.node_min <= q_min <= tree_node.node_max or tree_node.node_min <= q_max <= tree_node.node_max:
                # tree_node.overlap_count += 1
                # new_node.overlap_count += 1
                if (is_new_range):
                    if tree_node.node_min <= q_min < q_max <= tree_node.node_max:
                        tree_node.frequency += 1
                        new_node.frequency += 1
                        similar_node = new_node
                        is_new_range = False
                        if (tree_node.frequency > new_node.frequency):
                            new_node.frequency = tree_node.frequency
                    else:
                        diff = (int(tree_node.node_min) - int(q_min)) + (int(tree_node.node_max) - int(q_max))
                        if (abs(diff)/q_length) < self.diff_threshold:
                            new_node.frequency += 1
                            tree_node.frequency +=1
                            is_new_range = False
                            if (tree_node.frequency > new_node.frequency):
                                new_node.frequency = tree_node.frequency
            # if subtree range of the tree node overlap with query
            # check its children
            if q_min <= tree_node.subtree_min <= q_max or q_min <= tree_node.subtree_max <= q_max or \
                    tree_node.subtree_min <= q_min <= tree_node.subtree_max or tree_node.subtree_min <= q_max <= tree_node.subtree_max:
                if tree_node.left != self.TNULL:
                    queue.put(tree_node.left)
                if tree_node.right != self.TNULL:
                    queue.put(tree_node.right)
        if similar_node is None:
            return new_node, is_new_range
        else:
            return similar_node, is_new_range

    def reset_overlap_count(self, node: IntervalNode):
        min = node.node_min
        max = node.node_max
        queue = Queue()
        queue.put(self.root)
        while queue.qsize() > 0:
            tree_node: IntervalNode = queue.get()
            # update overlap_count
            if min <= tree_node.node_min <= max or min <= tree_node.node_max <= max or \
                    tree_node.node_min <= min <= tree_node.node_max or tree_node.node_min <= max <= tree_node.node_max:
                tree_node.overlap_count -= 1
                node.overlap_count -= 1
            # if subtree range of the tree node overlap with query
            # check its children
            if min <= tree_node.subtree_min <= max or min <= tree_node.subtree_max <= max or \
                    tree_node.subtree_min <= min <= tree_node.subtree_max or tree_node.subtree_min <= max <= tree_node.subtree_max:
                if tree_node.left != self.TNULL:
                    queue.put(tree_node.left)
                if tree_node.right != self.TNULL:
                    queue.put(tree_node.right)

    def insert_new_key(self, new_key):
        queue = Queue()
        queue.put(self.root)
        while queue.qsize() > 0:
            tree_node: IntervalNode = queue.get()
            # if subtree range of the tree node overlap with new key
            # check its children
            if tree_node.subtree_min <= new_key <= tree_node.subtree_max:
                tree_node.update_count += 1
                if tree_node.left != self.TNULL:
                    queue.put(tree_node.left)
                if tree_node.right != self.TNULL:
                    queue.put(tree_node.right)

    # rotate left at node x
    def left_rotate(self, x: IntervalNode):
        y: IntervalNode = x.right
        x.right = y.left
        if y.left != self.TNULL:
            y.left.parent = x

        y.parent = x.parent
        if x.parent == None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y
        # update subtree range
        # update x first
        tem_max = x.node_max
        if x.left != self.TNULL and tem_max < x.left.subtree_max:
            tem_max = x.left.subtree_max
        if x.right != self.TNULL and tem_max < x.right.subtree_max:
            tem_max = x.right.subtree_max
        x.subtree_max = tem_max
        # update y
        if y.subtree_min > y.left.subtree_min:
            y.subtree_min = y.left.subtree_min
        tem_max = y.node_max
        if tem_max < y.left.subtree_max:
            tem_max = y.left.subtree_max
        if y.right != self.TNULL and tem_max < y.right.subtree_max:
            tem_max = y.right.subtree_max
        y.subtree_max = tem_max

    # rotate right at node x
    def right_rotate(self, x: IntervalNode):
        y: IntervalNode = x.left
        x.left = y.right
        if y.right != self.TNULL:
            y.right.parent = x

        y.parent = x.parent
        if x.parent == None:
            self.root = y
        elif x == x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y
        # update subtree range
        # update x first
        x.subtree_min = x.node_min
        tem_subtree = x.node_max
        if x.left != self.TNULL:
            if x.node_min > x.left.subtree_min:
                x.subtree_min = x.left.subtree_min
            if tem_subtree < x.left.subtree_max:
                tem_subtree = x.left.subtree_max
            if x.right != self.TNULL and tem_subtree < x.right.subtree_max:
                tem_subtree = x.right.subtree_max
        x.subtree_max = tem_subtree
        # y.subtree_min does not change
        tem_subtree = y.node_max
        if y.left != self.TNULL and tem_subtree < y.left.subtree_max:
            tem_subtree = y.left.subtree_max
        if tem_subtree < y.right.subtree_max:
            tem_subtree = y.right.subtree_max
        y.subtree_max = tem_subtree

    def insert(self, q_min, q_max, q_length, n, IO_unit, is_short_query) -> bool:
        # Ordinary Binary Search Insertion
        # node = self.create_new_node(q_min, q_max)
        node, is_new_range = self.create_new_node_with_similar_t(q_min, q_max, q_length)

        # print("[", q_min , "-", q_max, "] is new range = ", is_new_range)

        # increase tree size
        self.query_size += 1

        if is_new_range:
            node.parent = None
            node.data = q_min
            node.left = self.TNULL
            node.right = self.TNULL
            node.color = 1  # new node must be red

            y = None
            x: IntervalNode = self.root

            while x != self.TNULL:
                y = x
                if node.data < x.data:
                    if node.subtree_min < x.subtree_min:
                        x.subtree_min = node.subtree_min
                    if node.subtree_max > x.subtree_max:
                        x.subtree_max = node.subtree_max
                    x = x.left
                else:
                    if node.subtree_min < x.subtree_min:
                        x.subtree_min = node.subtree_min
                    if node.subtree_max > x.subtree_max:
                        x.subtree_max = node.subtree_max
                    x = x.right

            # y is parent of x
            node.parent = y
            if y == None:
                self.root = node
            elif node.data < y.data:
                y.left = node
            else:
                y.right = node

            # if new node is a root node, simply return
            if node.parent == None:
                node.color = 0
                return False
                # print("frozen")
                # return -1

            # if the grandparent is None, simply return
            if node.parent.parent == None:
                return False
                # print("frozen")
                # return -1

            # Fix the tree
            self.__fix_insert(node)

        if self.do_rebuilt:
            if self.query_size > self.frozen_threshold:
                # return self.estimate_frequency(node)
                estimate_frequency = self.estimate_frequency(node)
                # read only
                # if not node.create:
                #     return self.is_frequent(node, q_length, estimate_frequency, n, IO_unit, is_short_query)
                # else:
                #     node.save_frequency += 1
                #     return False
                # # do rebuilt
                if not node.create:
                    return self.is_frequent_rebuilt(node, q_length, estimate_frequency, n, IO_unit, is_short_query)
                else:
                    node.save_frequency += 1
                    return self.is_frequent_rebuilt(node, q_length, estimate_frequency, n, IO_unit, is_short_query)
            else:
                # print("frozen")
                return False
        else:
            # if the combination of overlap_count and update_count > threshold
            # create sp
            if self.query_size > self.frozen_threshold:
                # return self.estimate_frequency(node)
                estimate_frequency = self.estimate_frequency(node)
                # read only
                if not node.create:
                    return self.is_frequent(node, q_length, estimate_frequency, n, IO_unit, is_short_query)
                else:
                    node.save_frequency += 1
                    return False
            else:
                # print("frozen")
                return False
            # if self.query_size >= self.frozen_threshold:
            #     if (node.overlap_count - node.update_count) > self.query_size * self.frequent_threshold:
            #         if node.create_count is None:
            #             node.create_count = node.update_count
            #             self.reset_overlap_count(node)
            #             return True
            #         elif (node.update_count - node.create_count) > self.rebuild_threshold:
            #             node.create_count = node.update_count
            #             self.reset_overlap_count(node)
            #             return True
            # else:
            #     # print("frozen")
            #     return False

    def is_frequently_simpler(self, node: IntervalNode):
        if node.frequency > 1:
            return True
        else:
            return False

    def estimate_frequency(self, node: IntervalNode):
        estimate_frequency = node.frequency / self.query_size * (self.max_window_size - self.query_size)
        # print(node.frequency)
        # print(self.query_size)
        # print(node.frequency / self.query_size)
        # print(self.max_window_size - self.query_size)
        # print("estimate_frequency = ", estimate_frequency)
        return estimate_frequency

    def is_frequent(self, node: IntervalNode, q_length, estimate_frequency, n, IO_unit, is_short_query):

        write_cost = q_length / IO_unit
        # TODO: change to num of results
        if is_short_query:
            # short range query
            estimate_save_read_cost = estimate_frequency * (n+1-1/IO_unit)
        else:
            estimate_save_read_cost = estimate_frequency * (n+1-(n+1)/IO_unit)

        if estimate_save_read_cost > write_cost:
            node.create = True
            return True
        else:
            return False

    def is_frequent_rebuilt(self, node: IntervalNode, q_length, estimate_frequency, n, IO_unit, is_short_query):

        # if node.update_count / q_length < 0.1:
        #     return False

        if node.save_frequency < node.target_frequency:
            return False
        else:
            write_cost = node.q_write / IO_unit + (q_length + node.update_count) / IO_unit

            if is_short_query:
                # short range query
                estimate_save_read_cost = (estimate_frequency + node.save_frequency) * (n + 1 - 1 / IO_unit)
            else:
                estimate_save_read_cost = (estimate_frequency + node.save_frequency) * (n + 1 - (n + 1) / IO_unit)

            if estimate_save_read_cost > write_cost:
                node.create = True
                node.q_write += q_length + node.update_count
                node.target_frequency = estimate_save_read_cost
                node.save_frequency = 0
                return True
            else:
                return False

    def __fix_insert(self, k):
        while k.parent.color == 1:
            if k.parent == k.parent.parent.right:
                u = k.parent.parent.left  # uncle
                if u.color == 1:
                    # case 3.1
                    u.color = 0
                    k.parent.color = 0
                    k.parent.parent.color = 1
                    k = k.parent.parent
                else:
                    if k == k.parent.left:
                        # case 3.2.2
                        k = k.parent
                        self.right_rotate(k)
                    # case 3.2.1
                    k.parent.color = 0
                    k.parent.parent.color = 1
                    self.left_rotate(k.parent.parent)
            else:
                u = k.parent.parent.right  # uncle

                if u.color == 1:
                    # mirror case 3.1
                    u.color = 0
                    k.parent.color = 0
                    k.parent.parent.color = 1
                    k = k.parent.parent
                else:
                    if k == k.parent.right:
                        # mirror case 3.2.2
                        k = k.parent
                        self.left_rotate(k)
                    # mirror case 3.2.1
                    k.parent.color = 0
                    k.parent.parent.color = 1
                    self.right_rotate(k.parent.parent)
            if k == self.root:
                break
        self.root.color = 0

    # print the tree structure on the screen
    def pretty_print(self):
        self.__print_helper(self.root, "", True)

    def __print_helper(self, node: IntervalNode, indent, last):
        # print the tree structure on the screen
        if node != self.TNULL:
            sys.stdout.write(indent)
            if last:
                sys.stdout.write("R----")
                indent += "     "
            else:
                sys.stdout.write("L----")
                indent += "|    "

            s_color = "RED" if node.color == 1 else "BLACK"
            # print(str(node.data) + "(" + s_color + ")")
            print(str(node.node_min) + "-" + str(node.node_max) + "," \
                + str(node.subtree_min) + "-" + str(node.subtree_max) + "," \
                + str(node.overlap_count) + "," + str(node.update_count) + "," + str(node.create_count)
                  + "(" + s_color + ")")
            self.__print_helper(node.left, indent, False)
            self.__print_helper(node.right, indent, True)

# if __name__ == "__main__":
#     # bst = RedBlackTree()
#     # bst.insert(8)
#     # bst.insert(18)
#     # bst.insert(5)
#     # bst.insert(15)
#     # bst.insert(17)
#     # bst.insert(25)
#     # bst.insert(40)
#     # bst.insert(80)
#     # bst.delete_node(25)
#     # bst.pretty_print()
#     ist = IntervalStatisticTree()

    # max_window_size: int, frozen_threshold: int, similar_threshold: float
    # ist = IntervalStatisticTree(20, 4, 0.9)
    # print(ist.insert(16,21))
    # print(ist.insert(8,9))
    # print(ist.insert(5,8))
    # ist.insert_new_key(7)
    # print(ist.insert(25,30))
    # print(ist.insert(15,23))
    # ist.insert_new_key(17)
    # print(ist.insert(14,25))
    # ist.insert_new_key(20)
    # print(ist.insert(3, 19))
    # ist.pretty_print()

class RandomTree(IntervalStatisticTree):
    def __init__(self, max_window_size: int, random_ratio: float, frozen_threshold: int, similar_threshold: float, do_rebuilt):
        super().__init__(max_window_size, frozen_threshold, similar_threshold, do_rebuilt)
        # self.max_window_size = max_window_size
        self.random_ratio = random_ratio
        self.operation_list = self.random_decision()

    def random_decision(self):
        pass_operation = [0] * int(self.random_ratio * self.max_window_size)
        create_operation = [1] * int(self.max_window_size - self.random_ratio * self.max_window_size)
        operation_list = pass_operation + create_operation
        random.shuffle(operation_list)
        create_sp_f = "./create_sp_info.txt"
        create_sp_results = open(create_sp_f, 'a')
        for i in range(len(operation_list)):
            operation = str(i) + ":" + str(operation_list[i])
            create_sp_results.write(operation)
            create_sp_results.write("\n")
        create_sp_results.close()
        return operation_list

    def insert(self, q_min, q_max, q_length, n, IO_unit, is_short_query):
        # Ordinary Binary Search Insertion
        # node = self.create_new_node(q_min, q_max)
        node, is_new_range = self.create_new_node_with_similar_t(q_min, q_max, q_length)

        # print("[", q_min , "-", q_max, "] is new range = ", is_new_range)

        # increase tree size
        self.query_size += 1

        if is_new_range:
            node.parent = None
            node.data = q_min
            node.left = self.TNULL
            node.right = self.TNULL
            node.color = 1  # new node must be red

            y = None
            x: IntervalNode = self.root

            while x != self.TNULL:
                y = x
                if node.data < x.data:
                    if node.subtree_min < x.subtree_min:
                        x.subtree_min = node.subtree_min
                    if node.subtree_max > x.subtree_max:
                        x.subtree_max = node.subtree_max
                    x = x.left
                else:
                    if node.subtree_min < x.subtree_min:
                        x.subtree_min = node.subtree_min
                    if node.subtree_max > x.subtree_max:
                        x.subtree_max = node.subtree_max
                    x = x.right

            # y is parent of x
            node.parent = y
            if y == None:
                self.root = node
            elif node.data < y.data:
                y.left = node
            else:
                y.right = node

            # if new node is a root node, simply return
            if node.parent == None:
                node.color = 0
                return False
                # print("frozen")
                # return -1

            # if the grandparent is None, simply return
            if node.parent.parent == None:
                return False
                # print("frozen")
                # return -1

            # Fix the tree
            self.__fix_insert(node)

        if self.query_size >= self.frozen_threshold:
            operation = self.operation_list[self.query_size - 1]
            if operation == 1:
                return True
            else:
                return False
        else:
            # print("frozen")
            return False

    def __fix_insert(self, k):
        while k.parent.color == 1:
            if k.parent == k.parent.parent.right:
                u = k.parent.parent.left  # uncle
                if u.color == 1:
                    # case 3.1
                    u.color = 0
                    k.parent.color = 0
                    k.parent.parent.color = 1
                    k = k.parent.parent
                else:
                    if k == k.parent.left:
                        # case 3.2.2
                        k = k.parent
                        self.right_rotate(k)
                    # case 3.2.1
                    k.parent.color = 0
                    k.parent.parent.color = 1
                    self.left_rotate(k.parent.parent)
            else:
                u = k.parent.parent.right  # uncle

                if u.color == 1:
                    # mirror case 3.1
                    u.color = 0
                    k.parent.color = 0
                    k.parent.parent.color = 1
                    k = k.parent.parent
                else:
                    if k == k.parent.right:
                        # mirror case 3.2.2
                        k = k.parent
                        self.left_rotate(k)
                    # mirror case 3.2.1
                    k.parent.color = 0
                    k.parent.parent.color = 1
                    self.right_rotate(k.parent.parent)
            if k == self.root:
                break
        self.root.color = 0