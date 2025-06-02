from __future__ import annotations
from typing import Type, Literal

import nodes
from comfy_execution.graph_utils import is_link
from comfy.comfy_types.node_typing import ComfyNodeABC, InputTypeDict, InputTypeOptions

class DependencyCycleError(Exception):
    pass

class NodeInputError(Exception):
    pass

class NodeNotFoundError(Exception):
    pass

# 忽略ephemeral（直接默认他们是空的），DynamicPrompt就是对prompt的一个简单封装。
class DynamicPrompt:
    def __init__(self, original_prompt):
        # The original prompt provided by the user
        self.original_prompt = original_prompt
        # Any extra pieces of the graph created during execution 应该是for循环之类的subgraph相关的，可以不管
        self.ephemeral_prompt = {}
        self.ephemeral_parents = {}
        self.ephemeral_display = {}

    def get_node(self, node_id):
        if node_id in self.ephemeral_prompt:
            return self.ephemeral_prompt[node_id]
        if node_id in self.original_prompt:
            return self.original_prompt[node_id]
        raise NodeNotFoundError(f"Node {node_id} not found")

    def has_node(self, node_id):
        return node_id in self.original_prompt or node_id in self.ephemeral_prompt

    def add_ephemeral_node(self, node_id, node_info, parent_id, display_id):
        self.ephemeral_prompt[node_id] = node_info
        self.ephemeral_parents[node_id] = parent_id
        self.ephemeral_display[node_id] = display_id

    # 注释: 获得根节点的ID
    def get_real_node_id(self, node_id):
        while node_id in self.ephemeral_parents:
            node_id = self.ephemeral_parents[node_id]
        return node_id

    # 注释: 获得直接父节点的ID
    def get_parent_node_id(self, node_id):
        return self.ephemeral_parents.get(node_id, None)

    def get_display_node_id(self, node_id):
        while node_id in self.ephemeral_display:
            node_id = self.ephemeral_display[node_id]
        return node_id

    def all_node_ids(self):
        return set(self.original_prompt.keys()).union(set(self.ephemeral_prompt.keys()))

    def get_original_prompt(self):
        return self.original_prompt

def get_input_info(
    class_def: Type[ComfyNodeABC],
    input_name: str,
    valid_inputs: InputTypeDict | None = None
) -> tuple[str, Literal["required", "optional", "hidden"], InputTypeOptions] | tuple[None, None, None]:
    """Get the input type, category, and extra info for a given input name.

    Arguments:
        class_def: The class definition of the node.
        input_name: The name of the input to get info for.
        valid_inputs: The valid inputs for the node, or None to use the class_def.INPUT_TYPES().

    Returns:
        tuple[str, str, dict] | tuple[None, None, None]: The input type, category, and extra info for the input name.
    """

    valid_inputs = valid_inputs or class_def.INPUT_TYPES()
    input_info = None
    input_category = None
    if "required" in valid_inputs and input_name in valid_inputs["required"]:
        input_category = "required"
        input_info = valid_inputs["required"][input_name]
    elif "optional" in valid_inputs and input_name in valid_inputs["optional"]:
        input_category = "optional"
        input_info = valid_inputs["optional"][input_name]
    elif "hidden" in valid_inputs and input_name in valid_inputs["hidden"]:
        input_category = "hidden"
        input_info = valid_inputs["hidden"][input_name]
    if input_info is None:
        return None, None, None
    input_type = input_info[0]
    if len(input_info) > 1:
        extra_info = input_info[1]
    else:
        extra_info = {}
    return input_type, input_category, extra_info


# 因为TopologicalSort只有ExecutionList这个子类，这里为了方便理解，我把TopologicalSort和ExecutionList合并了一下
# class ExecutionList(TopologicalSort):
class ExecutionList:
    """
    ExecutionList implements a topological dissolve of the graph. After a node is staged for execution,
    it can still be returned to the graph after having further dependencies added.
    
    ExecutionList 实现了一种图拓扑的解法。当一个节点被标记为可执行的时候，如果有更近一步的关系被添加时，这个节点仍然可以变为graph的其他状态(非准备状态)。
    """
    def __init__(self, dynprompt, output_cache):        
        self.dynprompt = dynprompt
        self.pendingNodes = {} # key: node_id, value: 是否等待中。其实用个list或者set来替代会好一点。只要node_id在这个dict中，那么就是在等待，否则就不是在等待。
        self.blockCount = {} # key: node_id, value: 这个节点的被多少个先前的节点阻塞住，如果是0，那么就可以执行。
        self.blocking = {} # key: node_id, value: dict。value记录这个节点的第i个输出阻塞了哪些后续节点。比如，self.blocking[0][1][2] = True，表示第0个节点的第2个输出，阻塞了第1个节点。
        self.output_cache = output_cache # 如果这个节点的执行结果已经在缓存中了，那么就不再执行了
        self.staged_node_id = None

    def get_input_info(self, unique_id, input_name):
        class_type = self.dynprompt.get_node(unique_id)["class_type"]
        class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
        return get_input_info(class_def, input_name)

    def make_input_strong_link(self, to_node_id, to_input):
        inputs = self.dynprompt.get_node(to_node_id)["inputs"]
        if to_input not in inputs:
            raise NodeInputError(f"Node {to_node_id} says it needs input {to_input}, but there is no input to that node at all")
        value = inputs[to_input]
        if not is_link(value):
            raise NodeInputError(f"Node {to_node_id} says it needs input {to_input}, but that value is a constant")
        from_node_id, from_socket = value
        self.add_strong_link(from_node_id, from_socket, to_node_id)

    def add_strong_link(self, from_node_id, from_socket, to_node_id):
        if not self.is_cached(from_node_id):
            self.add_node(from_node_id) # 递归把所有需要计算的节点都加到计算图里面。这里有个小疑问，进add_node的时候似乎一定会命中unique_id in self.pendingNodes那行？那这样就相当于不需要执行。
            if to_node_id not in self.blocking[from_node_id]:
                self.blocking[from_node_id][to_node_id] = {}
                self.blockCount[to_node_id] += 1
            self.blocking[from_node_id][to_node_id][from_socket] = True

    def add_node(self, node_unique_id, include_lazy=False, subgraph_nodes=None):
        node_ids = [node_unique_id] # 用栈来实现深度优先遍历
        links = []

        while len(node_ids) > 0:
            unique_id = node_ids.pop()
            if unique_id in self.pendingNodes:
                continue

            self.pendingNodes[unique_id] = True
            self.blockCount[unique_id] = 0
            self.blocking[unique_id] = {}

            inputs = self.dynprompt.get_node(unique_id)["inputs"]
            for input_name in inputs:
                value = inputs[input_name]
                if is_link(value): # 如果是连接，说明前面存在一个节点
                    from_node_id, from_socket = value # 前面节点的id就是from_node_id；当前节点的这个input_name的输入需要的是from_node_id的第from_socket个输出（每个节点可能有多个输出）
                    if subgraph_nodes is not None and from_node_id not in subgraph_nodes:
                        continue
                    _, _, input_info = self.get_input_info(unique_id, input_name)
                    is_lazy = input_info is not None and "lazy" in input_info and input_info["lazy"]
                    if (include_lazy or not is_lazy) and not self.is_cached(from_node_id): # 如果from_node_id的结果已经缓存了，就不会再添加了
                        node_ids.append(from_node_id)
                        links.append((from_node_id, from_socket, unique_id))

        for link in links:
            self.add_strong_link(*link)

    # 获取当前可以直接执行的节点列表（也就是前面没有被阻塞的节点）
    def get_ready_nodes(self):
        return [node_id for node_id in self.pendingNodes if self.blockCount[node_id] == 0]

    def pop_node(self, unique_id):
        del self.pendingNodes[unique_id]
        for blocked_node_id in self.blocking[unique_id]:
            self.blockCount[blocked_node_id] -= 1
        del self.blocking[unique_id]

    def is_empty(self):
        return len(self.pendingNodes) == 0

    def is_cached(self, node_id):
        return self.output_cache.get(node_id) is not None

    def stage_node_execution(self):
        assert self.staged_node_id is None
        if self.is_empty():
            return None, None, None
        available = self.get_ready_nodes()
        if len(available) == 0: # 如果没有for循环，应该不用管这里？
            cycled_nodes = self.get_nodes_in_cycle()
            # Because cycles composed entirely of static nodes are caught during initial validation,
            # we will 'blame' the first node in the cycle that is not a static node.
            blamed_node = cycled_nodes[0]
            for node_id in cycled_nodes:
                display_node_id = self.dynprompt.get_display_node_id(node_id)
                if display_node_id != node_id:
                    blamed_node = display_node_id
                    break
            ex = DependencyCycleError("Dependency cycle detected")
            error_details = {
                "node_id": blamed_node,
                "exception_message": str(ex),
                "exception_type": "graph.DependencyCycleError",
                "traceback": [],
                "current_inputs": []
            }
            return None, error_details, ex

        self.staged_node_id = self.ux_friendly_pick_node(available)
        return self.staged_node_id, None, None

    # 从available里面优先挑一个可以可视化的节点 先做计算，但其实不会减少总的计算耗时。
    def ux_friendly_pick_node(self, node_list):
        # If an output node is available, do that first.
        # Technically this has no effect on the overall length of execution, but it feels better as a user
        # for a PreviewImage to display a result as soon as it can
        # Some other heuristics could probably be used here to improve the UX further.
        def is_output(node_id):
            class_type = self.dynprompt.get_node(node_id)["class_type"]
            class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
            if hasattr(class_def, 'OUTPUT_NODE') and class_def.OUTPUT_NODE == True:
                return True
            return False

        for node_id in node_list:
            if is_output(node_id):
                return node_id

        #This should handle the VAEDecode -> preview case
        for node_id in node_list:
            for blocked_node_id in self.blocking[node_id]:
                if is_output(blocked_node_id):
                    return node_id

        #This should handle the VAELoader -> VAEDecode -> preview case
        for node_id in node_list:
            for blocked_node_id in self.blocking[node_id]:
                for blocked_node_id1 in self.blocking[blocked_node_id]:
                    if is_output(blocked_node_id1):
                        return node_id

        #TODO: this function should be improved
        return node_list[0]

    def unstage_node_execution(self):
        assert self.staged_node_id is not None
        self.staged_node_id = None

    def complete_node_execution(self):
        node_id = self.staged_node_id
        self.pop_node(node_id)
        self.staged_node_id = None

    def get_nodes_in_cycle(self):
        # We'll dissolve the graph in reverse topological order to leave only the nodes in the cycle.
        # We're skipping some of the performance optimizations from the original TopologicalSort to keep
        # the code simple (and because having a cycle in the first place is a catastrophic error)
        blocked_by = { node_id: {} for node_id in self.pendingNodes }
        for from_node_id in self.blocking:
            for to_node_id in self.blocking[from_node_id]:
                if True in self.blocking[from_node_id][to_node_id].values():
                    blocked_by[to_node_id][from_node_id] = True
        to_remove = [node_id for node_id in blocked_by if len(blocked_by[node_id]) == 0]
        while len(to_remove) > 0:
            for node_id in to_remove:
                for to_node_id in blocked_by:
                    if node_id in blocked_by[to_node_id]:
                        del blocked_by[to_node_id][node_id]
                del blocked_by[node_id]
            to_remove = [node_id for node_id in blocked_by if len(blocked_by[node_id]) == 0]
        return list(blocked_by.keys())

class ExecutionBlocker:
    """
    Return this from a node and any users will be blocked with the given error message.
    If the message is None, execution will be blocked silently instead.
    Generally, you should avoid using this functionality unless absolutely necessary. Whenever it's
    possible, a lazy input will be more efficient and have a better user experience.
    This functionality is useful in two cases:
    1. You want to conditionally prevent an output node from executing. (Particularly a built-in node
       like SaveImage. For your own output nodes, I would recommend just adding a BOOL input and using
       lazy evaluation to let it conditionally disable itself.)
    2. You have a node with multiple possible outputs, some of which are invalid and should not be used.
       (I would recommend not making nodes like this in the future -- instead, make multiple nodes with
       different outputs. Unfortunately, there are several popular existing nodes using this pattern.)
    """
    def __init__(self, message):
        self.message = message

