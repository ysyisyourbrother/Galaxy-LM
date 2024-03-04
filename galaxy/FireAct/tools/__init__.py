from .search import search
call_count = 0
def call_tools(tool_name, tool_input):
    if tool_name == "search":
        global call_count
        call_count += 1
        return search(tool_input)
    else:
        raise ValueError("Unknown tool name: {}".format(tool_name))
