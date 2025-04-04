import ast
import os

bert_path = "models/bert/modeling_bert.py"

# 存储调用信息 (文件名, 行号, 调用表达式)
call_graph = []
import_aliases = {}  # 存储 import 别名映射，例如 {"nn": "torch.nn", "CrossEntropyLoss": "torch.nn.CrossEntropyLoss"}

class PyTorchCallVisitor(ast.NodeVisitor):
    def __init__(self, filename):
        self.filename = filename

    def visit_ImportFrom(self, node):
        """解析 `from torch.nn import BCEWithLogitsLoss` 形式"""
        if node.module and node.module.startswith("torch"):
            for alias in node.names:
                full_name = f"{node.module}.{alias.name}"
                import_aliases[alias.asname or alias.name] = full_name  # 处理 `import as`

    def visit_Import(self, node):
        """解析 `import torch.nn as nn` 形式"""
        for alias in node.names:
            if alias.name.startswith("torch"):
                import_aliases[alias.asname or alias.name] = alias.name

    def visit_Call(self, node):
        """解析所有 PyTorch API 调用"""
        try:
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                module_name = node.func.value.id  # 例如 'torch', 'nn'
                function_name = node.func.attr    # 例如 'matmul', 'Linear'

                # 解析别名，例如 `nn.CrossEntropyLoss` -> `torch.nn.CrossEntropyLoss`
                full_name = f"{module_name}.{function_name}"
                if module_name in import_aliases:
                    full_name = f"{import_aliases[module_name]}.{function_name}"

                # 只记录 PyTorch API
                if full_name.startswith("torch"):
                    line_number = node.lineno
                    full_expr = ast.unparse(node)  # 获取完整调用表达式
                    call_graph.append((self.filename, line_number, full_name, full_expr))

            elif isinstance(node.func, ast.Name):  # 解析 `CrossEntropyLoss()`
                if node.func.id in import_aliases:
                    full_name = import_aliases[node.func.id]
                    line_number = node.lineno
                    full_expr = ast.unparse(node)
                    call_graph.append((self.filename, line_number, full_name, full_expr))

        except Exception:
            pass

        self.generic_visit(node)

def extract_pytorch_calls(filename):
    with open(filename, "r", encoding="utf-8") as f:
        code = f.read()
        tree = ast.parse(code)
        visitor = PyTorchCallVisitor(filename)
        visitor.visit(tree)

# 解析 `BertModel`
extract_pytorch_calls(bert_path)

# 打印结果
for filename, lineno, full_name, expr in call_graph:
    print(f"{filename}:{lineno} -> {full_name} ({expr})")