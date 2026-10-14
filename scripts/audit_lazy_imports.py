#!/usr/bin/env python3
"""Audit heavy imports and safely localize imports used only in function bodies.

This is a mechanical import-only rewrite. Definitions, decorators, defaults,
annotations, guarded availability checks and side-effecting top-level code are
not moved. Tests are audited as well; pytest's collection checks stay explicit.
"""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ('basil_core', 'noise_comm', 'gui', 'reporting', 'scripts', 'tests')
HEAVY = {'numpy', 'tensorflow', 'matplotlib', 'PIL', 'wandb', 'pandas',
         'scipy', 'sklearn', 'seaborn', 'psutil', 'torch', 'tonic'}


def source_files():
    return sorted({ROOT/'run_gui.py'} | {p for d in DIRECTORIES for p in (ROOT/d).rglob('*.py')})


def plan(path):
    source = path.read_text()
    tree = ast.parse(source)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    # An import inside an outer function serves its nested functions too.
    owners = {}
    for function in functions:
        for statement in function.body:
            for node in ast.walk(statement):
                owners.setdefault(node, function)
    lines = source.splitlines(keepends=True)
    edits = []; insertions = {}; inventory = []
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or '']
        deferred_loss = isinstance(node, ast.ImportFrom) and (node.module or '').endswith('trainer') and any(a.name == 'lossFn' for a in node.names)
        if not deferred_loss and not any(m.split('.')[0] in HEAVY for m in modules):
            continue
        names = {a.asname or (a.name.split('.')[0] if isinstance(node, ast.Import) else a.name) for a in node.names}
        loads = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in names]
        # Annotations remain global for get_type_hints; definition-time use is
        # not equivalent to a function-body import. Preserve and explain it.
        required = [n for n in loads if n not in owners]
        if required or any(isinstance(n, (ast.Global, ast.Nonlocal)) and names.intersection(n.names) for n in ast.walk(tree)):
            inventory.append(dict(file=str(path.relative_to(ROOT)), line=node.lineno,
                                  statement=ast.unparse(node), action='definition_or_annotation_dependency'))
            continue
        users = {owners[n] for n in loads}
        if not users:
            # Retain unused exports rather than silently breaking external APIs.
            inventory.append(dict(file=str(path.relative_to(ROOT)), line=node.lineno,
                                  statement=ast.unparse(node), action='public_or_unused_binding'))
            continue
        original = ''.join(lines[node.lineno-1:node.end_lineno]).strip()
        prepared = []
        for user in users:
            body = user.body
            if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                first = body[1] if len(body) > 1 else None
            else:
                first = body[0]
            if first is None or first.lineno <= user.lineno:
                required.append(user)
                break
            indent = ' ' * first.col_offset
            prepared.append((first.lineno-1, indent + original.replace('\n', '\n'+indent) + '\n'))
        if required:
            inventory.append(dict(file=str(path.relative_to(ROOT)), line=node.lineno,
                                  statement=ast.unparse(node), action='one_line_function_binding'))
            continue
        for index, addition in prepared:
            insertions.setdefault(index, []).append(addition)
        edits.append((node.lineno-1, node.end_lineno, []))
        inventory.append(dict(file=str(path.relative_to(ROOT)), line=node.lineno,
                              statement=ast.unparse(node), action='function_local', functions=sorted(f.name for f in users)))
    for index, additions in insertions.items():
        edits.append((index, index, sorted(set(additions))))
    for start, end, replacement in sorted(edits, reverse=True):
        lines[start:end] = replacement
    rewritten = ''.join(lines)
    ast.parse(rewritten)
    return rewritten, inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Apply mechanical import-only changes.')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    inventory = []; changed = []
    for path in source_files():
        rewritten, entries = plan(path)
        inventory.extend(entries)
        if rewritten != path.read_text():
            changed.append(str(path.relative_to(ROOT)))
            if args.apply:
                path.write_text(rewritten)
    report = dict(sourceFiles=len(source_files()), filesWithMovableImports=changed, imports=inventory)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(sourceFiles=report['sourceFiles'], changedFiles=len(changed),
                         moved=sum(i['action']=='function_local' for i in inventory),
                         retained=sum(i['action']!='function_local' for i in inventory))))


if __name__ == '__main__':
    main()
