"""Verify submission.zip integrity."""
import zipfile, ast

z = zipfile.ZipFile('submissions/submission.zip', 'r')
files = z.namelist()
print('Files:', files)
assert 'run.py' in files

run_py = z.read('run.py').decode('utf-8')
tree = ast.parse(run_py)
print('AST: OK')

banned = ['import os', 'import subprocess', 'import socket', 'import ctypes', 'import builtins']
for b in banned:
    assert b not in run_py, f'Banned import found: {b}'
print('Banned imports: CLEAN')

assert 'box_and_model_avg' in run_py, 'conf_type not found'
assert 'sigma=5.0' in run_py, 'sigma=5.0 not found'
assert 'sigma=1.5' not in run_py, 'OLD sigma=1.5 found!'
assert 'conf_type=conf_type,' in run_py, 'conf_type param not passed'
print('Params: conf_type=bma OK, sigma=5.0 OK')

total = sum(z.getinfo(n).file_size for n in files)
print(f'Uncompressed: {total/1e6:.1f} MB (limit 420)')
onnx_count = len([f for f in files if f.endswith('.onnx')])
print(f'Weight files: {onnx_count} (limit 3)')
py_count = len([f for f in files if f.endswith('.py')])
print(f'Python files: {py_count} (limit 10)')
print('ALL CHECKS PASSED')
z.close()
