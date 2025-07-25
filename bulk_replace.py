import os

def replace_in_file(filepath, needle, replacement):
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    new_content = content.replace(needle, replacement)
    if new_content != content:
        print(f"Rewriting: {filepath}")
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(new_content)

def bulk_replace(root_dir, needle, replacement):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.py'):
                fullpath = os.path.join(dirpath, filename)
                replace_in_file(fullpath, needle, replacement)

if __name__ == "__main__":
    root = os.path.dirname(os.path.abspath(__file__))
    bulk_replace(root, '"TENSOR"', '"POSE"')
