# Code Formatting in MorphoSnaker

MorphoSnaker uses several tools to maintain code quality and consistency. This guide outlines how to use these tools effectively.

## Formatting Tools

### Black

Black is our primary code formatter. It enforces a consistent style across the entire codebase.

To run Black:

```bash
    black .
```

### isort

isort is used to automatically sort and organize imports.
To run isort:

```bash
    isort .
```

### Flake8
Flake8 checks the code against the PEP 8 style guide and for programmatic errors.
To run Flake8:

```bash
    flake8 .
```

### mypy
mypy performs static type checking on the codebase.
To run mypy:


```bash
    mypy .
```

## Automatic Formatting
To automatically format your code before committing, you can set up pre-commit hooks:

### Install pre-commit:

```bash
    pip install pre-commit
```

### Set up the pre-commit hooks:
```bash
    pre-commit install
```

## VSCode Integration
If you're using VSCode, you can set up automatic formatting on save. Add the following to your settings.json:

```json
    {
    "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
        "source.organizeImports": "explicit"
        },
        "editor.rulers": [
        79
        ],
        "files.trimTrailingWhitespace": true
    },
    "black-formatter.args": [
        "--line-length",
        "79",
        "--preview"
    ],
    "isort.args": [
        "--profile",
        "black",
        "--line-length",
        "79"
    ],
    "flake8.args": [
        "--max-line-length=79"
    ],
    "python.linting.mypyEnabled": true,
    "python.languageServer": "Pylance",
    "autoDocstring.docstringFormat": "google",
    "autoDocstring.startOnNewLine": true,
    "autoDocstring.includeName": true,
    "python.formatting.provider": "black"
    }
```

# Install required VSCode extensions
To fully utilize these settings, install the following VSCode extensions:

Python
Black Formatter
isort
Flake8
mypy
Python Docstring Formatter

These settings and tools will help maintain consistent code formatting, including properly formatted docstrings, across your project.

# Select appropriate python env for development 
If some of the imports appear missing such as "Import "n2v.internals.N2V_DataGenerator" could not be resolvedPylancereportMissingImports"

1.	Open the command palette (Ctrl+Shift+P or Cmd+Shift+P on macOS) and search for “Python: Select Interpreter”.

2.	Choose the interpreter that points to your Poetry environment. This usually appears as something like .venv/bin/python or path/to/your/poetry/environment/bin/python.