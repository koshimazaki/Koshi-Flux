# Deforum

AI generation protocol and CLI tool.

## Installation

```bash
pip install deforum
```

## Publish
```bash
python -m build
python -m twine upload dist/*
```

## Structure
```
  src/
  ├── cli/                     # Interface layer CLI
  │   └── main.py
  └── deforum/                 # Core library
      ├── config/              # Configuration management (settings, validation, etc.)
      ├── core/                # Core shared utilities (exceptions, logging, etc.)
      └── utils/               # Utility functions (file handling, etc.)
```