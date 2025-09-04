# Python Overview Examples

## Objective

This repository contains comprehensive Python examples designed to demonstrate and explain fundamental Python concepts and programming patterns. The examples are structured to provide hands-on learning experiences for students studying discrete algorithms and scientific applications.

## Topics Covered

### Core Python Concepts
- **Functions**: Function definitions, parameters, return values, and scope
- **Classes**: Object-oriented programming, inheritance, and encapsulation
- **Exceptions**: Error handling, try-catch blocks, and custom exceptions
- **Type Hints**: Static typing with `typing` module and type annotations
- **Lambda Functions**: Anonymous functions and functional programming concepts

### Python-Specific Features
- **List Comprehensions**: Concise list creation and filtering
- **Dictionary Comprehensions**: Efficient dictionary operations
- **Set Comprehensions**: Set operations and filtering
- **Generator Expressions**: Memory-efficient iteration patterns

### Advanced Patterns
- **Decorators**: Function and class decorators
- **Context Managers**: Resource management with `with` statements
- **Iterators and Generators**: Custom iteration patterns
- **File I/O**: Reading and writing different file formats

## Project Structure

```
PythonOverview/
├── README.md
├── testenv.py
├── examples/
│   ├── functions/
│   ├── classes/
│   ├── comprehensions/
│   ├── exceptions/
│   ├── type_hints/
│   └── advanced_patterns/
└── requirements.txt
```

## Getting Started

### Prerequisites
- Python 3.8 or higher
- `uv` package manager (recommended)

### Setup Environment
```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate     # On Windows

# Install dependencies
uv pip install -r requirements.txt
```

### Running Examples
```bash
# Run specific example
uv run python examples/functions/basic_functions.py

# Run all examples
uv run python -m pytest tests/
```

## Learning Objectives

By working through these examples, students will:

1. **Understand Python syntax and semantics** through practical examples
2. **Master functional programming concepts** including lambda functions and comprehensions
3. **Learn object-oriented programming** with classes and inheritance
4. **Handle errors gracefully** using exception handling mechanisms
5. **Write type-safe code** using modern Python type hints
6. **Apply Python patterns** to scientific and algorithmic problems

## Target Audience

- Students in ISC 4221C: Discrete Algorithms for Science Applications
- Python beginners looking for practical examples
- Scientists and researchers learning Python for data analysis
- Anyone interested in modern Python programming practices

## Contributing

Feel free to add new examples or improve existing ones. Please ensure:
- All code follows PEP 8 style guidelines
- Functions include proper type hints and docstrings
- Examples are clear and well-documented
- Code is tested and runs without errors

## License

This project is part of the ISC 4221C course materials and is intended for educational purposes.
