# Contributing Guidelines

Thank you for your interest in contributing to the **Campus Placement Predictor** project!

## How to Contribute
1. **Fork the Repository**: Create your personal branch from `main` or `master`.
2. **Set Up Development Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Or on Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Make Atomic Changes**:
   - Follow clean code practices (PEP 8).
   - Maintain docstrings and type hints.
   - Include unit tests in `tests/` for any new functionality.
4. **Run the Test Suite**:
   ```bash
   pytest tests/ -v
   ```
5. **Submit a Pull Request**: Provide a descriptive summary of your changes and reference any related issues.
