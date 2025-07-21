# Contributing to NiceWebRL

Thank you for your interest in contributing to NiceWebRL! This document provides guidelines and instructions for contributing to the project.

## Development Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/nicewebrl.git
   cd nicewebrl
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. Install JAX and JAXlib:
   ```bash
   pip install "jax>=0.2.26" "jaxlib>=0.1.74"
   ```

## Code Style and Formatting

We use Ruff for code formatting and linting. To ensure your code follows our style guidelines:

1. Install Ruff:
   ```bash
   pip install ruff
   ```

2. Run Ruff to check your code:
   ```bash
   ruff check .
   ```

3. Run Ruff to automatically fix issues:
   ```bash
   ruff check --fix .
   ```

4. Format your code:
   ```bash
   ruff format .
   ```

### Style Guidelines

- Follow the Google Python Style Guide
- Use type hints for all function parameters and return values
- Write docstrings for all public functions, classes, and methods
- Keep lines under 80 characters
- Use meaningful variable and function names
- Add comments for complex logic

## Running Tests

Before submitting a pull request, ensure all tests pass:

1. Run the test suite:
   ```bash
   pytest
   ```

2. Run tests with coverage:
   ```bash
   pytest --cov=nicewebrl
   ```

## Pull Request Process

1. **Issue First**: For significant changes, please open an issue first to discuss the proposed changes.

2. **Work-in-Progress PRs**: We accept work-in-progress PRs for early feedback. Please mark them with the "WIP" prefix in the title.

3. **Branch Naming**: Use descriptive branch names:
   - `feature/your-feature-name`
   - `fix/your-fix-name`
   - `docs/your-docs-update`

4. **Commit Messages**: Write clear, descriptive commit messages that explain the "why" of your changes.

5. **PR Description**: Include:
   - A clear description of the changes
   - Related issue numbers
   - Any breaking changes
   - Screenshots for UI changes

## Development Workflow

1. Keep your fork up to date:
   ```bash
   git remote add upstream https://github.com/original-owner/nicewebrl.git
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. Make your changes and commit them:
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

4. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

## Additional Guidelines

- Write clear, concise documentation
- Add tests for new features
- Update existing tests if you change functionality
- Keep dependencies up to date
- Follow semantic versioning for releases

## Questions?

If you have any questions about contributing, please open an issue or contact the maintainers.

Thank you for contributing to NiceWebRL! 