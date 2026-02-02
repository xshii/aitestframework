# CLAUDE.md - AI Assistant Guide for aitestframework

This file provides guidance for AI assistants working with this codebase.

## Project Overview

**aitestframework** is an AI testing framework project. This repository is currently in its initial setup phase.

## Repository Structure

```
aitestframework/
├── CLAUDE.md           # This file - AI assistant guidance
├── README.md           # Project documentation (to be created)
├── src/                # Source code (to be created)
│   ├── core/           # Core framework components
│   ├── runners/        # Test runners
│   ├── reporters/      # Test result reporters
│   └── utils/          # Utility functions
├── tests/              # Test files (to be created)
├── docs/               # Documentation (to be created)
└── examples/           # Example usage (to be created)
```

## Development Status

**Current State**: Empty repository - initial setup required

### Recommended Initial Setup Tasks

1. Initialize package management (e.g., `npm init`, `poetry init`, or similar)
2. Set up linting and formatting tools
3. Create basic project structure
4. Add CI/CD configuration
5. Write initial README.md

## Development Workflows

### Getting Started

```bash
# Clone the repository
git clone <repository-url>
cd aitestframework

# Install dependencies (once package manager is configured)
# npm install / pip install -e . / etc.
```

### Branch Naming Convention

- Feature branches: `feature/<description>`
- Bug fixes: `fix/<description>`
- Documentation: `docs/<description>`
- Claude AI branches: `claude/<description>-<session-id>`

### Commit Message Convention

Follow conventional commits format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `chore`: Maintenance tasks

### Code Style Guidelines

- Write clean, readable code with meaningful variable names
- Include docstrings/comments for public APIs
- Follow language-specific best practices
- Keep functions focused and single-purpose
- Prefer composition over inheritance

## Testing

### Running Tests

```bash
# Run all tests (once configured)
# npm test / pytest / etc.
```

### Test File Naming

- Unit tests: `test_<module>.py` or `<module>.test.ts`
- Integration tests: `<module>.integration.test.*`
- Place tests in `tests/` directory mirroring source structure

## Key Conventions for AI Assistants

### When Working on This Codebase

1. **Read Before Writing**: Always read existing files before modifying them
2. **Minimal Changes**: Make focused changes; avoid over-engineering
3. **Test Coverage**: Add tests for new functionality
4. **Documentation**: Update relevant docs when adding features
5. **Commit Often**: Make small, atomic commits with clear messages

### File Operations

- Prefer editing existing files over creating new ones
- Use descriptive file names that reflect content
- Maintain consistent directory structure

### Code Quality

- No hardcoded credentials or secrets
- Handle errors appropriately
- Validate inputs at system boundaries
- Write self-documenting code where possible

### What to Avoid

- Don't add unnecessary dependencies
- Don't over-abstract for hypothetical future needs
- Don't leave TODO comments without context
- Don't commit generated files (build artifacts, node_modules, etc.)

## Environment Variables

Document environment variables here as they are added:

```
# Example (add actual variables as needed)
# AI_TEST_DEBUG=true        # Enable debug logging
# AI_TEST_TIMEOUT=30        # Test timeout in seconds
```

## Dependencies

No dependencies configured yet. Update this section as the project evolves.

## CI/CD

No CI/CD pipeline configured yet. Recommended setup:

- GitHub Actions or similar for automated testing
- Lint checks on pull requests
- Test coverage reporting
- Automated releases

## Troubleshooting

### Common Issues

Document common issues and solutions here as they arise.

## Contributing

1. Create a feature branch from main
2. Make your changes with clear commits
3. Ensure tests pass
4. Submit a pull request with description of changes

## Contact

Repository: xshii/aitestframework

---

*Last updated: 2026-02-02*
*This document should be updated as the project evolves.*
