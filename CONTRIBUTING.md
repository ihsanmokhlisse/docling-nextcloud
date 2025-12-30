# Contributing to Docling Knowledge Base

First off, **thank you** for considering contributing to Docling KB! 🎉

Every contribution helps make this project better for everyone.

## 💖 Ways to Contribute

### 🐛 Report Bugs
Found a bug? Please open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Your environment (Nextcloud version, OS, etc.)

### 💡 Suggest Features
Have an idea? We'd love to hear it!
- Check if it's already been suggested
- Open an issue with the "feature request" label
- Describe the use case and potential benefits

### 💻 Submit Code
Ready to code? Awesome!

1. **Fork** the repository
2. **Clone** your fork locally
3. **Create a branch** for your feature/fix:
   ```bash
   git checkout -b feature/amazing-feature
   ```
4. **Make your changes** and test them
5. **Commit** with a clear message:
   ```bash
   git commit -m "Add amazing feature that does X"
   ```
6. **Push** to your fork:
   ```bash
   git push origin feature/amazing-feature
   ```
7. **Open a Pull Request**

### 📚 Improve Documentation
Documentation improvements are always welcome:
- Fix typos
- Add examples
- Clarify confusing sections
- Translate to other languages

### 🌍 Translations
Help make Docling KB accessible to more users by adding translations.

## 🛠️ Development Setup

```bash
# Clone the repository
git clone https://github.com/ihsanmokhlisse/docling-nextcloud.git
cd docling-nextcloud

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# Run locally
make dev

# Run tests
make test

# Run linters
make lint
```

## 📋 Code Guidelines

- **Python**: Follow PEP 8, use type hints
- **Formatting**: Use `black` and `ruff`
- **Tests**: Add tests for new features
- **Commits**: Use clear, descriptive commit messages
- **Documentation**: Update docs for any user-facing changes

## 🧪 Testing

Before submitting a PR, please ensure:

```bash
# Run all tests
make test

# Run linters
make lint

# Format code
make format
```

## 📝 Pull Request Guidelines

- Keep PRs focused on a single feature/fix
- Update documentation if needed
- Add tests for new functionality
- Ensure all tests pass
- Reference related issues in the PR description

## 💬 Questions?

- Open a [Discussion](https://github.com/ihsanmokhlisse/docling-nextcloud/discussions)
- Check existing [Issues](https://github.com/ihsanmokhlisse/docling-nextcloud/issues)

## 🏆 Contributors

Thank you to all our contributors! Your support keeps this project alive.

---

## 💖 Support the Project

If you find this project useful, please consider:

- ⭐ **Starring** the repository
- 💝 **Sponsoring** on [GitHub Sponsors](https://github.com/sponsors/ihsanmokhlisse)
- ☕ **Buying a coffee** on [Buy Me a Coffee](https://buymeacoffee.com/ihsanmokhlisse)

Every contribution, whether code or coffee, helps maintain and improve this project!

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/ihsanmokhlisse">Ihsan Mokhlis</a> and contributors
</p>
