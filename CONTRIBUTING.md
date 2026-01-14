# Contributing to Fair ECG Decision System

Thank you for your interest in contributing! This project focuses on fairness, safety, and interpretability in healthcare AI.

## Ways to Contribute

- **Report Bugs:** Use GitHub Issues with detailed reproduction steps
- **Suggest Features:** Propose enhancements via GitHub Issues
- **Submit Code:** Fork, create feature branch, and submit pull request
- **Improve Documentation:** Help make the project more accessible
- **Validate Results:** Test on new datasets or clinical settings

## Development Setup

```bash
# Clone repository
git clone https://github.com/Lisa-0110/fair-ecg-decision-system.git
cd fair-ecg-decision-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/
```

## Code Guidelines

### Style
- Follow PEP 8
- Use type hints for function signatures
- Write docstrings for all public functions
- Maximum line length: 100 characters

### Testing
- Add tests for new features in `tests/`
- Maintain or improve test coverage
- Run `pytest tests/` before submitting

### Documentation
- Update README.md for user-facing changes
- Include docstrings with examples
- Document clinical relevance of new features

## Pull Request Process

**Before submitting:**
1. Code follows style guidelines
2. Tests pass locally
3. Documentation updated
4. Meaningful commit messages

**PR Checklist:**
- [ ] Code follows PEP 8
- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] Fairness implications considered
- [ ] Clinical validity addressed

## Research Ethics

When contributing, please consider:

**Fairness:**
- Evaluate impact on demographic subgroups
- Consider intersectionality
- Prioritize vulnerable populations
- Document tradeoffs transparently

**Clinical Validity:**
- Consult medical literature
- Validate with domain experts
- Consider patient safety
- Think about real-world deployment

**Transparency:**
- Report negative results
- Acknowledge limitations
- Be honest about uncertainties
- Avoid overselling capabilities

## Areas for Contribution

### Short-Term
- Multi-lead fusion (currently Lead II only)
- Additional unit tests
- Documentation improvements
- Web interface for clinicians

### Medium-Term
- Deep learning models (CNN for raw ECG)
- SHAP/LIME explainability
- External dataset validation
- Real-time processing optimization

### Long-Term
- Federated learning across hospitals
- Personalized thresholds
- Multi-modal integration (ECG + labs + notes)
- Clinical validation study

## Code of Conduct

We are committed to providing a welcoming and inclusive environment.

**Expected behavior:**
- Use welcoming and inclusive language
- Respect differing viewpoints
- Accept constructive criticism
- Focus on what's best for the project

**Unacceptable behavior:**
- Harassment of any kind
- Discriminatory language or imagery
- Personal attacks
- Publishing others' private information

Violations may be reported to the project maintainers.

## Getting Help

- **Questions:** GitHub Discussions
- **Bugs:** GitHub Issues
- **General Contact:** See README.md

## Recognition

Contributors will be:
- Listed in project contributors
- Mentioned in release notes
- Acknowledged in research publications (if applicable)

---

**Thank you for helping build fair, safe, and effective healthcare AI!**
