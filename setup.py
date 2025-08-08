#!/usr/bin/env python3
"""
Setup script for Symbolic Security Layer (SSL)
"""

from setuptools import setup, find_packages
import os
import re

# Read version from __init__.py
def get_version():
    init_file = os.path.join(os.path.dirname(__file__), 'src', 'symbolic_security_layer', '__init__.py')
    with open(init_file, 'r', encoding='utf-8') as f:
        content = f.read()
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", content, re.M)
        if version_match:
            return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

# Read long description from README
def get_long_description():
    readme_file = os.path.join(os.path.dirname(__file__), 'README.md')
    with open(readme_file, 'r', encoding='utf-8') as f:
        return f.read()

# Core dependencies
INSTALL_REQUIRES = [
    'numpy>=1.19.0',
    'requests>=2.25.0',
    'python-dateutil>=2.8.0',
]

# Optional dependencies for different features
EXTRAS_REQUIRE = {
    'tensorflow': [
        'tensorflow>=2.8.0',
    ],
    'pytorch': [
        'torch>=1.10.0',
    ],
    'ml': [
        'tensorflow>=2.8.0',
        'torch>=1.10.0',
        'scikit-learn>=1.0.0',
    ],
    'visualization': [
        'matplotlib>=3.3.0',
        'seaborn>=0.11.0',
        'plotly>=5.0.0',
    ],
    'dev': [
        'pytest>=6.0.0',
        'pytest-cov>=2.10.0',
        'black>=21.0.0',
        'flake8>=3.8.0',
        'mypy>=0.800',
        'sphinx>=4.0.0',
        'sphinx-rtd-theme>=0.5.0',
    ],
    'openai': [
        'openai>=1.0.0',
    ],
    'huggingface': [
        'transformers>=4.20.0',
        'datasets>=2.0.0',
    ],
    'all': [
        # ML frameworks
        'tensorflow>=2.8.0',
        'torch>=1.10.0',
        'scikit-learn>=1.0.0',
        # Visualization
        'matplotlib>=3.3.0',
        'seaborn>=0.11.0',
        'plotly>=5.0.0',
        # AI platforms
        'openai>=1.0.0',
        'transformers>=4.20.0',
        'datasets>=2.0.0',
    ]
}

# Full installation includes all optional dependencies
EXTRAS_REQUIRE['full'] = EXTRAS_REQUIRE['all']

setup(
    name='symbolic-security-layer',
    version=get_version(),
    author='SSL Development Team',
    author_email='ssl-dev@example.com',
    description='Prevents symbolic corruption in AI workflows through semantic anchoring and procedural validation',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/ssl-team/symbolic-security-layer',
    project_urls={
        'Documentation': 'https://ssl-docs.example.com',
        'Source': 'https://github.com/ssl-team/symbolic-security-layer',
        'Tracker': 'https://github.com/ssl-team/symbolic-security-layer/issues',
        'Changelog': 'https://github.com/ssl-team/symbolic-security-layer/blob/main/CHANGELOG.md',
    },
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    include_package_data=True,
    package_data={
        'symbolic_security_layer': [
            'data/*.json',
            'data/*.txt',
            'templates/*.html',
            'templates/*.md',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Security',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Text Processing :: Linguistic',
    ],
    keywords=[
        'ai', 'security', 'symbols', 'unicode', 'machine-learning',
        'natural-language-processing', 'semantic-anchoring', 'tensorflow',
        'pytorch', 'openai', 'huggingface', 'symbolic-ai', 'content-validation'
    ],
    python_requires='>=3.8',
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points={
        'console_scripts': [
            'ssl-validate=symbolic_security_layer.cli:validate_command',
            'ssl-secure=symbolic_security_layer.cli:secure_command',
            'ssl-report=symbolic_security_layer.cli:report_command',
            'ssl-export=symbolic_security_layer.cli:export_command',
        ],
    },
    zip_safe=False,
    test_suite='tests',
    tests_require=EXTRAS_REQUIRE['dev'],
)

