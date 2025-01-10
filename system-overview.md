# AI Developer Worker System Overview

## Core Purpose
An AI-powered CLI tool that helps developers create applications in any programming language (Python, JavaScript, Solidity). Uses OpenRouter for code generation, implements language-specific validation, and supports extensible modules for enhanced functionality.

## Directory Structure
```
ai_developer_worker/
├── README.md
├── pyproject.toml
├── .env.example          # Only OpenRouter configuration
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── cli.py                    # Command-line interface
│   ├── main.py                   # Main application entry point
│   ├── interactive.py            # Interactive session handler
│   ├── worker/
│   │   ├── __init__.py
│   │   ├── code_generator.py     # OpenRouter-based code generation
│   │   ├── analyzer/
│   │   │   ├── __init__.py
│   │   │   ├── python.py         # Python code analysis (pylint)
│   │   │   ├── javascript.js     # JS code analysis (eslint)
│   │   │   └── solidity.py       # Solidity analysis (solhint, slither)
│   │   └── fixer.py             # Code fixing using OpenRouter
│   ├── llm/
│   │   ├── __init__.py
│   │   └── router.py            # OpenRouter integration
│   ├── modules/                  # Extensible module system
│   │   ├── __init__.py
│   │   ├── base_module.py       # Base class for all modules
│   │   ├── serpapi/            # Example module
│   │   │   ├── __init__.py
│   │   │   ├── serpapi.py
│   │   │   └── .env           # Module-specific configuration
│   │   └── marketplace/        # Module management
│   │       ├── __init__.py
│   │       └── registry.py
│   └── utils/
│       ├── __init__.py
│       └── logger.py            # Logging setup
├── tests/
│   ├── __init__.py
│   ├── test_worker/
│   │   ├── test_code_generator.py
│   │   └── test_analyzer/
│   ├── test_llm/
│   └── test_modules/
└── docs/
    └── usage.md
```

## Example Usage Flows

1. **Module Installation**
   ```bash
   $ python -m ai_developer_worker module install serpapi
   Welcome to AI Developer Worker!
   
   Installing SerpAPI module...
   Please enter your SerpAPI key: xxx
   Creating module configuration...
   - Module configuration saved to modules/serpapi/.env
   - Module installed successfully.
   ```

2. **Next.js E-commerce Project (with SerpAPI module)**
   ```bash
   $ python -m ai_developer_worker create
   Welcome to AI Developer Worker!
   
   What would you like to build?
   > An e-commerce site for smartphones using Next.js
   
   [SerpAPI module activates]
   Loading module configuration...
   Analyzing current smartphone market...
   Found top 10 competing sites...
   Identifying common features and patterns...
   
   I'll help you create that. Let me ask a few clarifying questions:
   
   Based on market analysis, these features are common:
   - Product comparison
   - Spec filtering
   - Price alerts
   Would you like to include these? (yes/no)
   > yes
   
   Would you like to include authentication?
   > Yes, with Google and email
   
   What payment system would you prefer?
   > Stripe
   
   Would you like inventory management?
   > Yes
   ```

3. **Smart Contract Development**
   ```bash
   $ python -m ai_developer_worker create
   Welcome to AI Developer Worker!
   
   What would you like to build?
   > An NFT smart contract with royalties
   
   Tell me more about the specific features you need:
   > ERC721 with 5% royalties and whitelist support
   
   I'll help you create that. Let me ask a few clarifying questions:
   
   What's the maximum supply for your NFT collection?
   > 10000
   
   Do you need custom metadata handling?
   > Yes, IPFS integration
   
   Would you like to include a presale mechanism?
   > Yes, with merkle tree whitelist
   
   Should royalties be split among multiple addresses?
   > No, single address is fine
   ```

4. **Generation Process**
   ```
   Thanks! I understand what you need.
   
   [1/4] Creating project structure...
   - Setting up development environment
   - Adding core dependencies
   
   [2/4] Generating code...
   - Core functionality
   - Feature implementation
   - Test cases
   
   [3/4] Running analysis...
   - Code validation
   - Security checks
   - Best practices
   
   [4/4] Creating documentation...
   - Generated README
   - Added setup instructions
   - API documentation
   ```

5. **Final Output**
   ```
   Project generated successfully.
   
   Your project is ready at: ./my-project/
   
   To get started:
   1. cd my-project
   2. [Project-specific setup commands]
   3. [Additional configuration steps]
   
   See README.md for complete setup and usage instructions.
   
   Need anything else? (yes/no)
   > no
   
   Good luck with your project.
   ```

## Configuration
Only core configuration in main .env:
```
OPENROUTER_API_KEY=your-key
```

Each module manages its own configuration independently in its directory.

## Key Features

1. **Language Support**
   - Python (with pylint validation)
   - JavaScript/TypeScript (with eslint)
   - Solidity (with solhint/slither)

2. **Smart Interaction**
   - Natural language understanding
   - Context-aware follow-ups
   - Real-time progress updates

3. **Module System**
   - Self-contained modules with independent configuration
   - Easy module installation
   - Enhanced functionality
   - Real-world data integration

4. **Quality Assurance**
   - Language-specific validation
   - Security best practices
   - Automated fixing

5. **Documentation**
   - Setup instructions
   - Usage guides
   - API documentation
