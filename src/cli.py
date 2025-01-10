# src/cli.py

import os
import typer
import asyncio
import questionary
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from loguru import logger
import json
import re
from datetime import datetime

from src.utils.logger import setup_logging
from src.worker.code_generator import CodeGenerator, GenerationContext
from src.llm.providers.openrouter import OpenRouterProvider
from src.worker.analyzer.python import PythonAnalyzer

app = typer.Typer(
    name="ai_developer_worker",
    help="AI-powered CLI tool for developer assistance",
    add_completion=False,
)


def version_callback(value: bool):
    """Print version information."""
    if value:
        typer.echo("AI Developer Worker v0.1.0")
        raise typer.Exit()


@app.callback()
def main(
        version: Optional[bool] = typer.Option(
            None,
            "--version",
            "-v",
            help="Show version information.",
            callback=version_callback,
            is_eager=True,
        ),
        debug: bool = typer.Option(
            False,
            "--debug",
            "-d",
            help="Enable debug logging.",
        ),
):
    """AI Developer Worker - Your AI-powered development assistant."""
    setup_logging(debug)


async def discuss_project_requirements(llm_provider: OpenRouterProvider, initial_description: str) -> Tuple[
    str, List[str], Dict[str, any]]:
    """Have an interactive conversation about project requirements."""
    # Get event loop
    loop = asyncio.get_event_loop()

    # Initial analysis prompt
    analysis_prompt = f"""You are an experienced software architect helping to plan a new project. 
    The user wants to build: {initial_description}

    Based on this description, ask questions to determine:
    1. User Experience & Interface Needs
       - Visual design requirements
       - Interactivity needs
       - Client-side functionality

    2. Content & Data Requirements
       - Static vs dynamic content
       - Data storage needs
       - Update frequency

    3. Technical Requirements
       - Server-side processing needs
       - Database requirements
       - Integration points

    4. Deployment & Maintenance
       - Hosting requirements
       - Content update workflow
       - Maintenance needs

    Return exactly 3-5 most important questions in this format:
    QUESTION: [your question here]"""

    # Get clarifying questions
    response = await llm_provider.analyze_request(analysis_prompt)

    # More robust question parsing
    questions = []
    for line in response.split('\n'):
        line = line.strip()
        if line.startswith('QUESTION:'):
            question = line.replace('QUESTION:', '').strip()
            if question and len(question) > 10:  # Basic validation
                questions.append(question)

    # Ensure we have questions
    if not questions:
        logger.warning("No valid questions found in response, using defaults")
        questions = [
            "What types of calculations need to be supported?",
            "Is this intended to be a web-based calculator or command-line tool?",
            "Do you need to maintain calculation history?"
        ]

    # Ask each question synchronously
    answers = {}
    for question in questions:
        try:
            answer = await loop.run_in_executor(
                None,
                lambda q=question: questionary.text(q).ask()
            )
            if answer is None:  # Handle interrupts
                logger.info("User interrupted question flow")
                raise typer.Exit(code=1)
            answers[question] = answer
        except Exception as e:
            logger.error(f"Error during question flow: {e}")
            raise typer.Exit(code=1)

    # Generate project recommendation
    recommendation_prompt = f"""As an experienced software architect, analyze this project request and recommend the most suitable technical approach.

    Project Description: {initial_description}

    User's Answers:
    {chr(10).join(f'Q: {q}{chr(10)}A: {a}' for q, a in answers.items())}

    First, determine the PRIMARY nature of this project:
    1. Is this primarily a FRONTEND project? Consider:
       - Is it mainly about displaying content?
       - Does it focus on user interface and design?
       - Are interactions client-side focused?
       - Could this be a static site with minimal backend?

    2. Only if needed, consider BACKEND requirements:
       - Is there complex server-side logic?
       - Is there a need for a database?
       - Are there heavy data processing needs?
       - Are there complex security requirements?

    IMPORTANT PRINCIPLES:
    1. Start with the simplest solution
    2. Static is better than dynamic when possible
    3. Client-side is preferable for simple interactions
    4. Only add backend complexity if absolutely required
    5. Consider future maintenance and scalability

    For a website/webapp, prefer this order of consideration:
    1. Static site (Next.js)
    2. Single page application (React)
    3. Server-side rendering (Only if specifically needed)

    Based on this analysis, provide your recommendation in this exact format:
    PROJECT_TYPE: [one of: nextjs, react, python-cli, flask, django, smart-contract, hardhat]
    LANGUAGE: [primary programming language]
    FEATURES: [comma-separated list of specific features]
    REQUIREMENTS: [key:value pairs for additional requirements]
    REASONING: [detailed explanation of why this is the simplest appropriate solution]"""

    response = await llm_provider.analyze_request(recommendation_prompt)

    # Parse recommendation
    lines = response.split("\n")
    project_type = ""
    language = ""
    features = []
    requirements = {}

    for line in lines:
        if line.startswith("PROJECT_TYPE:"):
            project_type = line.replace("PROJECT_TYPE:", "").strip()
        elif line.startswith("LANGUAGE:"):
            language = line.replace("LANGUAGE:", "").strip()
        elif line.startswith("FEATURES:"):
            features = [f.strip() for f in line.replace("FEATURES:", "").strip().split(",")]
        elif line.startswith("REQUIREMENTS:"):
            req_str = line.replace("REQUIREMENTS:", "").strip()
            for pair in req_str.split(","):
                if ":" in pair:
                    k, v = pair.split(":")
                    requirements[k.strip()] = v.strip()
        elif line.startswith("REASONING:"):
            reasoning = line.replace("REASONING:", "").strip()
            typer.echo(f"\nRecommended Approach:\n{reasoning}")

    # Confirm with user
    typer.echo(f"\nBased on our discussion, I recommend:")
    typer.echo(f"- Project Type: {project_type}")
    typer.echo(f"- Primary Language: {language}")
    typer.echo("- Features:")
    for feature in features:
        typer.echo(f"  - {feature}")
    if requirements:
        typer.echo("- Technical Requirements:")
        for k, v in requirements.items():
            typer.echo(f"  - {k}: {v}")

    # Run confirmation in executor
    confirmed = await loop.run_in_executor(
        None,
        lambda: questionary.confirm("Would you like to proceed with this setup?").ask()
    )

    if confirmed:
        return project_type, features, requirements

    # If user doesn't confirm, fall back to manual selection
    return await loop.run_in_executor(None, lambda: get_project_details(None))

def _setup_logging_dir() -> Path:
    """Setup logging directory and return its path."""
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)
    return log_dir

async def async_create(
    output: str,
    description: Optional[str] = None,
    project_type: Optional[str] = None,
    features: Optional[List[str]] = None,
    language: Optional[str] = None,
    interactive: bool = True,
) -> None:
    """Create a new project asynchronously."""
    try:
        # Setup logging
        log_dir = _setup_logging_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create interaction log file
        interaction_log = log_dir / f"create_interaction_{timestamp}.txt"
        
        # Log initial request
        log_content = [
            "=== Project Creation Request ===",
            f"Output: {output}",
            f"Description: {description}",
            f"Project Type: {project_type}",
            f"Features: {features}",
            f"Language: {language}",
            f"Interactive: {interactive}",
            "\n=== AI Interactions ===\n"
        ]
        
        interaction_log.write_text("\n".join(log_content))
        
        # Initialize OpenRouter provider
        provider = OpenRouterProvider()

        feature_list = []
        requirements = {}

        if interactive:
            # Create event loop for sync operations
            loop = asyncio.get_event_loop()
            
            # If no description or project type provided, ask what they want to build
            if not description and not project_type:
                description = await loop.run_in_executor(
                    None,
                    lambda: questionary.text(
                        "What would you like to build? Please describe your project:"
                    ).ask()
                )

            if description:
                # Have a conversation about the project
                project_type, feature_list, requirements = await discuss_project_requirements(provider, description)
            else:
                # Use manual selection
                project_type_selected, feature_list, requirements = await loop.run_in_executor(
                    None,
                    lambda: get_project_details(project_type)
                )
                project_type = project_type_selected
        else:
            # Non-interactive mode requires project type
            if not project_type:
                typer.echo("Error: --type is required when not in interactive mode")
                raise typer.Exit(code=1)
            feature_list = features.split(",") if features else []

        # Create code generator
        generator = CodeGenerator(provider)

        # Create output directory path
        output_path = Path(output).absolute()

        # Create generation context
        context = GenerationContext(
            project_type=project_type,
            language=language or _detect_language(project_type),
            features=feature_list,
            requirements=requirements,
            output_dir=output_path
        )

        # Run generation
        logger.info(f"Creating new {project_type} project in {output_path}")
        result = await generator.generate_project(context)

        # Even if analysis shows issues, consider the project created successfully
        if result is not None:  # Change this condition
            typer.echo(f"\nProject created successfully in {output_path}")
            typer.echo("\nNext steps:")
            typer.echo(f"1. cd {output_path}")
            typer.echo("2. Follow the setup instructions in README.md")
            
            # If there were analysis warnings, show them but don't fail
            if hasattr(result, 'score') and result.score < 7.0:
                typer.echo("\nNote: Some quality issues were detected. Run 'ai-dev analyze' for details.")
            
            return  # Successfully exit
        else:
            typer.echo("Project creation failed. Check the logs for details.")
            raise typer.Exit(code=1)

    except Exception as e:
        logger.exception(f"Error creating project: {str(e)}")
        typer.echo(f"Error: {str(e)}")
        raise typer.Exit(code=1)


def get_project_details(project_type: Optional[str] = None) -> Tuple[str, List[str], Dict[str, any]]:
    """Get project details through manual selection."""
    project_features = {
        "python-cli": [
            "config-file",
            "logging",
            "async-support",
            "database",
            "api-client",
            "cli-colors",
        ],
        "flask": [
            "authentication",
            "database",
            "api",
            "admin-panel",
            "user-management",
            "file-upload",
        ],
        "django": [
            "authentication",
            "database",
            "api",
            "admin-panel",
            "user-management",
            "file-upload",
        ],
        "nextjs": [
            "authentication",
            "api-routes",
            "database",
            "styling-solution",
            "state-management",
            "testing",
        ],
        "react": [
            "authentication",
            "routing",
            "state-management",
            "api-integration",
            "testing",
            "styling",
        ],
        "smart-contract": [
            "erc20",
            "erc721",
            "upgradeable",
            "access-control",
            "multisig",
        ],
        "hardhat": [
            "erc20",
            "erc721",
            "upgradeable",
            "access-control",
            "deployment-scripts",
        ],
    }

    if not project_type:
        project_type = questionary.select(
            "What type of project would you like to create?",
            choices=list(project_features.keys())
        ).ask()

    features = questionary.checkbox(
        "Select features to include:",
        choices=project_features.get(project_type, [])
    ).ask()

    requirements = {}
    if project_type == "python-cli":
        requirements["cli_framework"] = questionary.select(
            "Choose CLI framework:",
            choices=["typer", "click", "argparse"]
        ).ask()
    elif project_type in ["flask", "django"]:
        requirements["database"] = questionary.select(
            "Choose database:",
            choices=["postgresql", "mysql", "sqlite"]
        ).ask()
    elif project_type in ["nextjs", "react"]:
        requirements["styling"] = questionary.select(
            "Choose styling solution:",
            choices=["tailwind", "styled-components", "css-modules"]
        ).ask()

    return project_type, features, requirements


@app.command()
def create(
    output: str = typer.Option(
        "./output",
        "--output",
        "-o",
        help="Output directory for the project",
    ),
    description: str = typer.Option(
        None,
        "--desc",
        "-d",
        help="Description of what you want to build",
    ),
    project_type: str = typer.Option(
        None,
        "--type",
        "-t",
        help="Type of project to create",
    ),
    features: str = typer.Option(
        None,
        "--features",
        "-f",
        help="Comma-separated list of features to include",
    ),
    language: str = typer.Option(
        None,
        "--language",
        "-l",
        help="Programming language to use",
    ),
    interactive: bool = typer.Option(
        True,
        "--no-interactive",
        help="Disable interactive prompts",
        is_flag=True,
    ),
):
    """Create a new project with AI assistance."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(async_create(
            output=output,
            description=description,
            project_type=project_type,
            features=features,
            language=language,
            interactive=interactive
        ))
        loop.close()
    except Exception as e:
        logger.exception(f"Error creating project: {str(e)}")
        typer.echo(f"Error: {str(e)}")
        raise typer.Exit(code=1)


@app.command()
def module(
        action: str = typer.Argument(..., help="Action to perform (install/uninstall/list)"),
        name: Optional[str] = typer.Argument(None, help="Name of the module"),
):
    """Manage AI Developer Worker modules."""
    logger.info(f"Module action: {action} {name or ''}")
    # TODO: Implement module management logic


def _detect_language(project_type: str) -> str:
    """Detect default language based on project type."""
    type_to_language = {
        "nextjs": "typescript",
        "react": "typescript",
        "python-cli": "python",
        "flask": "python",
        "django": "python",
        "smart-contract": "solidity",
        "hardhat": "solidity",
    }
    return type_to_language.get(project_type, "typescript")


@app.command()
def analyze(
        path: str = typer.Argument(
            ...,
            help="Path to the project to analyze"
        ),
        fix: bool = typer.Option(
            False,
            "--fix",
            "-f",
            help="Automatically fix issues found"
        )
):
    """Analyze a project for issues and potential improvements."""
    try:
        project_dir = Path(path).absolute()
        if not project_dir.exists():
            typer.echo(f"Error: Directory {project_dir} does not exist")
            raise typer.Exit(1)

        # Initialize OpenRouter provider
        provider = OpenRouterProvider()

        # Detect project type and language
        analyzer = None
        language = None

        # Check for Python project
        if (project_dir / "setup.py").exists() or (project_dir / "requirements.txt").exists():
            language = "python"
            from src.worker.analyzer.python import PythonAnalyzer
            analyzer = PythonAnalyzer(provider)

        # Check for JavaScript/TypeScript project
        elif (project_dir / "package.json").exists():
            # Check if it's a Hardhat/Solidity project first
            with open(project_dir / "package.json") as f:
                pkg_data = json.load(f)
                if any(dep.startswith('hardhat') for dep in pkg_data.get("devDependencies", {}).keys()):
                    language = "solidity"
                    from src.worker.analyzer.solidity import SolidityAnalyzer
                    analyzer = SolidityAnalyzer(provider)
                else:
                    # Check for TypeScript configuration
                    if (project_dir / "tsconfig.json").exists() or any(project_dir.glob("**/*.ts")):
                        language = "typescript"
                        from src.worker.analyzer.typescript import TypeScriptAnalyzer
                        analyzer = TypeScriptAnalyzer(provider)
                    else:
                        language = "javascript"
                        from src.worker.analyzer.javascript import JavaScriptAnalyzer
                        analyzer = JavaScriptAnalyzer(provider)

        # Check for Solidity project (without package.json)
        elif any(project_dir.glob("**/*.sol")):
            language = "solidity"
            from src.worker.analyzer.solidity import SolidityAnalyzer
            analyzer = SolidityAnalyzer(provider)

        # Check for style files
        elif any(project_dir.glob("**/*.css")) or any(project_dir.glob("**/*.scss")) or any(project_dir.glob("**/*.sass")):
            language = "style"
            from src.worker.analyzer.style import StyleAnalyzer
            analyzer = StyleAnalyzer(provider)

        # Check for Rust project
        elif (project_dir / "Cargo.toml").exists():
            language = "rust"
            from src.worker.analyzer.rust import RustAnalyzer
            analyzer = RustAnalyzer(provider)

        if not analyzer:
            typer.echo("Error: Could not determine project type")
            raise typer.Exit(1)

        # Run analysis
        logger.info(f"Analyzing {language} project in {project_dir}")

        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Run analysis
        analysis_result = loop.run_until_complete(analyzer.analyze_project(project_dir))

        # Display results
        typer.echo(f"\nAnalysis Results:")
        typer.echo(f"Score: {analysis_result.score:.1f}/10.0")
        typer.echo(f"Files analyzed: {len(analysis_result.files_analyzed)}")
        typer.echo(f"Issues found: {len(analysis_result.issues)}")

        # Display language-specific results
        if language == "solidity":
            typer.echo(f"Gas optimization issues: {len(analysis_result.gas_issues)}")
            typer.echo(f"Security issues: {len(analysis_result.security_issues)}")
        elif language == "style":
            typer.echo(f"Specificity issues: {len(analysis_result.specificity_issues)}")
            typer.echo(f"Performance issues: {len(analysis_result.performance_issues)}")
            typer.echo(f"Unused styles: {len(analysis_result.unused_styles)}")
        elif language == "typescript":
            typer.echo(f"Type coverage: {analysis_result.type_coverage:.1f}%")
            typer.echo(f"Type issues: {len(analysis_result.type_issues)}")
            typer.echo(f"Interface issues: {len(analysis_result.interface_issues)}")
        elif language == "javascript":
            if hasattr(analysis_result, 'type_issues'):
                typer.echo(f"Type issues: {len(analysis_result.type_issues)}")
        elif language == "rust":
            typer.echo(f"Unsafe blocks: {len(analysis_result.unsafe_blocks)}")
            typer.echo(f"Lifetime issues: {len(analysis_result.lifetime_issues)}")
            typer.echo(f"Clippy warnings: {len(analysis_result.clippy_warnings)}")

        if analysis_result.suggestions:
            typer.echo("\nSuggestions:")
            for suggestion in analysis_result.suggestions:
                typer.echo(f"- {suggestion}")

        # Display dependencies info if available
        if hasattr(analysis_result, 'dependencies'):
            typer.echo("\nDependencies:")
            if isinstance(analysis_result.dependencies, set):
                typer.echo(f"Total dependencies: {len(analysis_result.dependencies)}")
            elif isinstance(analysis_result.dependencies, dict):
                typer.echo(f"Total dependencies: {len(analysis_result.dependencies.get('present', []))}")
                if analysis_result.dependencies.get('missing'):
                    typer.echo("Missing dependencies:")
                    for dep in analysis_result.dependencies['missing']:
                        typer.echo(f"- {dep}")

        # Fix issues if requested
        if fix and analysis_result.issues:
            typer.echo("\nAttempting to fix issues...")
            fixed = loop.run_until_complete(analyzer.fix_issues(project_dir, analysis_result.issues))
            if fixed:
                typer.echo("Successfully fixed some issues")
            else:
                typer.echo("No issues could be fixed automatically")

        loop.close()

    except Exception as e:
        logger.exception(f"Error analyzing project: {str(e)}")
        raise typer.Exit(code=1)


@app.command()
def fix(
    path: str = typer.Argument(
        ...,
        help="Path to the project to fix"
    )
):
    """Fix common issues in the project automatically."""
    try:
        project_dir = Path(path).absolute()
        if not project_dir.exists():
            typer.echo(f"Error: Directory {project_dir} does not exist")
            raise typer.Exit(1)

        # Initialize OpenRouter provider
        provider = OpenRouterProvider()

        # Detect project type and language
        analyzer = None
        language = None

        # Check for Python project
        if (project_dir / "setup.py").exists() or (project_dir / "requirements.txt").exists():
            language = "python"
            from src.worker.analyzer.python import PythonAnalyzer
            analyzer = PythonAnalyzer(provider)

        # Check for JavaScript/TypeScript project
        elif (project_dir / "package.json").exists():
            # Check if it's a Hardhat/Solidity project first
            with open(project_dir / "package.json") as f:
                pkg_data = json.load(f)
                if any(dep.startswith('hardhat') for dep in pkg_data.get("devDependencies", {}).keys()):
                    language = "solidity"
                    from src.worker.analyzer.solidity import SolidityAnalyzer
                    analyzer = SolidityAnalyzer(provider)
                else:
                    # Check for TypeScript configuration
                    if (project_dir / "tsconfig.json").exists() or any(project_dir.glob("**/*.ts")):
                        language = "typescript"
                        from src.worker.analyzer.typescript import TypeScriptAnalyzer
                        analyzer = TypeScriptAnalyzer(provider)
                    else:
                        language = "javascript"
                        from src.worker.analyzer.javascript import JavaScriptAnalyzer
                        analyzer = JavaScriptAnalyzer(provider)

        # Check for Solidity project (without package.json)
        elif any(project_dir.glob("**/*.sol")):
            language = "solidity"
            from src.worker.analyzer.solidity import SolidityAnalyzer
            analyzer = SolidityAnalyzer(provider)

        # Check for style files
        elif any(project_dir.glob("**/*.css")) or any(project_dir.glob("**/*.scss")):
            language = "style"
            from src.worker.analyzer.style import StyleAnalyzer
            analyzer = StyleAnalyzer(provider)

        # Check for Rust project
        elif (project_dir / "Cargo.toml").exists():
            language = "rust"
            from src.worker.analyzer.rust import RustAnalyzer
            analyzer = RustAnalyzer(provider)

        if not analyzer:
            typer.echo("Error: Could not determine project type")
            raise typer.Exit(1)

        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Run analysis first to get issues
        typer.echo(f"Analyzing {language} project...")
        analysis_result = loop.run_until_complete(analyzer.analyze_project(project_dir))

        if not analysis_result.issues:
            typer.echo("No issues found to fix")
            return

        # Fix issues
        typer.echo("Fixing issues...")
        fixed = loop.run_until_complete(analyzer.fix_issues(project_dir, analysis_result.issues))

        if fixed:
            typer.echo("Successfully fixed some issues")
        else:
            typer.echo("No issues could be fixed automatically")

        loop.close()

    except Exception as e:
        logger.exception(f"Error fixing project: {str(e)}")
        raise typer.Exit(1)

def handle_fix(args):
    project_dir = get_project_directory(args.project_dir)
    logger.info(f"Analyzing python project in {project_dir}")
    
    # Get the issues using pylint
    issues = get_project_issues(project_dir)
    
    # Instead of just printing issues, we should:
    # 1. Generate the fix using the AI
    # 2. Apply the changes to the files
    
    if issues:
        # Create the prompt for the AI including the issues
        prompt = f"{args.description}\n\nHere are the current issues:\n{json.dumps(issues, indent=2)}"
        
        # Get the AI's response
        response = get_ai_response(prompt, project_dir)
        
        # Parse and apply the changes from the AI response
        apply_changes(response, project_dir)
        
        logger.info("Applied fixes to the project")
    else:
        logger.info("No issues found in the project")

def apply_changes(ai_response, project_dir):
    """Parse the AI response and apply the changes to the files"""
    # Here we need to parse the AI's response and apply the changes
    # This is a basic implementation - you might need to enhance it
    try:
        changes = parse_ai_response(ai_response)
        for file_path, new_content in changes.items():
            full_path = os.path.join(project_dir, file_path)
            with open(full_path, 'w') as f:
                f.write(new_content)
            logger.info(f"Updated {file_path}")
    except Exception as e:
        logger.error(f"Failed to apply changes: {str(e)}")

def parse_ai_response(response):
    """Parse the AI response to extract file changes
    Returns a dict of {file_path: new_content}"""
    changes = {}
    # Implement the parsing logic based on your AI's response format
    # This is where you'd extract the file paths and their new contents
    return changes

def get_ai_response(prompt, project_dir):
    """Get the AI's response for fixing the code"""
    # Implement your AI interaction here
    # This should return the AI's suggested fixes
    pass

def _parse_error_message(error_msg: str) -> dict:
    """Parse error message to extract key information"""
    error_info = {
        'type': None,
        'file': None,
        'line': None,
        'message': None,
        'traceback': None
    }
    
    # Try to extract information from traceback
    if 'Traceback' in error_msg:
        lines = error_msg.split('\n')
        for line in lines:
            if 'File "' in line:
                error_info['file'] = line.split('File "')[1].split('"')[0]
                if ', line ' in line:
                    error_info['line'] = line.split(', line ')[1].split(',')[0]
            elif 'Error:' in line:
                error_info['type'] = line.split('Error:')[0].strip()
                error_info['message'] = line.split('Error:')[1].strip()
                
    return error_info

def _parse_file_changes(response: str) -> Dict[str, str]:
    """Parse the AI response to extract file changes with complete content handling."""
    # Log raw response for debugging
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_log_path = log_dir / f"raw_response_{timestamp}.txt"
    raw_log_path.write_text(response)
    
    file_changes = {}
    
    # First, split by FILE: markers
    file_sections = response.split("\nFILE: ")
    
    for section in file_sections[1:]:  # Skip the first empty section
        # Split into filename and content
        try:
            # Extract filename - it's the first line
            filename = section.split('\n')[0].strip()
            
            # Get everything after the filename
            content = '\n'.join(section.split('\n')[1:])
            
            if filename.endswith('.md'):
                # For markdown files, find content between ```markdown and ``` tags
                pattern = r'```markdown\n(.*?)\n```'
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    clean_content = match.group(1).strip()
                    # Remove any trailing END_FILE marker
                    clean_content = clean_content.split('END_FILE')[0].strip()
                    if clean_content:
                        file_changes[filename] = clean_content
            else:
                # For code files, find content between ``` and ``` tags
                pattern = r'```(?:\w+)?\n(.*?)\n```'
                matches = re.findall(pattern, content, re.DOTALL)
                if matches:
                    clean_content = '\n'.join(matches).strip()
                    # Remove any trailing END_FILE marker
                    clean_content = clean_content.split('END_FILE')[0].strip()
                    if clean_content:
                        file_changes[filename] = clean_content
        except Exception as e:
            logger.error(f"Error parsing section for {filename if 'filename' in locals() else 'unknown'}: {e}")
            continue
    
    # Log the processed changes
    parsed_log_path = log_dir / f"parsed_changes_{timestamp}.txt"
    log_content = ["=== Processed Files ==="]
    for file_path, content in file_changes.items():
        log_content.extend([
            f"\nFILE: {file_path}",
            "CONTENT:",
            content,
            "-" * 80
        ])
    parsed_log_path.write_text('\n'.join(log_content))
    
    return file_changes

def _parse_markdown_content(content: str) -> str:
    """Clean up and validate markdown content."""
    sections = {}
    current_section = None
    current_content = []
    
    for line in content.split('\n'):
        # Detect section headers
        if line.startswith('#'):
            if current_section and current_content:
                sections[current_section] = '\n'.join(current_content).strip()
            current_section = line.strip('# ')
            current_content = []
        elif current_section:
            current_content.append(line)
    
    # Add last section
    if current_section and current_content:
        sections[current_section] = '\n'.join(current_content).strip()
    
    # Validate required sections
    required_sections = [
        'Installation', 'Features', 'Configuration', 
        'Usage', 'Project Structure'
    ]
    
    for section in required_sections:
        if section not in sections:
            sections[section] = "Section needs to be added"
    
    # Rebuild markdown with validated content
    markdown = []
    for section, content in sections.items():
        markdown.append(f"## {section}\n\n{content}\n")
    
    return '\n'.join(markdown)