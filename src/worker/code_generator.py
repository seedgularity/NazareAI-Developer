# src/worker/code_generator.py

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import re
import asyncio
import logging
from datetime import datetime
import shutil

from ..llm.providers.base_provider import BaseProvider
from ..utils.logger import get_logger
from .analyzer.base_analyzer import BaseAnalysisResult
from .analyzer.typescript import TypeScriptAnalyzer

logger = get_logger(__name__)


@dataclass
class GenerationContext:
    """Context for code generation including project details and requirements"""
    project_type: str  # e.g., "nextjs", "python-cli", "smart-contract"
    language: str = field(default="typescript")  # Default to typescript for web projects
    features: List[str] = field(default_factory=list)
    requirements: Dict[str, any] = field(default_factory=dict)
    output_dir: Path = field(default_factory=lambda: Path("./output"))

    def __post_init__(self):
        """Validate and set proper defaults after initialization"""
        # Map project types to default languages if none specified
        project_language_map = {
            "nextjs": "typescript",
            "react": "typescript",
            "python-cli": "python",
            "flask": "python",
            "django": "python",
            "smart-contract": "solidity",
            "hardhat": "solidity",
            "ios": "swift",
            "android": "kotlin",
        }

        if not self.language and self.project_type in project_language_map:
            self.language = project_language_map[self.project_type]
        elif not self.language:
            self.language = "typescript"  # Default fallback


class CodeGenerator:
    """
    Core worker component responsible for code generation using LLM.
    Coordinates between LLM provider and language-specific analyzers.
    """

    def __init__(self, llm_provider: BaseProvider):
        self.llm = llm_provider
        self.logger = logger
        self.log_dir = Path("./logs")
        self.log_dir.mkdir(exist_ok=True)

    def _parse_file_content(self, response: str) -> List[Tuple[str, str]]:
        """Parse the LLM response to extract file paths and their content."""
        files = []
        directories = set()  # Track all directories
        
        # Log raw response
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        (self.log_dir / f"raw_response_{timestamp}.txt").write_text(response)

        # First, extract all directory paths mentioned in the response
        dir_matches = re.finditer(r'(?:^|\n)(?:directory:|dir:|folder:)\s*([^\n]+)', response, re.IGNORECASE)
        for match in dir_matches:
            dir_path = match.group(1).strip()
            if dir_path:
                directories.add(dir_path)

        # Split response into sections by FILE: markers
        sections = response.split("\nFILE: ")
        
        for section in sections[1:]:  # Skip first empty part
            try:
                # Get filename from first line
                lines = section.split('\n')
                filename = lines[0].strip()
                
                # Add the directory of this file to the set
                if '/' in filename:
                    directories.add(str(Path(filename).parent))
                
                # Extract content between markers
                content = section[section.find("###CONTENT_START###"):section.find("###CONTENT_END###")]
                if content:
                    # Remove the start marker
                    content = content.replace("###CONTENT_START###", "").strip()
                    
                    # Clean up the content based on file type
                    if filename.endswith('.md'):
                        clean_content = content
                    else:
                        clean_content = content.strip()
                    
                    if clean_content:
                        files.append((filename, clean_content))
            
            except Exception as e:
                self.logger.error(f"Error parsing file content for {filename if 'filename' in locals() else 'unknown'}: {e}")
                continue
        
        # Log parsed files and directories
        log_content = ["=== Parsed Files and Directories ==="]
        log_content.append("\nDirectories to create:")
        for directory in sorted(directories):
            log_content.append(f"- {directory}")
            # Add directory to files list with empty string content
            files.append((directory, ""))  # Use empty string instead of None
        
        log_content.append("\nFiles to create:")
        for filename, content in files:
            if content != "":  # Skip directories in the log
                log_content.extend([
                    f"\nFILE: {filename}",
                    "CONTENT:",
                    content,
                    "-" * 80
                ])
        
        (self.log_dir / f"parsed_files_{timestamp}.txt").write_text('\n'.join(log_content))

        return files

    def _clean_filepath(self, path: str) -> str:
        """Clean and validate file path."""
        # Remove any markdown, quotes or unwanted characters
        path = re.sub(r'["\'\`]', '', path.strip())
        # Ensure valid path format
        if self._is_valid_filename(path):
            return path
        return ""

    def _clean_content(self, content: str) -> str:
        """Clean up file content while preserving formatting."""
        # Remove any trailing END_FILE markers
        content = content.split('END_FILE')[0]
        
        # For non-markdown files, clean up whitespace
        if not content.startswith('# '):
            content = content.strip()
        
        return content

    def _is_valid_filename(self, filename: str) -> bool:
        """Check if a filename is valid."""
        return bool(re.match(r'^[\w\-./]+\.[a-zA-Z0-9]+$', filename))

    async def _write_file(self, filepath: Path, content: str) -> None:
        """Write content to a file or create directory."""
        try:
            # If content is empty string, this is a directory entry
            if content == "":
                filepath.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Created directory: {filepath}")
                return

            # Ensure the parent directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Clean up the content
            content = content.replace('\r\n', '\n')  # Normalize line endings
            content = re.sub(r'^```\w*\n|```$', '', content.strip())  # Remove markdown code blocks

            # Write the file with proper encoding
            filepath.write_text(content, encoding='utf-8')
            self.logger.debug(f"Successfully wrote file: {filepath}")

        except Exception as e:
            self.logger.error(f"Error writing file/directory {filepath}: {str(e)}")
            raise

    async def _analyze_and_fix_code(self, context: GenerationContext) -> None:
        """Analyze generated code and fix issues."""
        try:
            # Create appropriate analyzer based on language
            analyzer = None
            
            if context.language == "python":
                from .analyzer.python import PythonAnalyzer
                analyzer = PythonAnalyzer(self.llm)
            elif context.language == "typescript":
                from .analyzer.typescript import TypeScriptAnalyzer
                analyzer = TypeScriptAnalyzer(self.llm)
            elif context.language == "javascript":
                from .analyzer.javascript import JavaScriptAnalyzer
                analyzer = JavaScriptAnalyzer(self.llm)
            elif context.language == "solidity":
                from .analyzer.solidity import SolidityAnalyzer
                analyzer = SolidityAnalyzer(self.llm)
            elif context.language == "style":
                from .analyzer.style import StyleAnalyzer
                analyzer = StyleAnalyzer(self.llm)

            if not analyzer:
                self.logger.warning(f"No analyzer available for {context.language}")
                return

            # Run analysis
            self.logger.info(f"Analyzing generated {context.language} code...")
            try:
                analysis_result = await analyzer.analyze_project(context.output_dir)
            except Exception as e:
                self.logger.error(f"Error during code analysis: {e}")
                return

            # Log results
            self.logger.info(f"Analysis complete. Score: {analysis_result.score:.1f}")
            if analysis_result.suggestions:
                self.logger.info("Suggestions for improvement:")
                for suggestion in analysis_result.suggestions:
                    self.logger.info(f"- {suggestion}")

            # Fix critical issues
            try:
                critical_issues = [
                    issue for issue in analysis_result.issues
                    if issue.get('severity') in ['high', 'medium']
                ]
                if critical_issues:
                    self.logger.info(f"Attempting to fix {len(critical_issues)} critical issues...")
                    if await analyzer.fix_issues(context.output_dir, critical_issues):
                        self.logger.info("Fixed critical issues")
            except Exception as e:
                self.logger.error(f"Error fixing issues: {e}")

            # Handle dependencies
            try:
                if hasattr(analysis_result, 'dependencies'):
                    if isinstance(analysis_result.dependencies, dict):
                        missing_deps = analysis_result.dependencies.get('missing', [])
                        if missing_deps:
                            self.logger.warning(f"Missing dependencies: {', '.join(missing_deps)}")
                            await self._add_missing_dependencies(context, missing_deps)
            except Exception as e:
                self.logger.error(f"Error handling dependencies: {e}")

        except Exception as e:
            self.logger.error(f"Error during code analysis: {e}")
            # Don't raise the exception, just log it

    async def generate_project(self, context: GenerationContext) -> bool:
        """Generate a complete project based on the context."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            generation_log = self.log_dir / f"generation_{timestamp}.txt"
            
            # Log generation start
            log_content = [
                "=== Project Generation Started ===",
                f"Project Type: {context.project_type}",
                f"Language: {context.language}",
                f"Features: {context.features}",
                f"Requirements: {context.requirements}",
                "\n=== Generation Steps ===\n"
            ]
            
            # Get structure prompt
            structure_prompt = self._build_structure_prompt(context)
            log_content.append("Structure Prompt:")
            log_content.append(structure_prompt)
            
            # Get LLM response for structure
            structure_response = await self.llm.generate_code(
                structure_prompt,
                context.language
            )
            log_content.append("\nStructure Response:")
            log_content.append(structure_response)

            # Parse and create files - do this only once
            files = self._parse_file_content(structure_response)
            log_content.append("\nParsed Files:")
            for path, content in files:
                log_content.append(f"\nFILE: {path}")
                log_content.append("CONTENT:")
                log_content.append(content)
                log_content.append("-" * 80)
            
            # Write the log
            generation_log.write_text("\n".join(log_content))
            
            # Create output directory if it doesn't exist
            context.output_dir.mkdir(parents=True, exist_ok=True)

            # Write all files at once - remove the separate generation steps
            for filepath, content in files:
                await self._write_file(context.output_dir / filepath, content)

            # Run analysis without modifying existing files
            analysis_result = await self._analyze_code(context)
            
            # Only fix critical issues and create backups before modifying
            if analysis_result and analysis_result.issues:
                critical_issues = [
                    issue for issue in analysis_result.issues
                    if issue.get('severity') in ['high', 'critical']
                ]
                
                if critical_issues:
                    self.logger.info(f"Found {len(critical_issues)} critical issues to fix")
                    
                    # Create backups before fixing
                    for file_path in set(issue.get('file') for issue in critical_issues if issue.get('file')):
                        full_path = context.output_dir / file_path
                        if full_path.exists():
                            backup_path = full_path.with_suffix(full_path.suffix + '.bak')
                            self.logger.info(f"Creating backup: {backup_path}")
                            import shutil
                            shutil.copy2(full_path, backup_path)
                    
                    # Now apply fixes
                    await self._fix_critical_issues(context, critical_issues)

            self.logger.info("Project generation completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error during project generation: {e}")
            return False

    async def _analyze_code(self, context: GenerationContext) -> Optional[BaseAnalysisResult]:
        """Analyze code without modifying files"""
        try:
            # Create appropriate analyzer based on language
            analyzer = None
            
            if context.language == "python":
                from .analyzer.python import PythonAnalyzer
                analyzer = PythonAnalyzer(self.llm)
            elif context.language == "typescript":
                from .analyzer.typescript import TypeScriptAnalyzer
                analyzer = TypeScriptAnalyzer(self.llm)
            elif context.language == "javascript":
                from .analyzer.javascript import JavaScriptAnalyzer
                analyzer = JavaScriptAnalyzer(self.llm)
            elif context.language == "solidity":
                from .analyzer.solidity import SolidityAnalyzer
                analyzer = SolidityAnalyzer(self.llm)
            elif context.language == "style":
                from .analyzer.style import StyleAnalyzer
                analyzer = StyleAnalyzer(self.llm)

            if not analyzer:
                self.logger.warning(f"No analyzer available for {context.language}")
                return None

            # Run analysis
            self.logger.info(f"Analyzing generated {context.language} code...")
            return await analyzer.analyze_project(context.output_dir)

        except Exception as e:
            self.logger.error(f"Error during code analysis: {e}")
            return None

    async def _fix_critical_issues(self, context: GenerationContext, issues: List[Dict]) -> None:
        """Fix only critical issues with backups"""
        try:
            # Create appropriate analyzer
            analyzer = None
            if context.language == "python":
                from .analyzer.python import PythonAnalyzer
                analyzer = PythonAnalyzer(self.llm)
            elif context.language == "typescript":
                from .analyzer.typescript import TypeScriptAnalyzer
                analyzer = TypeScriptAnalyzer(self.llm)
            # ... add other language analyzers as needed

            if analyzer:
                await analyzer.fix_issues(context.output_dir, issues)
            else:
                self.logger.warning(f"No analyzer available to fix {context.language} issues")

        except Exception as e:
            self.logger.error(f"Error fixing critical issues: {e}")

    async def _create_project_structure(self, context: GenerationContext) -> List[Tuple[str, str]]:
        """Creates the basic project structure and essential files"""
        prompt = self._build_structure_prompt(context)
        response = await self.llm.generate_code(prompt, context.language)
        return self._parse_file_content(response)

    async def _generate_core_files(self, context: GenerationContext) -> List[Tuple[str, str]]:
        """Generates the core application files"""
        prompt = self._build_core_files_prompt(context)
        response = await self.llm.generate_code(prompt, context.language)
        return self._parse_file_content(response)

    async def _implement_features(self, context: GenerationContext) -> List[Tuple[str, str]]:
        """Implements requested features one by one"""
        all_files = []
        for feature in context.features:
            prompt = self._build_feature_prompt(context, feature)
            response = await self.llm.generate_code(prompt, context.language)
            files = self._parse_file_content(response)
            all_files.extend(files)
        return all_files

    async def _generate_tests(self, context: GenerationContext) -> List[Tuple[str, str]]:
        """Generates test cases for the implemented features"""
        prompt = self._build_tests_prompt(context)
        response = await self.llm.generate_code(prompt, context.language)
        return self._parse_file_content(response)

    async def _create_documentation(self, context: GenerationContext) -> List[Tuple[str, str]]:
        """Creates project documentation including README and setup instructions"""
        prompt = self._build_documentation_prompt(context)
        response = await self.llm.generate_code(prompt, context.language)
        return self._parse_file_content(response)

    def _build_structure_prompt(self, context: GenerationContext) -> str:
        """Builds prompt for project structure generation"""
        features_str = ", ".join(context.features) if context.features else "basic setup"

        # Add explicit directory structure guidance
        directory_guidance = """
        IMPORTANT Directory Structure Rules:
        1. All imports should be relative to the project root
        2. Do NOT include 'src' in import paths
        3. For a Python project, structure should be:
           project_root/
           ├── core/
           │   ├── __init__.py
           │   └── app.py
           ├── tests/
           │   ├── __init__.py
           │   └── test_*.py
           └── main.py

        Example correct imports:
        - Use: from core.app import Calculator
        - NOT: from src.core.app import Calculator
        """

        # Define project-specific essential files and structure
        project_templates = {
            "python-cli": f"""
            Essential files to include:
            1. core/
               - __init__.py
               - app.py          # Core functionality
            2. tests/
               - __init__.py
               - test_app.py     # Unit tests
            3. main.py          # Entry point
            4. README.md        # Documentation

            {directory_guidance}
            """,

            "nextjs": """
            Minimum, but not limited to, essential files to include:
            1. pages/
               - index.tsx
               - _app.tsx
            2. components/
               - Layout.tsx
            3. styles/
               - globals.css
            4. public/
            5. package.json
            6. tsconfig.json
            7. README.md

            IMPORTANT: All imports should be relative to the project root
            Example: import { Layout } from 'components/Layout'
            """,
            "flask": """
            Minimum, but not limited to, essential files to include:
            1. requirements.txt with all dependencies
            2. app.py or application factory
            3. config.py for configuration management
            4. templates/ directory for Jinja templates
            5. static/ directory for assets
            6. models/ directory for database models
            7. routes/ directory for view functions
            8. tests/ directory
            9. README.md with setup instructions""",

                "django": """
            Minimum, but not limited to, essential files to include:
            1. manage.py
            2. requirements.txt
            3. project/settings.py
            4. project/urls.py
            5. project/wsgi.py
            6. apps/main/models.py
            7. apps/main/views.py
            8. apps/main/urls.py
            9. templates/ directory
            10. static/ directory
            11. README.md with setup instructions""",

                "smart-contract": """
            Minimum, but not limited to, essential files to include:
            1. contracts/ directory with Solidity files
            2. package.json with development dependencies
            3. hardhat.config.js/ts for network config
            4. scripts/ directory for deployment
            5. test/ directory for contract tests
            6. .env.example for environment variables
            7. README.md with setup instructions""",

                "hardhat": """
            Minimum, but not limited to, essential files to include:
            1. package.json with Hardhat dependencies
            2. hardhat.config.ts with network configuration
            3. contracts/ directory for Solidity files
            4. scripts/ directory for deployment
            5. test/ directory for contract tests
            6. tasks/ directory for custom tasks
            7. .env.example for environment variables
            8. README.md with setup instructions""",

                "react": """
            Minimum, but not limited to, essential files to include:
            1. package.json with dependencies
            2. tsconfig.json for TypeScript
            3. src/App.tsx main component
            4. src/index.tsx entry point
            5. public/index.html
            6. src/components/ directory
            7. src/styles/ directory
            8. README.md with setup instructions""",

            "html": """
            Minimum, but not limited to, essential files to include:
            1. index.html         # Main entry point
            2. css/
               - styles.css       # Main stylesheet
            3. js/
               - main.js         # Main JavaScript file
            4. assets/
               - images/         # For image files
               - fonts/          # For font files
            5. pages/            # Additional HTML pages
            6. README.md         # Documentation

            IMPORTANT: 
            1. Use relative paths for all resources
            2. Follow HTML5 standards
            3. Include proper meta tags
            4. Ensure responsive design
            5. Include favicon
            """,
        }

        # Get template for the project type or use a generic one
        structure_template = project_templates.get(context.project_type, """
            Minimum, but not limited to, essential files to include:
            1. Main source code directory
            2. Configuration files
            3. Documentation files
            4. README.md with setup instructions

            IMPORTANT: All imports should be relative to the project root
            """)

        return f"""Create a complete {context.project_type} project with the following features:
                {features_str}

                Requirements:
                {self._format_requirements(context.requirements)}

                {structure_template}

                For each file, strictly use this format:
                FILE: path/to/file
                ###CONTENT_START###
                [Complete code content]
                ###CONTENT_END###

                IMPORTANT RULES:
                1. All imports must be relative to project root (no 'src' prefix)
                2. Each file must start with FILE: marker
                3. Content must be between ###CONTENT_START### and ###CONTENT_END### markers
                4. Include all necessary imports
                5. No placeholders or TODO items
                6. Follow the exact directory structure specified
                7. Use proper relative imports between files
                """

    def _build_core_files_prompt(self, context: GenerationContext) -> str:
        """Generates the core application files"""
        project_specific_instructions = {
            "python-cli": """
                Create a focused implementation that matches the project requirements.
                
                Key points:
                1. Keep the code simple and readable
                2. Only implement requested features
                3. Use clear function and variable names
                4. Add appropriate error handling
                5. Follow Python best practices
                """,
            
            "nextjs": """
                Create a focused Next.js implementation that matches the project requirements.
                
                Key points:
                1. Use TypeScript for type safety
                2. Implement proper component structure
                3. Follow React/Next.js best practices
                4. Include proper error boundaries
                5. Use proper data fetching methods
                6. Implement responsive design
                7. Follow accessibility guidelines
                """,

            "react": """
                Create a focused React implementation that matches the project requirements.
                
                Key points:
                1. Use TypeScript for type safety
                2. Follow component composition best practices
                3. Implement proper state management
                4. Use React hooks effectively
                5. Include error boundaries
                6. Follow accessibility guidelines
                7. Optimize performance with proper memoization
                """,

            "flask": """
                Create a focused Flask implementation that matches the project requirements.
                
                Key points:
                1. Follow Flask application factory pattern
                2. Implement proper error handling
                3. Use Blueprint structure for routes
                4. Include proper database models
                5. Implement RESTful API best practices
                6. Add proper request validation
                7. Include security best practices
                """,

            "django": """
                Create a focused Django implementation that matches the project requirements.
                
                Key points:
                1. Follow Django project structure best practices
                2. Implement proper models and migrations
                3. Use class-based views where appropriate
                4. Include proper form validation
                5. Implement Django REST framework if needed
                6. Follow security best practices
                7. Use Django's built-in features effectively
                """,

            "smart-contract": """
                Create a focused smart contract implementation that matches the project requirements.
                
                Key points:
                1. Follow Solidity best practices
                2. Implement proper security measures
                3. Optimize for gas efficiency
                4. Include comprehensive tests
                5. Add proper access controls
                6. Implement event emissions
                7. Follow upgrade patterns if needed
                """,

            "html": """
                Create a focused HTML/CSS implementation that matches the project requirements.
                
                Key points:
                1. Use semantic HTML5 elements
                2. Implement responsive design
                3. Follow accessibility guidelines
                4. Optimize for performance
                5. Use modern CSS features appropriately
                6. Ensure cross-browser compatibility
                7. Follow progressive enhancement principles
                """
        }

        extra_instructions = project_specific_instructions.get(context.project_type, "")

        return f'''Generate complete, production-ready core files for the {context.project_type} project.

                Features to implement:
                {', '.join(context.features) if context.features else 'basic setup'}

                Requirements:
                {self._format_requirements(context.requirements)}

                {extra_instructions}

                IMPORTANT: You must use EXACTLY this format for EACH file:
                FILE: path/to/file
                ###CONTENT_START###
                [complete file content here]
                ###CONTENT_END###

                Example of correct format:
                FILE: src/main.py
                ###CONTENT_START###
                def main():
                    print("Hello World")

                if __name__ == "__main__":
                    main()
                ###CONTENT_END###

                REQUIREMENTS:
                1. Every file must start with 'FILE: ' followed by the path
                2. Content must be between ###CONTENT_START### and ###CONTENT_END### markers
                3. Include complete, working code (no placeholders)
                4. Include all necessary imports
                5. For directories, use:
                   FILE: path/to/directory
                   ###CONTENT_START###
                   ###CONTENT_END###
                '''

    def _build_feature_prompt(self, context: GenerationContext, feature: str) -> str:
        """Builds prompt for specific feature implementation"""
        return f"""Implement the {feature} feature for the {context.project_type} project.

                Project requirements:
                {self._format_requirements(context.requirements)}

                IMPORTANT: You must use EXACTLY this format for EACH file:
                FILE: path/to/file
                ###CONTENT_START###
                [complete file content here]
                ###CONTENT_END###

                Example of correct format:
                FILE: src/components/Button.tsx
                ###CONTENT_START###
                import React from 'react';

                interface ButtonProps {{
                    label: string;
                    onClick: () => void;
                }}

                export const Button: React.FC<ButtonProps> = ({{ label, onClick }}) => {{
                    return (
                        <button onClick={{onClick}}>{{label}}</button>
                    );
                }};
                ###CONTENT_END###

                FILE: src/types/index.ts
                ###CONTENT_START###
                export interface User {{
                    id: string;
                    name: string;
                }}
                ###CONTENT_END###

                REQUIREMENTS:
                1. Every file must start with 'FILE: ' followed by the path
                2. Content must be between ###CONTENT_START### and ###CONTENT_END### markers
                3. Include complete, working code (no placeholders)
                4. Include all necessary imports
                5. For directories, use:
                   FILE: path/to/directory
                   ###CONTENT_START###
                   ###CONTENT_END###
                """

    def _build_tests_prompt(self, context: GenerationContext) -> str:
        """Builds prompt for test generation"""
        project_name = context.requirements.get('project_name', 'cli_app')
        return f"""Create comprehensive tests for the {context.project_type} project.

        Test files should:
        1. Be placed in the tests/ directory
        2. Use proper imports from {project_name} package
        3. Include both unit and integration tests
        4. Follow pytest best practices
        5. Have proper test documentation
        6. Cover the main functionality

        Features to test:
        {', '.join(context.features) if context.features else 'basic functionality'}

        Return each test file in this format:
        filename:
        ```
        [complete test implementation]
        ```"""

    def _build_documentation_prompt(self, context: GenerationContext) -> str:
        """Builds prompt for documentation generation"""
        return f"""Create COMPLETE documentation for a {context.project_type} project.

                Project Features:
                {', '.join(context.features) if context.features else 'basic setup'}

                Project Requirements:
                {self._format_requirements(context.requirements)}

                Create a comprehensive README.md that MUST include ONLY relevant sections for this specific project:

                1. Project Title and Description
                   - What the project does
                   - Key features and capabilities
                   - Any limitations or requirements

                2. Prerequisites
                   - Required Python version
                   - Required environment variables (e.g., API keys)
                   - Any system dependencies

                3. Installation & Setup
                   - EXACT steps to get the code running
                   - How to set up environment variables
                   - How to install dependencies (using pip and requirements.txt)
                   - NO generic pip package installation instructions unless project has setup.py

                4. Usage Guide
                   - EXACT commands to run the project
                   - Example inputs and outputs
                   - Common use cases
                   - Any configuration options

                5. Project Structure (if complex enough to warrant it)
                   - Key files and their purposes
                   - How the code is organized

                6. Contributing (only if project is open source)
                   - How to submit changes
                   - Development setup

                7. License (if applicable)

                IMPORTANT:
                - IT MUST HAVE ALL DETAILS FOR USER TO BE ABLE TO RUN THE PROJECT, INCLUDING ALL DEPENDENCIES, SETUP, AND INSTALLATION
                - IT MUST INCLUDE ALL THE FILES THAT USER NEEDS TO CREATE, INCLUDING FORMAT AND STRUCTURE
                - Include ONLY relevant sections
                - NO placeholder content
                - NO generic instructions that don't apply
                - ALL commands must be actual working commands
                - Installation instructions must match actual project structure
                - If project is local-only (no setup.py), don't include pip installation
                - Focus on getting the user running quickly
                - Include actual environment variable names
                - Show real example commands with expected output

                Return complete documentation in this format:
                FILE: README.md
                ```markdown
                [Complete README content following above structure]
                ```"""

    def _format_requirements(self, requirements: Dict[str, any]) -> str:
        """Formats requirements dict into a string for prompts"""
        if not requirements:
            return "No specific requirements provided"
        return "\n".join(f"- {k}: {v}" for k, v in requirements.items())