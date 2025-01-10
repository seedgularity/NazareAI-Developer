# src/worker/analyzer/python.py

from pathlib import Path
import ast
from typing import Dict, List, Set, Optional
import pylint.lint
from pylint.reporters import JSONReporter

from .base_analyzer import BaseAnalyzer, BaseAnalysisResult
from ...llm.providers.base_provider import BaseProvider
from ...utils.logger import get_logger

logger = get_logger(__name__)


class PythonAnalysisResult(BaseAnalysisResult):
    """Python-specific analysis results combining static and LLM analysis"""

    def __init__(
            self,
            score: float,
            issues: List[Dict],
            suggestions: List[str],
            files_analyzed: List[str],
            missing_imports: Set[str],
            requirements: Set[str],
            llm_insights: Dict[str, any] = None
    ):
        super().__init__(
            score=score,
            issues=issues,
            suggestions=suggestions,
            files_analyzed=files_analyzed
        )
        self.missing_imports = missing_imports
        self.requirements = requirements
        self.llm_insights = llm_insights or {}

    def to_dict(self) -> Dict:
        base_dict = super().to_dict()
        base_dict.update({
            "missing_imports": list(self.missing_imports),
            "requirements": list(self.requirements),
            "llm_insights": self.llm_insights
        })
        return base_dict


class PythonAnalyzer(BaseAnalyzer):
    """Python code analyzer using both static analysis and LLM"""

    def __init__(self, primary_llm: BaseProvider, backup_llm: Optional[BaseProvider] = None):
        super().__init__()
        self.primary_llm = primary_llm
        self.backup_llm = backup_llm
        self.logger = logger

    async def _get_llm_analysis(self, file_path: Path, content: str) -> Dict:
        """Get code analysis from LLM"""
        try:
            prompt = f"""Analyze this Python code for quality, security, and improvements:

            File: {file_path.name}

            ```python
            {content}
            ```

            Provide analysis in the following format:
            1. Security issues (if any)
            2. Code quality issues
            3. Performance improvements
            4. Best practice violations
            5. Suggested refactoring
            6. Missing error handling
            7. Dependencies and their purposes
            8. Potential bugs

            For each issue found, provide:
            - Issue location (line number if applicable)
            - Severity (high/medium/low)
            - Description
            - Suggested fix"""

            try:
                response = await self.primary_llm.analyze_code(content, "python", {"context": str(file_path)})
            except Exception as e:
                self.logger.warning(f"Primary LLM failed: {e}, trying backup...")
                if self.backup_llm:
                    response = await self.backup_llm.analyze_code(content, "python", {"context": str(file_path)})
                else:
                    raise

            return response

        except Exception as e:
            self.logger.error(f"LLM analysis failed: {e}")
            return {}

    async def analyze_project(self, project_dir: Path) -> PythonAnalysisResult:
        """Analyze entire Python project using both static analysis and LLM"""
        try:
            # Static analysis results
            all_issues = []
            all_imports = set()
            files_analyzed = []
            total_score = 0
            file_count = 0

            # LLM analysis results
            llm_insights = {
                "project_level": {},
                "file_level": {},
                "suggestions": []
            }

            # Get all Python files
            python_files = await self.get_language_files(project_dir)

            # Analyze each file
            for file_path in python_files:
                if "venv" not in str(file_path):
                    # Static analysis
                    file_result = await self.analyze_file(file_path)
                    all_issues.extend(file_result.get("issues", []))
                    all_imports.update(file_result.get("imports", set()))
                    files_analyzed.append(str(file_path))
                    total_score += file_result.get("score", 0)
                    file_count += 1

                    # LLM analysis
                    with open(file_path, 'r') as f:
                        content = f.read()
                    llm_result = await self._get_llm_analysis(file_path, content)
                    if llm_result:
                        llm_insights["file_level"][str(file_path)] = llm_result

            # Project-level LLM analysis
            project_prompt = f"""Analyze this Python project structure and provide insights:

            Files analyzed: {files_analyzed}

            Consider:
            1. Overall architecture
            2. Dependency management
            3. Project organization
            4. Testing coverage
            5. Documentation needs
            6. Deployment considerations
            7. Security considerations
            8. Performance optimization opportunities"""

            try:
                project_analysis = await self.primary_llm.analyze_code(
                    str(files_analyzed),
                    "python",
                    {"context": "project_structure"}
                )
                llm_insights["project_level"] = project_analysis
            except Exception as e:
                self.logger.warning(f"Project-level LLM analysis failed: {e}")

            # Calculate final results
            avg_score = total_score / file_count if file_count > 0 else 0
            dependencies = await self.validate_dependencies(project_dir)
            missing_imports = set(dependencies.get("missing", []))
            requirements = set(dependencies.get("present", []))

            # Combine static and LLM suggestions
            static_suggestions = self._generate_suggestions(all_issues, missing_imports)
            llm_suggestions = llm_insights.get("project_level", {}).get("suggestions", [])
            combined_suggestions = static_suggestions + llm_suggestions

            # Count errors and warnings
            error_count = sum(1 for issue in all_issues if issue.get('type') == 'error')
            warning_count = sum(1 for issue in all_issues if issue.get('type') == 'warning')

            return PythonAnalysisResult(
                score=avg_score,
                issues=all_issues,
                suggestions=combined_suggestions,
                files_analyzed=files_analyzed,
                missing_imports=missing_imports,
                requirements=requirements,
                llm_insights=llm_insights,
                error_count=error_count,
                warning_count=warning_count
            )

        except Exception as e:
            self.logger.error(f"Error analyzing Python project: {e}")
            raise

    async def fix_issues(self, project_dir: Path, issues: List[Dict]) -> bool:
        """Fix identified issues using both static and LLM-powered solutions."""
        try:
            fixed_count = 0

            # Group issues by file for efficiency
            issues_by_file = {}
            for issue in issues:
                file_path = Path(issue.get('file', ''))
                if not file_path.is_absolute():
                    file_path = project_dir / file_path

                if not file_path.exists():
                    continue

                if str(file_path) not in issues_by_file:
                    issues_by_file[str(file_path)] = []
                issues_by_file[str(file_path)].append(issue)

            # Process each file's issues
            for file_path_str, file_issues in issues_by_file.items():
                file_path = Path(file_path_str)

                # First try static fixes
                for issue in file_issues:
                    if issue["code"] in ["missing-docstring", "trailing-whitespace", "missing-final-newline"]:
                        if await self._fix_simple_issue(file_path, issue):
                            fixed_count += 1
                    elif issue["code"] in ["unused-import", "wrong-import-order"]:
                        if await self._fix_import_issue(file_path, issue):
                            fixed_count += 1
                    elif issue["code"].startswith("llm-") or issue.get("severity") in ["high", "medium"]:
                        # Use LLM for complex issues or those identified by LLM
                        if await self._apply_llm_fix(file_path, issue):
                            fixed_count += 1

            return fixed_count > 0

        except Exception as e:
            self.logger.error(f"Error fixing issues: {e}")
            return False

    async def _fix_simple_issue(self, file_path: Path, issue: Dict) -> bool:
        """Fix simple issues like whitespace and docstrings."""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            line_num = issue.get('line', 0) - 1
            if line_num < 0 or line_num >= len(lines):
                return False

            if issue["code"] == "missing-docstring":
                # Add simple docstring
                indent = len(lines[line_num]) - len(lines[line_num].lstrip())
                docstring = ' ' * indent + '"""Add docstring here."""\n'
                lines.insert(line_num + 1, docstring)

            elif issue["code"] == "trailing-whitespace":
                # Remove trailing whitespace
                lines[line_num] = lines[line_num].rstrip() + '\n'

            elif issue["code"] == "missing-final-newline":
                # Add final newline
                if not lines[-1].endswith('\n'):
                    lines[-1] = lines[-1] + '\n'

            with open(file_path, 'w') as f:
                f.writelines(lines)

            return True

        except Exception as e:
            self.logger.error(f"Error fixing simple issue: {e}")
            return False

    async def _fix_import_issue(self, file_path: Path, issue: Dict) -> bool:
        """Fix import-related issues."""
        try:
            with open(file_path, 'r') as f:
                tree = ast.parse(f.read())

            imports = []
            other_nodes = []

            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if issue["code"] != "unused-import" or self._is_import_used(node, tree):
                        imports.append(node)
                else:
                    other_nodes.append(node)

            # Sort imports
            imports.sort(key=lambda x: ast.unparse(x))

            # Reconstruct file
            new_tree = ast.Module(body=imports + other_nodes)

            with open(file_path, 'w') as f:
                f.write(ast.unparse(new_tree))

            return True

        except Exception as e:
            self.logger.error(f"Error fixing import issue: {e}")
            return False

    def _is_import_used(self, import_node: ast.AST, tree: ast.AST) -> bool:
        """Check if an import is used in the code."""
        # Get imported names
        if isinstance(import_node, ast.Import):
            names = [alias.name for alias in import_node.names]
        else:  # ImportFrom
            names = [f"{import_node.module}.{alias.name}" for alias in import_node.names]

        # Find all name references in the code
        name_finder = NameFinder(names)
        name_finder.visit(tree)

        return name_finder.found_names

    async def analyze_file(self, file_path: Path) -> Dict:
        """Analyze a single Python file using both static analysis and LLM."""
        try:
            # Run static analysis
            score, static_issues = self._run_pylint(file_path)
            imports = self._extract_imports(file_path)

            # Get LLM analysis
            with open(file_path, 'r') as f:
                content = f.read()
            llm_result = await self._get_llm_analysis(file_path, content)

            # Combine results
            combined_issues = static_issues
            if llm_result:
                llm_issues = self._convert_llm_insights_to_issues(llm_result)
                combined_issues.extend(llm_issues)
                score = min(score, self._calculate_llm_score(llm_result))

            return {
                "score": score,
                "issues": combined_issues,
                "imports": imports,
                "llm_insights": llm_result
            }

        except Exception as e:
            self.logger.error(f"Error analyzing file {file_path}: {e}")
            return {"score": 0, "issues": [], "imports": set()}

    def _convert_llm_insights_to_issues(self, llm_result: Dict) -> List[Dict]:
        """Convert LLM insights into standard issue format."""
        issues = []

        # Expected categories in LLM response
        categories = {
            'security_issues': 'security',
            'quality_issues': 'quality',
            'best_practices': 'convention',
            'potential_bugs': 'error'
        }

        for category, issue_type in categories.items():
            if category in llm_result:
                for issue in llm_result[category]:
                    issues.append({
                        'type': issue_type,
                        'line': issue.get('line'),
                        'column': issue.get('column'),
                        'message': issue.get('description'),
                        'severity': issue.get('severity', 'medium'),
                        'suggested_fix': issue.get('fix'),
                        'code': f'llm-{issue_type}'
                    })

        return issues

    def _calculate_llm_score(self, llm_result: Dict) -> float:
        """Calculate a score based on LLM analysis."""
        score = 10.0

        # Count issues by severity
        severity_weights = {'high': 2.0, 'medium': 1.0, 'low': 0.5}
        issue_count = 0

        for category in llm_result.values():
            if isinstance(category, list):
                for issue in category:
                    severity = issue.get('severity', 'medium').lower()
                    score -= severity_weights.get(severity, 1.0)
                    issue_count += 1

        # Normalize score based on number of issues
        if issue_count > 0:
            score = max(0.0, min(score, 10.0))

        return score

    async def _apply_llm_fix(self, file_path: Path, issue: Dict) -> bool:
        """Use LLM to fix complex issues that can't be handled by static analysis."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()

            # Create focused prompt for specific issue
            prompt = f"""Fix this specific issue in the Python code:

            File: {file_path.name}
            Issue: {issue['message']}
            Location: Line {issue.get('line', 'unknown')}
            Type: {issue.get('type', 'unknown')}
            Severity: {issue.get('severity', 'medium')}

            Current code:
            ```python
            {content}
            ```

            Requirements:
            1. Return the complete fixed code
            2. Add comments explaining the fix
            3. Ensure the fix doesn't introduce new issues
            4. Maintain code style and formatting
            5. Preserve existing functionality

            Return only the fixed code in a code block."""

            try:
                response = await self.primary_llm.generate_code(prompt, "python", {
                    "context": str(file_path),
                    "issue_type": issue.get('type', ''),
                    "severity": issue.get('severity', 'medium')
                })
            except Exception as e:
                if self.backup_llm:
                    response = await self.backup_llm.generate_code(prompt, "python")
                else:
                    raise

            # Extract and validate fixed code
            fixed_code = response.strip()
            if '```python' in fixed_code:
                fixed_code = fixed_code.split('```python')[1].split('```')[0].strip()

            if fixed_code and fixed_code != content:
                # Validate the fixed code parses correctly
                try:
                    ast.parse(fixed_code)
                    with open(file_path, 'w') as f:
                        f.write(fixed_code)
                    return True
                except SyntaxError:
                    self.logger.error(f"LLM generated invalid Python code for {file_path}")
                    return False

            return False

        except Exception as e:
            self.logger.error(f"Error applying LLM fix: {e}")
            return False

    def get_file_extensions(self) -> List[str]:
        """Get list of file extensions this analyzer can handle."""
        return [".py", ".pyw", ".pyi"]

    async def validate_dependencies(self, project_dir: Path) -> Dict[str, List[str]]:
        """Validate Python project dependencies - implementing abstract method."""
        try:
            # Parse requirements.txt
            requirements = self._parse_requirements(project_dir / "requirements.txt")

            # Get all imports from project
            all_imports = set()
            python_files = [f for f in await self.get_language_files(project_dir)
                            if "venv" not in str(f)]

            for file_path in python_files:
                imports = self._extract_imports(file_path)
                all_imports.update(imports)

            # Find missing and unused dependencies
            missing = all_imports - requirements
            unused = requirements - all_imports

            return {
                "missing": list(missing),
                "unused": list(unused),
                "present": list(requirements)
            }

        except Exception as e:
            self.logger.error(f"Error validating dependencies: {e}")
            return {"missing": [], "unused": [], "present": []}

    def _run_pylint(self, file_path: Path) -> tuple[float, List[Dict]]:
        """Run pylint on a single file and return score and issues."""
        try:
            # Create JSON reporter
            reporter = JSONReporter()

            # Run pylint (without do_exit parameter)
            args = [str(file_path)]
            pylint.lint.Run(args, reporter=reporter)

            # Extract results
            issues = reporter.data
            score = 10.0  # Default score

            # Calculate score based on issues
            if issues:
                score = max(0.0, 10.0 - (len(issues) * 0.5))

            return score, issues

        except Exception as e:
            self.logger.error(f"Error running pylint on {file_path}: {e}")
            return 0.0, []

    def _extract_imports(self, file_path: Path) -> Set[str]:
        """Extract all imports from a Python file using AST."""
        imports = set()
        try:
            with open(file_path, 'r') as f:
                tree = ast.parse(f.read())

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imports.add(name.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])

            # Filter out standard library modules
            std_libs = self._get_standard_library_modules()
            return {imp for imp in imports if imp not in std_libs}

        except Exception as e:
            self.logger.error(f"Error extracting imports from {file_path}: {e}")
            return set()

    def _parse_requirements(self, req_file: Path) -> Set[str]:
        """Parse requirements.txt and return set of package names."""
        requirements = set()
        if req_file.exists():
            try:
                with open(req_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            # Handle version specifiers
                            pkg_name = line.split('==')[0].split('>=')[0].split('<=')[0]
                            requirements.add(pkg_name)
            except Exception as e:
                self.logger.error(f"Error parsing requirements.txt: {e}")
        return requirements

    def _generate_suggestions(self, issues: List[Dict], missing_imports: Set[str]) -> List[str]:
        """Generate improvement suggestions based on analysis results."""
        suggestions = []

        # Group issues by type
        issue_types = {}
        for issue in issues:
            issue_type = issue.get('type', '')
            if issue_type not in issue_types:
                issue_types[issue_type] = 0
            issue_types[issue_type] += 1

        # Add suggestions based on issue patterns
        if issue_types.get('missing-docstring', 0) > 0:
            suggestions.append("Add docstrings to functions and classes for better documentation")

        if issue_types.get('invalid-name', 0) > 0:
            suggestions.append("Follow PEP 8 naming conventions for variables and functions")

        if missing_imports:
            suggestions.append(f"Add missing dependencies to requirements.txt: {', '.join(missing_imports)}")

        if issue_types.get('unused-import', 0) > 0:
            suggestions.append("Remove unused imports to improve code clarity")

        return suggestions

    def _get_standard_library_modules(self) -> Set[str]:
        """Get a set of Python standard library module names."""
        return {
            'abc', 'argparse', 'ast', 'asyncio', 'base64', 'collections',
            'configparser', 'contextlib', 'copy', 'csv', 'datetime', 'decimal',
            'enum', 'functools', 'glob', 'gzip', 'hashlib', 'hmac', 'html',
            'http', 'importlib', 'io', 'itertools', 'json', 'logging', 'math',
            'multiprocessing', 'operator', 'os', 'pathlib', 'pickle', 'platform',
            'queue', 'random', 're', 'shutil', 'signal', 'socket', 'sqlite3',
            'string', 'subprocess', 'sys', 'tempfile', 'threading', 'time',
            'typing', 'unittest', 'urllib', 'uuid', 'warnings', 'weakref',
            'xml', 'zipfile'
        }

    async def apply_fixes(self, project_dir: Path, fix_response: str) -> bool:
        """Apply fixes suggested by the LLM to the project files."""
        try:
            logger.debug(f"Applying fixes to {project_dir}")
            logger.debug(f"Fix response: {fix_response}")
            
            # Parse file blocks from the response
            file_blocks = re.finditer(
                r'FILE:\s*([^\n]+)\n```(?:python)?\n(.*?)\n```',
                fix_response,
                re.DOTALL
            )

            fixed_count = 0
            for match in file_blocks:
                filename = match.group(1).strip()
                new_content = match.group(2).strip()
                
                logger.debug(f"Found fix for file: {filename}")
                logger.debug(f"New content length: {len(new_content)}")

                file_path = project_dir / filename
                if not file_path.is_absolute():
                    file_path = project_dir / file_path

                # Validate the new content
                try:
                    ast.parse(new_content)
                except SyntaxError as e:
                    logger.error(f"Invalid Python syntax in fix for {filename}: {e}")
                    continue

                # Backup original file
                backup_path = file_path.with_suffix(file_path.suffix + '.bak')
                if file_path.exists():
                    logger.debug(f"Creating backup at {backup_path}")
                    file_path.rename(backup_path)

                try:
                    # Write new content
                    logger.debug(f"Writing new content to {file_path}")
                    with open(file_path, 'w') as f:
                        f.write(new_content)
                    fixed_count += 1
                    logger.info(f"Successfully updated {filename}")
                except Exception as e:
                    logger.error(f"Error writing fixes to {filename}: {e}")
                    # Restore backup
                    if backup_path.exists():
                        backup_path.rename(file_path)
                    continue

            logger.info(f"Fixed {fixed_count} files")
            return fixed_count > 0

        except Exception as e:
            logger.error(f"Error applying fixes: {e}")
            return False

class NameFinder(ast.NodeVisitor):
    """Helper class to find name usage in AST."""
    def __init__(self, names: List[str]):
        self.names = names
        self.found_names = False

    def visit_Name(self, node):
        if node.id in self.names:
            self.found_names = True