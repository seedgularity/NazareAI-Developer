from pathlib import Path
import json
from typing import Dict, List, Set, Optional
import re

from .base_analyzer import BaseAnalyzer, BaseAnalysisResult
from ...llm.providers.base_provider import BaseProvider
from ...utils.logger import get_logger

logger = get_logger(__name__)


class JavaScriptAnalysisResult(BaseAnalysisResult):
    """JavaScript/TypeScript-specific analysis results"""

    def __init__(
            self,
            score: float,
            issues: List[Dict],
            suggestions: List[str],
            files_analyzed: List[str],
            dependencies: Set[str],
            dev_dependencies: Set[str],
            type_issues: List[Dict] = None,
            llm_insights: Dict[str, any] = None
    ):
        super().__init__(
            score=score,
            issues=issues,
            suggestions=suggestions,
            files_analyzed=files_analyzed
        )
        self.dependencies = dependencies
        self.dev_dependencies = dev_dependencies
        self.type_issues = type_issues or []
        self.llm_insights = llm_insights or {}

    def to_dict(self) -> Dict:
        base_dict = super().to_dict()
        base_dict.update({
            "dependencies": list(self.dependencies),
            "dev_dependencies": list(self.dev_dependencies),
            "type_issues": self.type_issues,
            "llm_insights": self.llm_insights
        })
        return base_dict


class JavaScriptAnalyzer(BaseAnalyzer):
    """JavaScript/TypeScript code analyzer using ESLint, TypeScript and LLM"""

    def __init__(self, primary_llm: BaseProvider, backup_llm: Optional[BaseProvider] = None):
        super().__init__()
        self.primary_llm = primary_llm
        self.backup_llm = backup_llm
        self.logger = logger

    async def analyze_project(self, project_dir: Path) -> JavaScriptAnalysisResult:
        """Analyze entire JavaScript/TypeScript project"""
        try:
            all_issues = []
            files_analyzed = []
            total_score = 0
            file_count = 0
            dependencies = set()
            dev_dependencies = set()

            # Get all JS/TS files
            js_files = await self.get_language_files(project_dir)

            # LLM insights storage
            llm_insights = {
                "project_level": {},
                "file_level": {},
                "suggestions": []
            }

            # Analyze each file
            for file_path in js_files:
                if "node_modules" not in str(file_path):
                    # Static analysis
                    file_result = await self.analyze_file(file_path)
                    all_issues.extend(file_result.get("issues", []))
                    files_analyzed.append(str(file_path))
                    total_score += file_result.get("score", 0)
                    file_count += 1

                    # LLM analysis
                    with open(file_path, 'r') as f:
                        content = f.read()
                    llm_result = await self._get_llm_analysis(file_path, content)
                    if llm_result:
                        llm_insights["file_level"][str(file_path)] = llm_result

            # Parse package.json for dependencies
            pkg_json = project_dir / "package.json"
            if pkg_json.exists():
                with open(pkg_json) as f:
                    pkg_data = json.load(f)
                    dependencies.update(pkg_data.get("dependencies", {}).keys())
                    dev_dependencies.update(pkg_data.get("devDependencies", {}).keys())

            # Calculate final results
            avg_score = total_score / file_count if file_count > 0 else 0

            return JavaScriptAnalysisResult(
                score=avg_score,
                issues=all_issues,
                suggestions=self._generate_suggestions(all_issues),
                files_analyzed=files_analyzed,
                dependencies=dependencies,
                dev_dependencies=dev_dependencies,
                llm_insights=llm_insights
            )

        except Exception as e:
            self.logger.error(f"Error analyzing JavaScript project: {e}")
            raise

    async def analyze_file(self, file_path: Path) -> Dict:
        """Analyze a single JavaScript/TypeScript file"""
        try:
            issues = []
            score = 10.0

            # Read file content
            with open(file_path, 'r') as f:
                content = f.read()

            # Basic static analysis
            issues.extend(self._check_basic_issues(content))

            # TypeScript-specific checks for .ts/.tsx files
            if file_path.suffix in ['.ts', '.tsx']:
                type_issues = self._check_typescript_issues(content)
                issues.extend(type_issues)

            # React-specific checks for .jsx/.tsx files
            if file_path.suffix in ['.jsx', '.tsx']:
                react_issues = self._check_react_issues(content)
                issues.extend(react_issues)

            # Calculate score based on issues
            score -= len(issues) * 0.5

            return {
                "score": max(0.0, score),
                "issues": issues
            }

        except Exception as e:
            self.logger.error(f"Error analyzing file {file_path}: {e}")
            return {"score": 0, "issues": []}

    def _check_basic_issues(self, content: str) -> List[Dict]:
        """Check for basic JavaScript issues"""
        issues = []

        # Check for console.log statements
        if 'console.log' in content:
            issues.append({
                "code": "no-console",
                "message": "Avoid using console.log in production code",
                "severity": "warning"
            })

        # Check for undefined variables - updated regex to be more robust
        try:
            # Match variable assignments not preceded by declaration keywords
            undefined_vars = re.findall(
                r'(?<!let\s)(?<!const\s)(?<!var\s)(?<!function\s)(?<!class\s)(?<!interface\s)(?<!type\s)'
                r'\b([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=(?!=)',
                content
            )
            for var in undefined_vars:
                if var not in ['props', 'state']:  # Common React variables to ignore
                    issues.append({
                        "code": "no-undef",
                        "message": f"Variable {var} might be used before declaration",
                        "severity": "error"
                    })
        except re.error:
            self.logger.warning("Regex pattern failed in undefined variables check")

        return issues

    def _check_typescript_issues(self, content: str) -> List[Dict]:
        """Check for TypeScript-specific issues"""
        issues = []

        # Check for any/unknown types
        if 'any' in content:
            issues.append({
                "code": "no-explicit-any",
                "message": "Avoid using 'any' type",
                "severity": "warning"
            })

        # Check for missing type definitions
        if re.search(r'function\s+\w+\s*\([^:]+\)', content):
            issues.append({
                "code": "missing-types",
                "message": "Function parameters should have type annotations",
                "severity": "warning"
            })

        return issues

    def _check_react_issues(self, content: str) -> List[Dict]:
        """Check for React-specific issues"""
        issues = []

        # Check for missing dependency arrays in useEffect
        if 'useEffect(' in content and not re.search(r'useEffect\([^,]+,\s*\[[^\]]*\]\)', content):
            issues.append({
                "code": "react-hooks/exhaustive-deps",
                "message": "useEffect should have dependency array",
                "severity": "warning"
            })

        # Check for missing key prop in lists
        if 'map(' in content and not re.search(r'key={[^}]+}', content):
            issues.append({
                "code": "react/jsx-key",
                "message": "Elements in list should have key prop",
                "severity": "error"
            })

        return issues

    def get_file_extensions(self) -> List[str]:
        """Get list of file extensions this analyzer can handle"""
        return [".js", ".jsx", ".ts", ".tsx"]

    async def fix_issues(self, project_dir: Path, issues: List[Dict]) -> bool:
        """Fix JavaScript-specific issues"""
        try:
            files_modified = False
            issues_by_file = {}
            
            for issue in issues:
                file_path = issue.get('file')
                if file_path:
                    if file_path not in issues_by_file:
                        issues_by_file[file_path] = []
                    issues_by_file[file_path].append(issue)

            for file_path, file_issues in issues_by_file.items():
                try:
                    with open(project_dir / file_path, 'r') as f:
                        content = f.read()
                    
                    modified_content = content
                    
                    for issue in file_issues:
                        if issue['code'] == 'no-console':
                            # Replace console.log with proper logging
                            modified_content = re.sub(
                                r'console\.(log|warn|error)\((.*?)\)',
                                r'logger.\1(\2)',
                                modified_content
                            )
                            # Add logger import if not present
                            if 'import logger' not in modified_content:
                                modified_content = "import logger from './utils/logger';\n" + modified_content
                        
                        elif issue['code'] == 'no-var':
                            # Replace var with let/const
                            modified_content = re.sub(
                                r'\bvar\b\s+(\w+)\s*=\s*([^;]+);',
                                r'const \1 = \2;',
                                modified_content
                            )

                    if modified_content != content:
                        with open(project_dir / file_path, 'w') as f:
                            f.write(modified_content)
                        files_modified = True
                        self.logger.info(f"Fixed issues in {file_path}")

                except Exception as e:
                    self.logger.error(f"Error fixing issues in {file_path}: {e}")
                    continue

            return files_modified

        except Exception as e:
            self.logger.error(f"Error fixing JavaScript issues: {e}")
            return False

    async def validate_dependencies(self, project_dir: Path) -> Dict[str, List[str]]:
        """Validate JavaScript/TypeScript project dependencies"""
        try:
            pkg_json = project_dir / "package.json"
            if not pkg_json.exists():
                return {"missing": [], "unused": [], "present": []}

            with open(pkg_json) as f:
                pkg_data = json.load(f)

            # Get declared dependencies
            all_deps = {
                **pkg_data.get("dependencies", {}),
                **pkg_data.get("devDependencies", {})
            }

            # Find imports in project files
            used_deps = set()
            for file_path in await self.get_language_files(project_dir):
                if "node_modules" not in str(file_path):
                    with open(file_path) as f:
                        content = f.read()
                        # Match import statements
                        imports = re.findall(r'(?:import|require)\s*\(?[\'"]([^\'"\s]+)[\'"]', content)
                        for imp in imports:
                            # Get package name (first part of import path)
                            pkg_name = imp.split('/')[0]
                            if not pkg_name.startswith('.'):
                                used_deps.add(pkg_name)

            # Compare declared vs used dependencies
            declared_deps = set(all_deps.keys())
            missing_deps = used_deps - declared_deps
            unused_deps = declared_deps - used_deps

            return {
                "missing": list(missing_deps),
                "unused": list(unused_deps),
                "present": list(declared_deps)
            }

        except Exception as e:
            self.logger.error(f"Error validating dependencies: {e}")
            return {"missing": [], "unused": [], "present": []}

    async def _get_llm_analysis(self, file_path: Path, content: str) -> Dict:
        """Get code analysis from LLM"""
        try:
            prompt = f"""Analyze this JavaScript/TypeScript code for quality and improvements:

            File: {file_path.name}

            ```javascript
            {content}
            ```

            Analyze for:
            1. Code quality issues
            2. Performance concerns
            3. Security vulnerabilities
            4. Best practice violations
            5. Type safety (for TypeScript)
            6. React-specific issues (if applicable)
            7. Potential bugs
            8. Architectural improvements

            Format each issue as:
            - Location (line number if possible)
            - Severity (high/medium/low)
            - Description
            - Suggested fix"""

            try:
                response = await self.primary_llm.analyze_code(content, "javascript")
            except Exception as e:
                if self.backup_llm:
                    response = await self.backup_llm.analyze_code(content, "javascript")
                else:
                    raise

            return response

        except Exception as e:
            self.logger.error(f"LLM analysis failed: {e}")
            return {}

    def _generate_suggestions(self, issues: List[Dict]) -> List[str]:
        """Generate improvement suggestions based on analysis results"""
        suggestions = []

        # Count issues by type
        issue_types = {}
        for issue in issues:
            issue_type = issue.get('code', '')
            if issue_type not in issue_types:
                issue_types[issue_type] = 0
            issue_types[issue_type] += 1

        # Add suggestions based on patterns
        if issue_types.get('no-console', 0) > 0:
            suggestions.append("Replace console.log with proper logging system")

        if issue_types.get('missing-types', 0) > 0:
            suggestions.append("Add TypeScript type annotations to improve type safety")

        if issue_types.get('react/jsx-key', 0) > 0:
            suggestions.append("Add key prop to elements in lists/iterations")

        return suggestions
