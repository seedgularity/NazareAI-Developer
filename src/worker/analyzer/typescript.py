from pathlib import Path
from typing import Dict, List, Set, Optional
import re

from .javascript import JavaScriptAnalyzer, JavaScriptAnalysisResult
from ...llm.providers.base_provider import BaseProvider
from ...utils.logger import get_logger

logger = get_logger(__name__)


class TypeScriptAnalysisResult(JavaScriptAnalysisResult):
    """TypeScript-specific analysis results"""

    def __init__(
            self,
            score: float,
            issues: List[Dict],
            suggestions: List[str],
            files_analyzed: List[str],
            dependencies: Set[str],
            dev_dependencies: Set[str],
            type_issues: List[Dict] = None,
            type_coverage: float = 0.0,
            interface_issues: List[Dict] = None,
            llm_insights: Dict[str, any] = None
    ):
        super().__init__(
            score=score,
            issues=issues,
            suggestions=suggestions,
            files_analyzed=files_analyzed,
            dependencies=dependencies,
            dev_dependencies=dev_dependencies,
            type_issues=type_issues,
            llm_insights=llm_insights
        )
        self.type_coverage = type_coverage
        self.interface_issues = interface_issues or []

    def to_dict(self) -> Dict:
        base_dict = super().to_dict()
        base_dict.update({
            "type_coverage": self.type_coverage,
            "interface_issues": self.interface_issues
        })
        return base_dict


class TypeScriptAnalyzer(JavaScriptAnalyzer):
    """TypeScript code analyzer with enhanced type checking"""

    def __init__(self, primary_llm: BaseProvider, backup_llm: Optional[BaseProvider] = None):
        super().__init__(primary_llm, backup_llm)

    async def analyze_project(self, project_dir: Path) -> TypeScriptAnalysisResult:
        """Analyze entire TypeScript project"""
        try:
            # Initialize result variables
            all_issues = []
            files_analyzed = []
            total_score = 0
            file_count = 0
            type_coverage = 0.0
            interface_issues = []

            # Get base analysis from JavaScript analyzer
            try:
                js_result = await super().analyze_project(project_dir)
            except Exception as e:
                self.logger.error(f"Base analysis failed: {e}")
                # Create empty JavaScript result if base analysis fails
                js_result = JavaScriptAnalysisResult(
                    score=5.0,  # Default middle score
                    issues=[],
                    suggestions=[],
                    files_analyzed=[],
                    dependencies=set(),
                    dev_dependencies=set(),
                    type_issues=[]
                )

            # Additional TypeScript-specific analysis
            try:
                type_coverage = await self._calculate_type_coverage(project_dir)
            except Exception as e:
                self.logger.error(f"Type coverage calculation failed: {e}")
                type_coverage = 0.0

            try:
                interface_issues = await self._check_interface_issues(project_dir)
            except Exception as e:
                self.logger.error(f"Interface analysis failed: {e}")
                interface_issues = []

            # Convert to TypeScript result
            return TypeScriptAnalysisResult(
                score=js_result.score,
                issues=js_result.issues,
                suggestions=js_result.suggestions,
                files_analyzed=js_result.files_analyzed,
                dependencies=js_result.dependencies,
                dev_dependencies=js_result.dev_dependencies,
                type_issues=js_result.type_issues,
                type_coverage=type_coverage,
                interface_issues=interface_issues,
                llm_insights=js_result.llm_insights
            )

        except Exception as e:
            self.logger.error(f"Error analyzing TypeScript project: {e}")
            # Return a basic result instead of raising
            return TypeScriptAnalysisResult(
                score=5.0,
                issues=[],
                suggestions=["Unable to complete full analysis"],
                files_analyzed=[],
                dependencies=set(),
                dev_dependencies=set(),
                type_issues=[],
                type_coverage=0.0,
                interface_issues=[],
                llm_insights={}
            )

    async def analyze_file(self, file_path: Path) -> Dict:
        """Analyze a single TypeScript file"""
        try:
            # Get base analysis from JavaScript analyzer
            result = await super().analyze_file(file_path)

            # Add TypeScript-specific checks
            with open(file_path, 'r') as f:
                content = f.read()

            # Check for TypeScript-specific issues
            ts_issues = self._check_typescript_specific_issues(content)
            result["issues"].extend(ts_issues)

            # Adjust score based on TypeScript-specific issues
            result["score"] = max(0.0, result["score"] - (len(ts_issues) * 0.2))

            return result

        except Exception as e:
            self.logger.error(f"Error analyzing TypeScript file {file_path}: {e}")
            return {"score": 0, "issues": []}

    def _check_typescript_specific_issues(self, content: str) -> List[Dict]:
        """Check for TypeScript-specific issues"""
        issues = []

        try:
            # Check for proper interface naming - updated regex
            interface_matches = re.finditer(r'\binterface\s+(\w+)', content)
            for match in interface_matches:
                interface_name = match.group(1)
                if not interface_name.startswith('I'):
                    issues.append({
                        "code": "interface-naming",
                        "message": f"Interface {interface_name} should start with 'I'",
                        "severity": "info"
                    })

            # Check for type assertions - updated regex
            if re.search(r'\bas\s+[A-Z]\w+', content):
                issues.append({
                    "code": "type-assertion",
                    "message": "Consider using type guards instead of type assertions",
                    "severity": "warning"
                })

            # Check for proper enum usage - updated regex
            enum_matches = re.finditer(r'\benum\s+(\w+)', content)
            for match in enum_matches:
                enum_name = match.group(1)
                if not enum_name.endswith('Enum'):
                    issues.append({
                        "code": "enum-naming",
                        "message": f"Enum {enum_name} should end with 'Enum'",
                        "severity": "info"
                    })

            # Check for any usage in generics - updated regex
            if re.search(r'<\s*any\s*>', content):
                issues.append({
                    "code": "no-any-generics",
                    "message": "Avoid using 'any' in generic type parameters",
                    "severity": "warning"
                })

        except re.error as e:
            self.logger.error(f"Regex error in TypeScript analysis: {e}")

        return issues

    async def _calculate_type_coverage(self, project_dir: Path) -> float:
        """Calculate type coverage percentage"""
        try:
            total_lines = 0
            typed_lines = 0

            for file_path in await self.get_language_files(project_dir):
                if "node_modules" not in str(file_path):
                    with open(file_path, 'r') as f:
                        content = f.read()
                        lines = content.split('\n')
                        total_lines += len(lines)

                        # Updated regex for type annotations
                        try:
                            typed_lines += sum(
                                1 for line in lines
                                if re.search(r':\s*([A-Z]\w+|string|number|boolean|any|void|never|object|unknown)', line)
                            )
                        except re.error:
                            self.logger.warning(f"Regex error in type coverage calculation for {file_path}")

            return (typed_lines / total_lines * 100) if total_lines > 0 else 0

        except Exception as e:
            self.logger.error(f"Error calculating type coverage: {e}")
            return 0.0

    async def _check_interface_issues(self, project_dir: Path) -> List[Dict]:
        """Check for interface-related issues"""
        issues = []

        try:
            # Collect all interfaces
            interfaces = {}
            for file_path in await self.get_language_files(project_dir):
                if "node_modules" not in str(file_path):
                    with open(file_path, 'r') as f:
                        content = f.read()
                        # Updated regex for interface definitions
                        try:
                            for match in re.finditer(r'\binterface\s+(\w+)\s*{([^}]+)}', content, re.DOTALL):
                                interface_name = match.group(1)
                                interface_content = match.group(2)
                                interfaces[interface_name] = interface_content
                        except re.error:
                            self.logger.warning(f"Regex error in interface parsing for {file_path}")

            # Check for interface consistency
            for name, content in interfaces.items():
                try:
                    # Check for missing property types
                    if re.search(r':\s*any\b', content):
                        issues.append({
                            "code": "interface-any",
                            "message": f"Interface {name} contains 'any' types",
                            "severity": "warning"
                        })

                    # Check for optional properties consistency
                    optional_props = re.findall(r'\b\w+\?:', content)
                    required_props = re.findall(r'\b\w+:', content)
                    if optional_props and len(optional_props) == len(required_props):
                        issues.append({
                            "code": "all-optional",
                            "message": f"All properties in {name} are optional",
                            "severity": "warning"
                        })
                except re.error:
                    self.logger.warning(f"Regex error in interface analysis for {name}")

        except Exception as e:
            self.logger.error(f"Error checking interface issues: {e}")

        return issues

    def get_file_extensions(self) -> List[str]:
        """Get list of file extensions this analyzer can handle"""
        return [".ts", ".tsx"]

    def _generate_suggestions(self, issues: List[Dict]) -> List[str]:
        """Generate TypeScript-specific improvement suggestions"""
        suggestions = super()._generate_suggestions(issues)

        # Add TypeScript-specific suggestions
        issue_types = {}
        for issue in issues:
            issue_type = issue.get('code', '')
            if issue_type not in issue_types:
                issue_types[issue_type] = 0
            issue_types[issue_type] += 1

        if issue_types.get('interface-any', 0) > 0:
            suggestions.append("Replace 'any' types with more specific interfaces")

        if issue_types.get('type-assertion', 0) > 0:
            suggestions.append("Use type guards instead of type assertions for better type safety")

        if issue_types.get('no-any-generics', 0) > 0:
            suggestions.append("Specify concrete types for generic type parameters")

        return suggestions 

    async def fix_issues(self, project_dir: Path, issues: List[Dict]) -> bool:
        """Fix TypeScript-specific issues"""
        try:
            files_modified = False
            issues_by_file = {}
            
            # Group issues by file
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
                        if issue['code'] == 'interface-any':
                            # Replace 'any' types with 'unknown'
                            modified_content = re.sub(
                                r':\s*any\b',
                                r': unknown',
                                modified_content
                            )
                        
                        elif issue['code'] == 'type-assertion':
                            # Replace type assertions with type guards
                            modified_content = re.sub(
                                r'as\s+([A-Z]\w+)',
                                lambda m: f'/* TODO: Replace with type guard for {m.group(1)} */',
                                modified_content
                            )
                        
                        elif issue['code'] == 'no-any-generics':
                            # Add TODO comment for generic type parameters
                            modified_content = re.sub(
                                r'<\s*any\s*>',
                                '/* TODO: Specify concrete type */ <unknown>',
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
            self.logger.error(f"Error fixing TypeScript issues: {e}")
            return False 