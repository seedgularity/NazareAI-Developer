from pathlib import Path
from typing import Dict, List, Set, Optional
import re

from .base_analyzer import BaseAnalyzer, BaseAnalysisResult
from ...llm.providers.base_provider import BaseProvider
from ...utils.logger import get_logger

logger = get_logger(__name__)


class StyleAnalysisResult(BaseAnalysisResult):
    """CSS/Style-specific analysis results"""

    def __init__(
            self,
            score: float,
            issues: List[Dict],
            suggestions: List[str],
            files_analyzed: List[str],
            specificity_issues: List[Dict],
            performance_issues: List[Dict],
            unused_styles: Set[str],
            llm_insights: Dict[str, any] = None
    ):
        super().__init__(
            score=score,
            issues=issues,
            suggestions=suggestions,
            files_analyzed=files_analyzed
        )
        self.specificity_issues = specificity_issues
        self.performance_issues = performance_issues
        self.unused_styles = unused_styles
        self.llm_insights = llm_insights or {}

    def to_dict(self) -> Dict:
        base_dict = super().to_dict()
        base_dict.update({
            "specificity_issues": self.specificity_issues,
            "performance_issues": self.performance_issues,
            "unused_styles": list(self.unused_styles),
            "llm_insights": self.llm_insights
        })
        return base_dict


class StyleAnalyzer(BaseAnalyzer):
    """CSS/Style analyzer for CSS, SCSS, and styled-components"""

    def __init__(self, primary_llm: BaseProvider, backup_llm: Optional[BaseProvider] = None):
        super().__init__()
        self.primary_llm = primary_llm
        self.backup_llm = backup_llm
        self.logger = logger

    async def analyze_project(self, project_dir: Path) -> StyleAnalysisResult:
        """Analyze entire styling project"""
        try:
            all_issues = []
            specificity_issues = []
            performance_issues = []
            files_analyzed = []
            total_score = 0
            file_count = 0
            unused_styles = set()

            # Get all style files
            style_files = await self.get_language_files(project_dir)

            # LLM insights storage
            llm_insights = {
                "project_level": {},
                "file_level": {},
                "suggestions": []
            }

            # Analyze each file
            for file_path in style_files:
                if "node_modules" not in str(file_path):
                    # Static analysis
                    file_result = await self.analyze_file(file_path)
                    all_issues.extend(file_result.get("issues", []))
                    specificity_issues.extend(file_result.get("specificity_issues", []))
                    performance_issues.extend(file_result.get("performance_issues", []))
                    files_analyzed.append(str(file_path))
                    total_score += file_result.get("score", 0)
                    file_count += 1

                    # LLM analysis
                    with open(file_path, 'r') as f:
                        content = f.read()
                    llm_result = await self._get_llm_analysis(file_path, content)
                    if llm_result:
                        llm_insights["file_level"][str(file_path)] = llm_result

            # Calculate final results
            avg_score = total_score / file_count if file_count > 0 else 0

            # Find unused styles
            unused_styles = await self._find_unused_styles(project_dir)

            return StyleAnalysisResult(
                score=avg_score,
                issues=all_issues,
                suggestions=self._generate_suggestions(all_issues, specificity_issues, performance_issues, unused_styles),
                files_analyzed=files_analyzed,
                specificity_issues=specificity_issues,
                performance_issues=performance_issues,
                unused_styles=unused_styles,
                llm_insights=llm_insights
            )

        except Exception as e:
            self.logger.error(f"Error analyzing style project: {e}")
            raise

    async def analyze_file(self, file_path: Path) -> Dict:
        """Analyze a single style file"""
        try:
            issues = []
            specificity_issues = []
            performance_issues = []
            score = 10.0

            with open(file_path, 'r') as f:
                content = f.read()

            # Basic style analysis
            issues.extend(self._check_basic_issues(content))

            # Specificity checks
            specificity_issues.extend(self._check_specificity_issues(content))

            # Performance checks
            performance_issues.extend(self._check_performance_issues(content))

            # Additional checks based on file type
            if file_path.suffix == '.scss':
                issues.extend(self._check_scss_issues(content))
            elif 'styled-components' in content:
                issues.extend(self._check_styled_components_issues(content))

            # Calculate score based on issues
            score -= len(issues) * 0.3
            score -= len(specificity_issues) * 0.2
            score -= len(performance_issues) * 0.2

            return {
                "score": max(0.0, score),
                "issues": issues,
                "specificity_issues": specificity_issues,
                "performance_issues": performance_issues
            }

        except Exception as e:
            self.logger.error(f"Error analyzing file {file_path}: {e}")
            return {"score": 0, "issues": [], "specificity_issues": [], "performance_issues": []}

    def _check_basic_issues(self, content: str) -> List[Dict]:
        """Check for basic CSS issues"""
        issues = []

        # Check for !important usage
        important_matches = re.findall(r'!important', content)
        if important_matches:
            issues.append({
                "code": "no-important",
                "message": "Avoid using !important declarations",
                "severity": "warning",
                "count": len(important_matches)
            })

        # Check for vendor prefixes without standard property
        vendor_prefixes = ['-webkit-', '-moz-', '-ms-', '-o-']
        for prefix in vendor_prefixes:
            if prefix in content:
                issues.append({
                    "code": "vendor-prefix",
                    "message": f"Consider using autoprefixer instead of manual {prefix} prefixes",
                    "severity": "info"
                })

        # Check for potential color inconsistencies
        color_values = re.findall(r'#[0-9a-fA-F]{3,6}', content)
        if len(set(color_values)) > 10:
            issues.append({
                "code": "color-consistency",
                "message": "Consider using CSS variables for consistent colors",
                "severity": "warning"
            })

        return issues

    def _check_specificity_issues(self, content: str) -> List[Dict]:
        """Check for CSS specificity issues"""
        issues = []

        # Check for deep nesting
        if re.search(r'[^}]*{[^}]*{[^}]*{[^}]*}', content):
            issues.append({
                "code": "deep-nesting",
                "message": "Avoid deep nesting of selectors (max 3 levels recommended)",
                "severity": "warning"
            })

        # Check for overly specific selectors
        complex_selectors = re.findall(r'(?:\.|#)[^\s{]+(?:\s+(?:\.|#)[^\s{]+){3,}', content)
        for selector in complex_selectors:
            issues.append({
                "code": "high-specificity",
                "message": f"Overly specific selector: {selector}",
                "severity": "warning"
            })

        return issues

    def _check_performance_issues(self, content: str) -> List[Dict]:
        """Check for CSS performance issues"""
        issues = []

        # Check for universal selectors
        if re.search(r'\*\s*{', content):
            issues.append({
                "code": "universal-selector",
                "message": "Universal selectors can impact performance",
                "severity": "warning"
            })

        # Check for expensive properties
        expensive_props = ['box-shadow', 'transform', 'filter']
        for prop in expensive_props:
            if prop in content:
                issues.append({
                    "code": "expensive-property",
                    "message": f"Consider impact of {prop} on performance",
                    "severity": "info"
                })

        return issues

    def _check_scss_issues(self, content: str) -> List[Dict]:
        """Check for SCSS-specific issues"""
        issues = []

        # Check for nested extends
        if re.search(r'@extend[^;]+;[^}]*@extend', content):
            issues.append({
                "code": "nested-extends",
                "message": "Avoid multiple @extend within the same rule",
                "severity": "warning"
            })

        # Check for placeholder usage
        if not re.search(r'%[a-zA-Z]', content) and '@extend' in content:
            issues.append({
                "code": "missing-placeholder",
                "message": "Consider using placeholder selectors with @extend",
                "severity": "info"
            })

        return issues

    def _check_styled_components_issues(self, content: str) -> List[Dict]:
        """Check for styled-components specific issues"""
        issues = []

        # Check for props usage
        if re.search(r'\${props\s*=>\s*props\.[^}]+}', content):
            issues.append({
                "code": "props-destructuring",
                "message": "Consider destructuring props for cleaner syntax",
                "severity": "info"
            })

        # Check for theme usage
        if 'props.theme' in content and not re.search(r'ThemeProvider', content):
            issues.append({
                "code": "missing-theme-provider",
                "message": "Theme usage detected without ThemeProvider context",
                "severity": "warning"
            })

        return issues

    def get_file_extensions(self) -> List[str]:
        """Get list of file extensions this analyzer can handle"""
        return [".css", ".scss", ".sass", ".less", ".styled.js", ".styled.ts"]

    async def fix_issues(self, project_dir: Path, issues: List[Dict]) -> bool:
        """Fix CSS/SCSS style issues"""
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
                        if issue['code'] == 'specificity':
                            # Reduce selector specificity
                            modified_content = re.sub(
                                r'(#[\w-]+\s+)+([.[\w-]+)',
                                r'\2',
                                modified_content
                            )
                        
                        elif issue['code'] == 'unused-styles':
                            # Comment out unused selectors
                            for selector in issue.get('selectors', []):
                                modified_content = re.sub(
                                    f"({selector}[^{{]*{{[^}}]*}})",
                                    r"/* Unused: \1 */",
                                    modified_content
                                )
                        
                        elif issue['code'] == 'performance':
                            # Fix descendant selectors
                            modified_content = re.sub(
                                r'(\s+>?\s*\*\s*>?\s+)',
                                r' > ',
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
            self.logger.error(f"Error fixing style issues: {e}")
            return False

    async def validate_dependencies(self, project_dir: Path) -> Dict[str, List[str]]:
        """Validate style dependencies"""
        try:
            # Check package.json for style-related dependencies
            pkg_json = project_dir / "package.json"
            if pkg_json.exists():
                with open(pkg_json) as f:
                    pkg_data = json.load(f)
                dependencies = {
                    k: v for k, v in pkg_data.get("dependencies", {}).items()
                    if any(term in k for term in ['css', 'sass', 'less', 'styled', 'style'])
                }
                dev_dependencies = {
                    k: v for k, v in pkg_data.get("devDependencies", {}).items()
                    if any(term in k for term in ['css', 'sass', 'less', 'styled', 'style'])
                }
            else:
                dependencies = {}
                dev_dependencies = {}

            return {
                "missing": [],  # Determine based on imports/usage
                "unused": [],   # Determine based on imports/usage
                "present": list(dependencies.keys()) + list(dev_dependencies.keys())
            }

        except Exception as e:
            self.logger.error(f"Error validating dependencies: {e}")
            return {"missing": [], "unused": [], "present": []}

    async def _find_unused_styles(self, project_dir: Path) -> Set[str]:
        """Find unused CSS classes and selectors"""
        try:
            unused_styles = set()
            style_content = ""
            js_content = ""

            # Collect all CSS/SCSS content
            for style_file in await self.get_language_files(project_dir):
                with open(style_file, 'r') as f:
                    style_content += f.read() + "\n"

            # Collect all JS/TS content to check usage
            js_files = project_dir.rglob("*.[jt]s*")
            for js_file in js_files:
                if "node_modules" not in str(js_file):
                    with open(js_file, 'r') as f:
                        js_content += f.read() + "\n"

            # Extract class names from CSS
            class_names = re.findall(r'\.([a-zA-Z_-][a-zA-Z0-9_-]*)', style_content)

            # Check usage in JS/TS files
            for class_name in class_names:
                if class_name not in js_content:
                    unused_styles.add(class_name)

            return unused_styles

        except Exception as e:
            self.logger.error(f"Error finding unused styles: {e}")
            return set()

    async def _get_llm_analysis(self, file_path: Path, content: str) -> Dict:
        """Get code analysis from LLM"""
        try:
            prompt = f"""Analyze this CSS/style code for quality and improvements:

            File: {file_path.name}

            ```css
            {content}
            ```

            Analyze for:
            1. Maintainability issues
            2. Performance concerns
            3. Best practice violations
            4. Specificity problems
            5. Responsive design issues
            6. Browser compatibility
            7. Potential refactoring
            8. Design system consistency

            Format each issue as:
            - Location (line number if possible)
            - Severity (high/medium/low)
            - Description
            - Suggested fix"""

            try:
                response = await self.primary_llm.analyze_code(content, "css")
            except Exception as e:
                if self.backup_llm:
                    response = await self.backup_llm.analyze_code(content, "css")
                else:
                    raise

            return response

        except Exception as e:
            self.logger.error(f"LLM analysis failed: {e}")
            return {}

    def _generate_suggestions(self, issues: List[Dict], specificity_issues: List[Dict], 
                            performance_issues: List[Dict], unused_styles: Set[str]) -> List[str]:
        """Generate improvement suggestions based on analysis results"""
        suggestions = []

        # Add specificity-related suggestions
        if specificity_issues:
            suggestions.append("Consider reducing selector specificity for better maintainability")

        # Add performance-related suggestions
        if performance_issues:
            suggestions.append("Review and optimize performance-critical styles")

        # Add unused styles suggestions
        if unused_styles:
            suggestions.append(f"Remove {len(unused_styles)} unused CSS classes")

        # Add general improvement suggestions
        if issues:
            suggestions.append("Review and address style quality issues")

        return suggestions 