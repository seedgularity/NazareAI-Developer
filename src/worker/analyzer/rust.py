from pathlib import Path
from typing import Dict, List, Set, Optional
import re
import toml

from .base_analyzer import BaseAnalyzer, BaseAnalysisResult
from ...llm.providers.base_provider import BaseProvider
from ...utils.logger import get_logger

logger = get_logger(__name__)


class RustAnalysisResult(BaseAnalysisResult):
    """Rust-specific analysis results"""

    def __init__(
            self,
            score: float,
            issues: List[Dict],
            suggestions: List[str],
            files_analyzed: List[str],
            dependencies: Set[str],
            unsafe_blocks: List[Dict],
            lifetime_issues: List[Dict],
            clippy_warnings: List[Dict],
            llm_insights: Dict[str, any] = None
    ):
        super().__init__(
            score=score,
            issues=issues,
            suggestions=suggestions,
            files_analyzed=files_analyzed
        )
        self.dependencies = dependencies
        self.unsafe_blocks = unsafe_blocks
        self.lifetime_issues = lifetime_issues
        self.clippy_warnings = clippy_warnings
        self.llm_insights = llm_insights or {}

    def to_dict(self) -> Dict:
        base_dict = super().to_dict()
        base_dict.update({
            "dependencies": list(self.dependencies),
            "unsafe_blocks": self.unsafe_blocks,
            "lifetime_issues": self.lifetime_issues,
            "clippy_warnings": self.clippy_warnings,
            "llm_insights": self.llm_insights
        })
        return base_dict


class RustAnalyzer(BaseAnalyzer):
    """Rust code analyzer using static analysis and LLM"""

    def __init__(self, primary_llm: BaseProvider, backup_llm: Optional[BaseProvider] = None):
        super().__init__()
        self.primary_llm = primary_llm
        self.backup_llm = backup_llm
        self.logger = logger

    async def analyze_project(self, project_dir: Path) -> RustAnalysisResult:
        """Analyze entire Rust project"""
        try:
            all_issues = []
            files_analyzed = []
            total_score = 0
            file_count = 0
            dependencies = set()
            unsafe_blocks = []
            lifetime_issues = []
            clippy_warnings = []

            # Parse Cargo.toml for dependencies
            cargo_toml = project_dir / "Cargo.toml"
            if cargo_toml.exists():
                try:
                    with open(cargo_toml) as f:
                        cargo_data = toml.load(f)
                        dependencies.update(cargo_data.get("dependencies", {}).keys())
                except Exception as e:
                    self.logger.error(f"Error parsing Cargo.toml: {e}")

            # Analyze each Rust file
            for file_path in await self.get_language_files(project_dir):
                if "target" not in str(file_path):  # Skip build directory
                    try:
                        file_result = await self.analyze_file(file_path)
                        all_issues.extend(file_result.get("issues", []))
                        unsafe_blocks.extend(file_result.get("unsafe_blocks", []))
                        lifetime_issues.extend(file_result.get("lifetime_issues", []))
                        clippy_warnings.extend(file_result.get("clippy_warnings", []))
                        
                        files_analyzed.append(str(file_path))
                        total_score += file_result.get("score", 0)
                        file_count += 1
                    except Exception as e:
                        self.logger.error(f"Error analyzing file {file_path}: {e}")

            # Calculate final score
            avg_score = total_score / file_count if file_count > 0 else 0

            return RustAnalysisResult(
                score=avg_score,
                issues=all_issues,
                suggestions=self._generate_suggestions(all_issues, unsafe_blocks, lifetime_issues),
                files_analyzed=files_analyzed,
                dependencies=dependencies,
                unsafe_blocks=unsafe_blocks,
                lifetime_issues=lifetime_issues,
                clippy_warnings=clippy_warnings,
                llm_insights={}
            )

        except Exception as e:
            self.logger.error(f"Error analyzing Rust project: {e}")
            # Return a basic result instead of raising
            return RustAnalysisResult(
                score=5.0,
                issues=[],
                suggestions=["Unable to complete full analysis"],
                files_analyzed=[],
                dependencies=set(),
                unsafe_blocks=[],
                lifetime_issues=[],
                clippy_warnings=[],
                llm_insights={}
            )

    async def analyze_file(self, file_path: Path) -> Dict:
        """Analyze a single Rust file"""
        try:
            issues = []
            score = 10.0
            unsafe_blocks = []
            lifetime_issues = []
            clippy_warnings = []

            with open(file_path, 'r') as f:
                content = f.read()

            # Basic static analysis
            issues.extend(self._check_basic_issues(content))
            unsafe_blocks.extend(self._check_unsafe_blocks(content))
            lifetime_issues.extend(self._check_lifetime_issues(content))

            # Adjust score based on issues
            score -= len(issues) * 0.5
            score -= len(unsafe_blocks) * 0.3
            score -= len(lifetime_issues) * 0.2

            return {
                "score": max(0.0, score),
                "issues": issues,
                "unsafe_blocks": unsafe_blocks,
                "lifetime_issues": lifetime_issues,
                "clippy_warnings": clippy_warnings
            }

        except Exception as e:
            self.logger.error(f"Error analyzing Rust file {file_path}: {e}")
            return {"score": 0, "issues": [], "unsafe_blocks": [], "lifetime_issues": [], "clippy_warnings": []}

    def _check_basic_issues(self, content: str) -> List[Dict]:
        """Check for basic Rust issues"""
        issues = []

        try:
            # Check for println! debugging
            if 'println!' in content:
                issues.append({
                    "code": "no-println",
                    "message": "Avoid using println! in production code",
                    "severity": "warning"
                })

            # Check for unwrap usage
            if re.search(r'\.unwrap\(\)', content):
                issues.append({
                    "code": "no-unwrap",
                    "message": "Avoid using .unwrap(), handle errors explicitly",
                    "severity": "warning"
                })

            # Check for panic! usage
            if 'panic!' in content:
                issues.append({
                    "code": "no-panic",
                    "message": "Avoid using panic!, handle errors properly",
                    "severity": "warning"
                })

        except re.error as e:
            self.logger.error(f"Regex error in basic issues check: {e}")

        return issues

    def _check_unsafe_blocks(self, content: str) -> List[Dict]:
        """Check for unsafe blocks and their context"""
        unsafe_blocks = []

        try:
            # Find unsafe blocks
            unsafe_matches = re.finditer(r'unsafe\s*{([^}]+)}', content, re.DOTALL)
            for match in unsafe_matches:
                unsafe_blocks.append({
                    "code": "unsafe-block",
                    "content": match.group(1).strip(),
                    "message": "Unsafe block detected, ensure it's necessary",
                    "severity": "warning"
                })

        except re.error as e:
            self.logger.error(f"Regex error in unsafe block check: {e}")

        return unsafe_blocks

    def _check_lifetime_issues(self, content: str) -> List[Dict]:
        """Check for potential lifetime issues"""
        issues = []

        try:
            # Check for multiple lifetimes
            lifetime_matches = re.findall(r"'[a-z]+", content)
            if len(set(lifetime_matches)) > 2:
                issues.append({
                    "code": "complex-lifetimes",
                    "message": "Complex lifetime usage detected, consider simplifying",
                    "severity": "info"
                })

            # Check for 'static lifetime usage
            if "'static" in content:
                issues.append({
                    "code": "static-lifetime",
                    "message": "Static lifetime usage detected, ensure it's necessary",
                    "severity": "info"
                })

        except re.error as e:
            self.logger.error(f"Regex error in lifetime check: {e}")

        return issues

    def _generate_suggestions(self, issues: List[Dict], unsafe_blocks: List[Dict], lifetime_issues: List[Dict]) -> List[str]:
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
        if issue_types.get('no-println', 0) > 0:
            suggestions.append("Replace println! with proper logging")

        if issue_types.get('no-unwrap', 0) > 0:
            suggestions.append("Use proper error handling instead of .unwrap()")

        if issue_types.get('no-panic', 0) > 0:
            suggestions.append("Implement proper error handling instead of panic!()")

        if unsafe_blocks:
            suggestions.append(f"Review {len(unsafe_blocks)} unsafe blocks for necessity")

        if lifetime_issues:
            suggestions.append("Consider simplifying lifetime annotations")

        return suggestions

    def get_file_extensions(self) -> List[str]:
        """Get list of file extensions this analyzer can handle"""
        return [".rs"] 

    async def fix_issues(self, project_dir: Path, issues: List[Dict]) -> bool:
        """Fix common Rust issues automatically"""
        try:
            files_modified = False
            
            # Group issues by file
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
                        if issue['code'] == 'no-println':
                            # Replace println! with log::info!
                            modified_content = re.sub(
                                r'println!\((.*)\)',
                                r'log::info!(\1)',
                                modified_content
                            )
                            # Add log dependency if not present
                            await self._ensure_log_dependency(project_dir)
                            
                        elif issue['code'] == 'no-unwrap':
                            # Replace unwrap with proper error handling
                            modified_content = re.sub(
                                r'\.unwrap\(\)',
                                r'.expect("Failed to unwrap value")',  # Basic fix, could be smarter
                                modified_content
                            )
                            
                        elif issue['code'] == 'no-panic':
                            # Replace panic! with Result return
                            modified_content = re.sub(
                                r'panic!\("([^"]*)"\)',
                                r'return Err(anyhow::anyhow!("\1"))',
                                modified_content
                            )
                            # Add anyhow dependency if not present
                            await self._ensure_anyhow_dependency(project_dir)

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
            self.logger.error(f"Error fixing Rust issues: {e}")
            return False

    async def _ensure_log_dependency(self, project_dir: Path) -> None:
        """Ensure the log crate is in dependencies"""
        cargo_toml = project_dir / "Cargo.toml"
        try:
            with open(cargo_toml) as f:
                cargo_data = toml.load(f)
            
            dependencies = cargo_data.get("dependencies", {})
            if "log" not in dependencies:
                dependencies["log"] = "0.4"
                cargo_data["dependencies"] = dependencies
                
                with open(cargo_toml, 'w') as f:
                    toml.dump(cargo_data, f)
                
                self.logger.info("Added log dependency to Cargo.toml")
        except Exception as e:
            self.logger.error(f"Error adding log dependency: {e}")

    async def _ensure_anyhow_dependency(self, project_dir: Path) -> None:
        """Ensure the anyhow crate is in dependencies"""
        cargo_toml = project_dir / "Cargo.toml"
        try:
            with open(cargo_toml) as f:
                cargo_data = toml.load(f)
            
            dependencies = cargo_data.get("dependencies", {})
            if "anyhow" not in dependencies:
                dependencies["anyhow"] = "1.0"
                cargo_data["dependencies"] = dependencies
                
                with open(cargo_toml, 'w') as f:
                    toml.dump(cargo_data, f)
                
                self.logger.info("Added anyhow dependency to Cargo.toml")
        except Exception as e:
            self.logger.error(f"Error adding anyhow dependency: {e}") 