# src/worker/analyzer/base_analyzer.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Optional
from ...utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BaseAnalysisResult:
    """Base class for analysis results across all languages"""
    score: float  # Quality score (0-10)
    issues: List[Dict]  # List of detected issues
    suggestions: List[str]  # Improvement suggestions
    files_analyzed: List[str]  # List of analyzed files
    error_count: int = 0  # Number of errors found
    warning_count: int = 0  # Number of warnings found

    def to_dict(self) -> Dict:
        """Convert analysis result to dictionary format"""
        return {
            "score": self.score,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "files_analyzed": self.files_analyzed,
            "error_count": self.error_count,
            "warning_count": self.warning_count
        }


class BaseAnalyzer(ABC):
    """Base class for all language-specific code analyzers"""

    def __init__(self):
        self.logger = logger

    @abstractmethod
    async def analyze_project(self, project_dir: Path) -> BaseAnalysisResult:
        """
        Analyze an entire project directory.

        Args:
            project_dir: Path to project root directory

        Returns:
            BaseAnalysisResult containing analysis findings
        """
        pass

    @abstractmethod
    async def analyze_file(self, file_path: Path) -> Dict:
        """
        Analyze a single file.

        Args:
            file_path: Path to the file to analyze

        Returns:
            Dictionary containing analysis results for the file
        """
        pass

    @abstractmethod
    async def fix_issues(self, project_dir: Path, issues: List[Dict]) -> bool:
        """
        Attempt to automatically fix identified issues.

        Args:
            project_dir: Path to project root directory
            issues: List of issues to fix

        Returns:
            bool indicating whether fixes were successfully applied
        """
        pass

    @abstractmethod
    async def validate_dependencies(self, project_dir: Path) -> Dict[str, List[str]]:
        """
        Validate project dependencies.

        Args:
            project_dir: Path to project root directory

        Returns:
            Dictionary containing missing and unused dependencies
        """
        pass

    async def get_language_files(self, project_dir: Path) -> List[Path]:
        """
        Get list of relevant files for the specific language.

        Args:
            project_dir: Path to project root directory

        Returns:
            List of file paths to analyze
        """
        extensions = self.get_file_extensions()
        files = []

        try:
            for ext in extensions:
                files.extend(project_dir.rglob(f"*{ext}"))
        except Exception as e:
            self.logger.error(f"Error finding language files: {e}")

        return files

    @abstractmethod
    def get_file_extensions(self) -> List[str]:
        """
        Get list of file extensions for the specific language.

        Returns:
            List of file extensions (e.g., ['.py', '.pyw'])
        """
        pass

    def _calculate_score(self, issues: List[Dict]) -> float:
        """
        Calculate quality score based on issues.

        Args:
            issues: List of detected issues

        Returns:
            Float score between 0 and 10
        """
        if not issues:
            return 10.0

        # Count issues by severity
        error_count = sum(1 for issue in issues if issue.get('severity') == 'error')
        warning_count = sum(1 for issue in issues if issue.get('severity') == 'warning')

        # Calculate score (errors count more than warnings)
        score = 10.0 - (error_count * 1.0) - (warning_count * 0.5)
        return max(0.0, score)

    def _format_issue(
            self,
            code: str,
            message: str,
            file: str,
            line: Optional[int] = None,
            column: Optional[int] = None,
            severity: str = "warning"
    ) -> Dict:
        """
        Format an issue in a standardized way.

        Args:
            code: Issue code/identifier
            message: Issue description
            file: Path to file containing issue
            line: Line number of issue
            column: Column number of issue
            severity: Issue severity (error/warning/info)

        Returns:
            Dictionary containing formatted issue details
        """
        return {
            "code": code,
            "message": message,
            "file": str(file),
            "line": line,
            "column": column,
            "severity": severity
        }