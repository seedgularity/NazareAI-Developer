from pathlib import Path
from typing import Dict, List, Set, Optional
import re

from .base_analyzer import BaseAnalyzer, BaseAnalysisResult
from ...llm.providers.base_provider import BaseProvider
from ...utils.logger import get_logger

logger = get_logger(__name__)


class SolidityAnalysisResult(BaseAnalysisResult):
    """Solidity-specific analysis results"""

    def __init__(
            self,
            score: float,
            issues: List[Dict],
            suggestions: List[str],
            files_analyzed: List[str],
            gas_issues: List[Dict],
            security_issues: List[Dict],
            dependencies: Set[str],
            llm_insights: Dict[str, any] = None
    ):
        super().__init__(
            score=score,
            issues=issues,
            suggestions=suggestions,
            files_analyzed=files_analyzed
        )
        self.gas_issues = gas_issues
        self.security_issues = security_issues
        self.dependencies = dependencies
        self.llm_insights = llm_insights or {}

    def to_dict(self) -> Dict:
        base_dict = super().to_dict()
        base_dict.update({
            "gas_issues": self.gas_issues,
            "security_issues": self.security_issues,
            "dependencies": list(self.dependencies),
            "llm_insights": self.llm_insights
        })
        return base_dict


class SolidityAnalyzer(BaseAnalyzer):
    """Solidity smart contract analyzer using static analysis and LLM"""

    def __init__(self, primary_llm: BaseProvider, backup_llm: Optional[BaseProvider] = None):
        super().__init__()
        self.primary_llm = primary_llm
        self.backup_llm = backup_llm
        self.logger = logger

    async def analyze_project(self, project_dir: Path) -> SolidityAnalysisResult:
        """Analyze entire Solidity project"""
        try:
            all_issues = []
            gas_issues = []
            security_issues = []
            files_analyzed = []
            total_score = 0
            file_count = 0
            dependencies = set()

            # Get all Solidity files
            sol_files = await self.get_language_files(project_dir)

            # LLM insights storage
            llm_insights = {
                "project_level": {},
                "file_level": {},
                "suggestions": []
            }

            # Analyze each file
            for file_path in sol_files:
                # Static analysis
                file_result = await self.analyze_file(file_path)
                all_issues.extend(file_result.get("issues", []))
                gas_issues.extend(file_result.get("gas_issues", []))
                security_issues.extend(file_result.get("security_issues", []))
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

            return SolidityAnalysisResult(
                score=avg_score,
                issues=all_issues,
                suggestions=self._generate_suggestions(all_issues, gas_issues, security_issues),
                files_analyzed=files_analyzed,
                gas_issues=gas_issues,
                security_issues=security_issues,
                dependencies=dependencies,
                llm_insights=llm_insights
            )

        except Exception as e:
            self.logger.error(f"Error analyzing Solidity project: {e}")
            raise

    async def analyze_file(self, file_path: Path) -> Dict:
        """Analyze a single Solidity file"""
        try:
            issues = []
            gas_issues = []
            security_issues = []
            score = 10.0

            with open(file_path, 'r') as f:
                content = f.read()

            # Basic static analysis
            issues.extend(self._check_basic_issues(content))

            # Gas optimization checks
            gas_issues.extend(self._check_gas_issues(content))

            # Security checks
            security_issues.extend(self._check_security_issues(content))

            # Calculate score based on issues
            score -= len(issues) * 0.3
            score -= len(gas_issues) * 0.2
            score -= len(security_issues) * 0.5

            return {
                "score": max(0.0, score),
                "issues": issues,
                "gas_issues": gas_issues,
                "security_issues": security_issues
            }

        except Exception as e:
            self.logger.error(f"Error analyzing file {file_path}: {e}")
            return {"score": 0, "issues": [], "gas_issues": [], "security_issues": []}

    def _check_basic_issues(self, content: str) -> List[Dict]:
        """Check for basic Solidity issues"""
        issues = []

        # Check pragma version
        if not re.search(r'pragma solidity \^0\.[0-9]+\.[0-9]+;', content):
            issues.append({
                "code": "invalid-pragma",
                "message": "Missing or invalid pragma directive",
                "severity": "error"
            })

        # Check contract name matches file name
        contract_match = re.search(r'contract\s+(\w+)', content)
        if contract_match:
            contract_name = contract_match.group(1)
            if not content.lower().endswith(f"{contract_name.lower()}.sol"):
                issues.append({
                    "code": "contract-file-name",
                    "message": "Contract name should match file name",
                    "severity": "warning"
                })

        return issues

    def _check_gas_issues(self, content: str) -> List[Dict]:
        """Check for gas optimization issues"""
        issues = []

        # Check for unoptimized storage
        if re.search(r'mapping\s*\([^)]+\)\s+public', content):
            issues.append({
                "code": "gas-public-mapping",
                "message": "Public mappings auto-generate getter functions, consider using private/internal",
                "severity": "warning"
            })

        # Check for unnecessary SLOAD
        if re.search(r'for\s*\([^;]+;\s*[^;]+;\s*[^)]+\)\s*{[^}]*storage', content):
            issues.append({
                "code": "gas-storage-loop",
                "message": "Consider caching storage variables outside loops",
                "severity": "warning"
            })

        return issues

    def _check_security_issues(self, content: str) -> List[Dict]:
        """Check for security issues"""
        issues = []

        # Check for reentrancy vulnerabilities
        if 'call{value:' in content and not 'ReentrancyGuard' in content:
            issues.append({
                "code": "security-reentrancy",
                "message": "Potential reentrancy vulnerability detected",
                "severity": "high"
            })

        # Check for unchecked return values
        if '.call{' in content and not re.search(r'require\([^)]*\.call', content):
            issues.append({
                "code": "security-unchecked-call",
                "message": "Unchecked return value from low-level call",
                "severity": "high"
            })

        return issues

    def get_file_extensions(self) -> List[str]:
        """Get list of file extensions this analyzer can handle"""
        return [".sol"]

    async def fix_issues(self, project_dir: Path, issues: List[Dict]) -> bool:
        """Fix Solidity-specific issues"""
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
                        if issue['code'] == 'gas-optimization':
                            # Replace uint256 with uint8/uint16 where appropriate
                            modified_content = re.sub(
                                r'uint256\s+(\w+)\s*=\s*(\d+)',
                                lambda m: f'uint8 {m.group(1)} = {m.group(2)}' if int(m.group(2)) < 256 else m.group(0),
                                modified_content
                            )
                        
                        elif issue['code'] == 'reentrancy':
                            # Add ReentrancyGuard
                            if 'contract' in modified_content and 'ReentrancyGuard' not in modified_content:
                                modified_content = re.sub(
                                    r'contract\s+(\w+)\s*{',
                                    r'import "@openzeppelin/contracts/security/ReentrancyGuard.sol";\n\ncontract \1 is ReentrancyGuard {',
                                    modified_content
                                )
                        
                        elif issue['code'] == 'unchecked-return':
                            # Add return value checks
                            modified_content = re.sub(
                                r'(\w+)\.transfer\((.*?)\)',
                                r'require(\1.transfer(\2), "Transfer failed")',
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
            self.logger.error(f"Error fixing Solidity issues: {e}")
            return False

    async def validate_dependencies(self, project_dir: Path) -> Dict[str, List[str]]:
        """Validate Solidity project dependencies"""
        try:
            # Check for package.json (if using Hardhat/Truffle)
            pkg_json = project_dir / "package.json"
            if pkg_json.exists():
                with open(pkg_json) as f:
                    pkg_data = json.load(f)
                dependencies = set(pkg_data.get("dependencies", {}).keys())
                dev_dependencies = set(pkg_data.get("devDependencies", {}).keys())
            else:
                dependencies = set()
                dev_dependencies = set()

            # Check for imported contracts
            imported_contracts = set()
            for file_path in await self.get_language_files(project_dir):
                with open(file_path) as f:
                    content = f.read()
                    imports = re.findall(r'import\s+[\'"]([^\'"]+)[\'"]', content)
                    imported_contracts.update(imports)

            return {
                "missing": [],  # Determine missing deps based on imports
                "unused": [],   # Determine unused deps
                "present": list(dependencies | dev_dependencies)
            }

        except Exception as e:
            self.logger.error(f"Error validating dependencies: {e}")
            return {"missing": [], "unused": [], "present": []}

    async def _get_llm_analysis(self, file_path: Path, content: str) -> Dict:
        """Get code analysis from LLM"""
        try:
            prompt = f"""Analyze this Solidity smart contract for quality and security:

            File: {file_path.name}

            ```solidity
            {content}
            ```

            Analyze for:
            1. Security vulnerabilities
            2. Gas optimization opportunities
            3. Best practice violations
            4. Potential bugs
            5. Upgradeability concerns
            6. Access control issues
            7. Event emission completeness
            8. Input validation

            Format each issue as:
            - Location (line number if possible)
            - Severity (high/medium/low)
            - Description
            - Suggested fix"""

            try:
                response = await self.primary_llm.analyze_code(content, "solidity")
            except Exception as e:
                if self.backup_llm:
                    response = await self.backup_llm.analyze_code(content, "solidity")
                else:
                    raise

            return response

        except Exception as e:
            self.logger.error(f"LLM analysis failed: {e}")
            return {}

    def _generate_suggestions(self, issues: List[Dict], gas_issues: List[Dict], security_issues: List[Dict]) -> List[str]:
        """Generate improvement suggestions based on analysis results"""
        suggestions = []

        # Add security-related suggestions
        if any(issue["severity"] == "high" for issue in security_issues):
            suggestions.append("Critical security issues found - address before deployment")

        # Add gas optimization suggestions
        if gas_issues:
            suggestions.append("Consider implementing suggested gas optimizations")

        # Add general improvement suggestions
        if issues:
            suggestions.append("Review and address code quality issues")

        return suggestions
