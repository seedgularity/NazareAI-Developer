from pathlib import Path
from typing import Dict, List, Set, Optional
import re
import json

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
            # First check project structure and dependencies
            structure_issues = await self._check_project_structure(project_dir)
            
            # Get static analysis results
            static_issues = []
            files_analyzed = []
            
            # Analyze each TypeScript file
            for file_path in await self.get_language_files(project_dir):
                if "node_modules" not in str(file_path):
                    try:
                        file_result = await self.analyze_file(file_path)
                        static_issues.extend(file_result.get("issues", []))
                        files_analyzed.append(str(file_path))
                    except Exception as e:
                        self.logger.error(f"Error analyzing {file_path}: {e}")
                        continue
            
            # Try to get LLM insights, but don't fail if it errors
            try:
                llm_insights = await self._get_llm_analysis(project_dir)
            except Exception as e:
                self.logger.warning(f"LLM analysis failed: {e}")
                llm_insights = {}
            
            # Combine all issues
            all_issues = static_issues + structure_issues
            
            return TypeScriptAnalysisResult(
                score=self._calculate_score(all_issues),
                issues=all_issues,
                suggestions=self._generate_suggestions(all_issues),
                files_analyzed=files_analyzed,
                dependencies=self._get_dependencies(project_dir),
                dev_dependencies=self._get_dev_dependencies(project_dir),
                type_issues=[i for i in all_issues if i["code"].startswith("type-")],
                type_coverage=await self._calculate_type_coverage(project_dir),
                interface_issues=[i for i in all_issues if i["code"].startswith("interface-")],
                llm_insights=llm_insights
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing TypeScript project: {e}")
            raise

    async def _check_project_structure(self, project_dir: Path) -> List[Dict]:
        """Check for missing files and configuration issues"""
        try:
            # First detect existing structure
            existing_structure = {
                "components": None,
                "styles": None,
                "pages": None,
                "public": None,
                "src": None
            }

            # Check if project uses src directory
            src_dir = project_dir / "src"
            uses_src = src_dir.exists() and any(src_dir.iterdir())

            # Check existing component locations
            if (project_dir / "components").exists():
                existing_structure["components"] = "root"
            elif uses_src and (src_dir / "components").exists():
                existing_structure["components"] = "src"

            # Check existing styles location
            if (project_dir / "styles").exists():
                existing_structure["styles"] = "root"
            elif uses_src and (src_dir / "styles").exists():
                existing_structure["styles"] = "src"

            # Check existing pages location
            if (project_dir / "pages").exists():
                existing_structure["pages"] = "root"
            elif uses_src and (src_dir / "pages").exists():
                existing_structure["pages"] = "src"

            issues = []
            
            # Check package.json
            package_json = project_dir / "package.json"
            if not package_json.exists():
                issues.append({
                    "code": "missing-package-json",
                    "message": "Missing package.json file",
                    "severity": "error",
                    "file": str(package_json),
                    "fix_type": "create",
                    "fix_content": self._generate_package_json()
                })
            else:
                with open(package_json) as f:
                    pkg_data = json.load(f)
                    # Check scripts
                    if not pkg_data.get("scripts"):
                        issues.append({
                            "code": "missing-scripts",
                            "message": "No scripts defined in package.json",
                            "severity": "error",
                            "file": str(package_json),
                            "fix_type": "update",
                            "fix_content": {"scripts": self._generate_default_scripts()}
                        })
                    # Check dependencies
                    missing_deps = self._check_required_dependencies(pkg_data)
                    if missing_deps:
                        issues.append({
                            "code": "missing-dependencies",
                            "message": f"Missing required dependencies: {', '.join(missing_deps)}",
                            "severity": "error",
                            "file": str(package_json),
                            "fix_type": "update",
                            "fix_content": {"dependencies": {dep: "latest" for dep in missing_deps}}
                        })

            # Check tsconfig.json
            tsconfig = project_dir / "tsconfig.json"
            if not tsconfig.exists():
                issues.append({
                    "code": "missing-tsconfig",
                    "message": "Missing tsconfig.json file",
                    "severity": "error",
                    "file": str(tsconfig),
                    "fix_type": "create",
                    "fix_content": self._generate_tsconfig()
                })

            # Create missing directories in the correct location
            essential_dirs = ["components", "styles", "pages", "public"]
            for dir_name in essential_dirs:
                # Determine correct location based on existing structure
                if existing_structure[dir_name] == "src":
                    dir_path = src_dir / dir_name
                elif existing_structure[dir_name] == "root":
                    dir_path = project_dir / dir_name
                else:
                    # If no existing structure, prefer src if it exists
                    dir_path = (src_dir if uses_src else project_dir) / dir_name

                if not dir_path.exists():
                    issues.append({
                        "code": "missing-directory",
                        "message": f"Missing {dir_name} directory",
                        "severity": "warning",
                        "file": str(dir_path),
                        "fix_type": "create_dir"
                    })

            # Check for essential files in correct locations
            essential_files = {}
            
            # Determine correct paths based on structure
            if uses_src:
                essential_files.update({
                    "src/index.tsx": self._generate_index_tsx,
                    "src/components/Layout.tsx": self._generate_layout_component,
                    "src/styles/globals.css": self._generate_global_styles
                })
            else:
                essential_files.update({
                    "pages/index.tsx": self._generate_index_tsx,
                    "components/Layout.tsx": self._generate_layout_component,
                    "styles/globals.css": self._generate_global_styles
                })

            # Common files regardless of structure
            essential_files.update({
                ".gitignore": self._generate_gitignore,
                "README.md": self._generate_readme,
                ".env": self._generate_env,
                "next.config.js": self._generate_next_config
            })
            
            for file_name, generator in essential_files.items():
                file_path = project_dir / file_name
                if not file_path.exists():
                    issues.append({
                        "code": "missing-file",
                        "message": f"Missing {file_name}",
                        "severity": "warning",
                        "file": str(file_path),
                        "fix_type": "create",
                        "fix_content": generator()
                    })

            return issues

        except Exception as e:
            self.logger.error(f"Error checking project structure: {e}")
            return []

    def _check_required_dependencies(self, pkg_data: Dict) -> List[str]:
        """Check for missing required dependencies"""
        required_deps = {
            "react": "latest",
            "react-dom": "latest",
            "next": "latest",
            "typescript": "latest",
            "@types/react": "latest",
            "@types/react-dom": "latest",
            "@types/node": "latest"
        }
        
        missing_deps = []
        existing_deps = {
            **pkg_data.get("dependencies", {}),
            **pkg_data.get("devDependencies", {})
        }
        
        for dep in required_deps:
            if dep not in existing_deps:
                missing_deps.append(dep)
                
        return missing_deps

    async def analyze_file(self, file_path: Path) -> Dict:
        """Analyze a single TypeScript file"""
        try:
            # Get base analysis from JavaScript analyzer
            result = await super().analyze_file(file_path)

            # Add TypeScript-specific checks
            with open(file_path, 'r') as f:
                content = f.read()

            # Pass file_path to the check function
            ts_issues = self._check_typescript_specific_issues(content, file_path)
            result["issues"].extend(ts_issues)

            # Adjust score based on TypeScript-specific issues
            result["score"] = max(0.0, result["score"] - (len(ts_issues) * 0.2))

            return result

        except Exception as e:
            self.logger.error(f"Error analyzing TypeScript file {file_path}: {e}")
            return {"score": 0, "issues": []}

    def _check_typescript_specific_issues(self, content: str, file_path: Path) -> List[Dict]:
        """Check for TypeScript-specific issues"""
        issues = []

        try:
            # Split content into lines for line number tracking
            lines = content.split('\n')
            
            # Check for proper interface naming
            for i, line in enumerate(lines, 1):
                interface_match = re.search(r'\binterface\s+(\w+)', line)
                if interface_match:
                    interface_name = interface_match.group(1)
                    if not interface_name.startswith('I'):
                        issues.append({
                            "code": "interface-naming",
                            "message": f"Interface {interface_name} should start with 'I'",
                            "severity": "info",
                            "file": str(file_path),
                            "line": i
                        })

            # Check for type assertions
            for i, line in enumerate(lines, 1):
                if re.search(r'\bas\s+[A-Z]\w+', line):
                    issues.append({
                        "code": "type-assertion",
                        "message": "Consider using type guards instead of type assertions",
                        "severity": "warning",
                        "file": str(file_path),
                        "line": i
                    })

            # Check for variable declarations
            for i, line in enumerate(lines, 1):
                var_match = re.search(r'\b(const|let|var)\s+(\w+)\s*(?:=|:)', line)
                if var_match:
                    var_name = var_match.group(2)
                    if not re.search(r':\s*\w+', line):  # No type annotation
                        issues.append({
                            "code": "missing-type",
                            "message": f"Variable {var_name} might be used before declaration",
                            "severity": "error",
                            "file": str(file_path),
                            "line": i
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
        """Fix TypeScript-specific issues including project structure"""
        try:
            files_modified = False
            
            # First handle structural issues
            for issue in issues:
                if issue.get("fix_type") == "create":
                    # Create new file
                    file_path = Path(issue["file"])
                    if not file_path.parent.exists():
                        file_path.parent.mkdir(parents=True)
                    
                    self.logger.info(f"Creating {file_path}")
                    with open(file_path, 'w') as f:
                        if isinstance(issue["fix_content"], dict):
                            json.dump(issue["fix_content"], f, indent=2)
                        else:
                            f.write(issue["fix_content"])
                    files_modified = True
                
                elif issue.get("fix_type") == "update":
                    # Update existing file
                    file_path = Path(issue["file"])
                    if file_path.exists():
                        with open(file_path) as f:
                            content = json.load(f)
                        
                        # Update content
                        content.update(issue["fix_content"])
                        
                        self.logger.info(f"Updating {file_path}")
                        with open(file_path, 'w') as f:
                            json.dump(content, f, indent=2)
                        files_modified = True
                
                elif issue.get("fix_type") == "create_dir":
                    # Create directory
                    dir_path = Path(issue["file"])
                    if not dir_path.exists():
                        self.logger.info(f"Creating directory {dir_path}")
                        dir_path.mkdir(parents=True)
                        files_modified = True

            # Then handle code issues
            code_fixes = await super().fix_issues(project_dir, [i for i in issues if "fix_type" not in i])
            
            return files_modified or code_fixes

        except Exception as e:
            self.logger.error(f"Error fixing TypeScript issues: {e}")
            return False

    def _infer_type_from_assignment(self, line: str) -> Optional[str]:
        """Try to infer TypeScript type from assignment value"""
        # Check for string literal
        if re.search(r'=\s*[\'"]', line):
            return "string"
        # Check for number
        elif re.search(r'=\s*\d+', line):
            return "number"
        # Check for boolean
        elif re.search(r'=\s*(true|false)\b', line):
            return "boolean"
        # Check for array
        elif re.search(r'=\s*\[', line):
            return "any[]"  # Could be more specific based on content
        # Check for object
        elif re.search(r'=\s*\{', line):
            return "Record<string, any>"  # Could be more specific based on content
        # Check for function
        elif re.search(r'=\s*\([^)]*\)\s*=>', line):
            return "Function"
        return None

    def _generate_type_guard(self, type_name: str) -> str:
        """Generate a type guard function for the given type"""
        return f"""
function is{type_name}(value: any): value is {type_name} {{
    // TODO: Implement type guard logic
    return true;  // Replace with actual type checking
}}
""" 

    def _generate_package_json(self) -> Dict:
        """Generate default package.json content"""
        return {
            "name": "typescript-project",
            "version": "0.1.0",
            "private": True,
            "scripts": self._generate_default_scripts(),
            "dependencies": {
                "react": "latest",
                "react-dom": "latest",
                "next": "latest"
            },
            "devDependencies": {
                "typescript": "latest",
                "@types/react": "latest",
                "@types/react-dom": "latest",
                "@types/node": "latest"
            }
        }

    def _generate_default_scripts(self) -> Dict:
        """Generate default npm scripts"""
        return {
            "dev": "next dev",
            "build": "next build",
            "start": "next start",
            "lint": "next lint",
            "type-check": "tsc --noEmit"
        }

    def _generate_tsconfig(self) -> Dict:
        """Generate default tsconfig.json content"""
        return {
            "compilerOptions": {
                "target": "es5",
                "lib": ["dom", "dom.iterable", "esnext"],
                "allowJs": True,
                "skipLibCheck": True,
                "strict": True,
                "forceConsistentCasingInFileNames": True,
                "noEmit": True,
                "esModuleInterop": True,
                "module": "esnext",
                "moduleResolution": "node",
                "resolveJsonModule": True,
                "isolatedModules": True,
                "jsx": "preserve",
                "incremental": True
            },
            "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx"],
            "exclude": ["node_modules"]
        } 

    def _generate_index_tsx(self) -> str:
        """Generate default index.tsx content"""
        return '''import React from 'react';
import type { NextPage } from 'next';

const Home: NextPage = () => {
  return (
    <div>
      <h1>Welcome to Next.js!</h1>
    </div>
  );
};

export default Home;
'''

    def _generate_gitignore(self) -> str:
        """Generate default .gitignore content"""
        return '''# dependencies
/node_modules
/.pnp
.pnp.js

# testing
/coverage

# next.js
/.next/
/out/

# production
/build

# misc
.DS_Store
*.pem

# debug
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# local env files
.env*.local

# typescript
*.tsbuildinfo
next-env.d.ts
'''

    def _generate_readme(self) -> str:
        """Generate default README.md content"""
        return '''# Next.js TypeScript Project

This is a [Next.js](https://nextjs.org/) project bootstrapped with TypeScript.

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.
'''

    def _generate_env(self) -> str:
        """Generate default .env content"""
        return '''# Environment variables
NEXT_PUBLIC_API_URL=http://localhost:3000/api
'''

    def _generate_next_config(self) -> str:
        """Generate default next.config.js content"""
        return '''/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
}

module.exports = nextConfig
''' 

    def _generate_layout_component(self) -> str:
        """Generate default Layout component"""
        return '''import React from 'react';
import styles from '../styles/Layout.module.css';  // Path will be adjusted based on structure

interface LayoutProps {
    children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
    return (
        <div className={styles.container}>
            <main>{children}</main>
        </div>
    );
};

export default Layout;
''' 

    def _get_dependencies(self, project_dir: Path) -> Set[str]:
        """Get project dependencies from package.json"""
        try:
            package_json = project_dir / "package.json"
            if package_json.exists():
                with open(package_json) as f:
                    pkg_data = json.load(f)
                    return set(pkg_data.get("dependencies", {}).keys())
            return set()
        except Exception as e:
            self.logger.error(f"Error getting dependencies: {e}")
            return set()

    def _get_dev_dependencies(self, project_dir: Path) -> Set[str]:
        """Get project dev dependencies from package.json"""
        try:
            package_json = project_dir / "package.json"
            if package_json.exists():
                with open(package_json) as f:
                    pkg_data = json.load(f)
                    return set(pkg_data.get("devDependencies", {}).keys())
            return set()
        except Exception as e:
            self.logger.error(f"Error getting dev dependencies: {e}")
            return set()

    def _generate_global_styles(self) -> str:
        """Generate default global styles"""
        return '''/* Global styles */
:root {
  --primary-color: #0070f3;
  --background-color: #ffffff;
  --text-color: #000000;
}

html,
body {
  padding: 0;
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Oxygen,
    Ubuntu, Cantarell, Fira Sans, Droid Sans, Helvetica Neue, sans-serif;
}

a {
  color: inherit;
  text-decoration: none;
}

* {
  box-sizing: border-box;
}
'''

    async def _get_llm_analysis(self, project_dir: Path) -> Dict:
        """Get LLM analysis for the entire project"""
        try:
            # Get all TypeScript files
            files = []
            for ext in self.get_file_extensions():
                files.extend(project_dir.glob(f"**/*{ext}"))

            # Combine content for analysis
            combined_content = ""
            for file in files:
                if "node_modules" not in str(file):
                    try:
                        with open(file) as f:
                            combined_content += f"\n\n// {file.name}\n{f.read()}"
                    except Exception as e:
                        self.logger.error(f"Error reading {file}: {e}")

            # Get LLM analysis
            if combined_content:
                return await self.primary_llm.analyze_code(
                    combined_content,
                    "typescript",
                    {"context": str(project_dir)}
                )
            return {}

        except Exception as e:
            self.logger.error(f"Error getting LLM analysis: {e}")
            return {} 