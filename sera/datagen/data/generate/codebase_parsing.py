import os
import glob
import logging
import json
import re
import shutil
import subprocess
import tempfile

from typing import List, Optional, Dict, Set, Tuple

##############################
# codebase folder wildcards

# Directories to always exclude when walking a TypeScript codebase
TS_EXCLUDE_DIRS: Set[str] = {
    "node_modules", "dist", "build", ".next", "out", "coverage",
    ".turbo", ".cache", ".git", "__pycache__", ".tox", ".mypy_cache",
}

# File patterns that indicate test files (should be filtered from call graph)
TS_TEST_PATTERNS: List[str] = [
    "__tests__", ".test.", ".spec.", "test/", "tests/", "e2e/",
    "__mocks__", "__fixtures__", "fixtures/", ".stories.",
]

# Supported TypeScript / JavaScript extensions
TS_EXTENSIONS: Tuple[str, ...] = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".mts")
# Primary TypeScript extensions (used as default filter)
TS_PRIMARY_EXTENSIONS: Tuple[str, ...] = (".ts", ".tsx")


def convert_path_to_wildcard(path: str):
    assert "/" in path
    path, filetype = os.path.splitext(path)
    split_path = path.split("/")
    for i, component in enumerate(split_path):
        if i > 0:
            split_path[i] = "*"
    return "/".join(split_path) + filetype


def _is_excluded_dir(dirpath: str) -> bool:
    """Check if any component of dirpath is in the exclude set."""
    parts = dirpath.replace("\\", "/").split("/")
    return any(part in TS_EXCLUDE_DIRS for part in parts)


def _is_test_file(filepath: str) -> bool:
    """Check if a filepath matches any TypeScript test file pattern."""
    normalized = filepath.replace("\\", "/")
    return any(pattern in normalized for pattern in TS_TEST_PATTERNS)


def get_folder_wildcards(path: str, filetype_filter=None, language: str = "typescript"):
    """
    Walk a directory tree and collect wildcard patterns for source files.

    Args:
        path: Root directory to walk.
        filetype_filter: Explicit extension filter. If None, defaults based on language.
        language: "python" or "typescript" -- determines default filter and exclusion rules.
    """
    # Determine effective filter
    if filetype_filter is not None:
        effective_filter = filetype_filter
    elif language == "python":
        effective_filter = ".py"
    else:
        effective_filter = TS_PRIMARY_EXTENSIONS  # tuple of extensions

    wildcard_set = set()
    for dirpath, dirnames, filenames in os.walk(path):
        # Skip excluded directories for TypeScript projects
        if language != "python" and _is_excluded_dir(dirpath):
            dirnames.clear()  # prevent os.walk from descending
            continue

        for filename in filenames:
            absolute_path = os.path.join(dirpath, filename)

            # Check extension
            if isinstance(effective_filter, tuple):
                if not absolute_path.endswith(effective_filter):
                    continue
            elif effective_filter and not absolute_path.endswith(effective_filter):
                continue

            # Skip test files for TypeScript projects
            if language != "python" and _is_test_file(absolute_path):
                continue

            wildcard_set.add(convert_path_to_wildcard(absolute_path))
    return list(wildcard_set)


def find_code_folders(repo_path: str, repo_last_name: str, base_commit: str,
                      top_level_folder: List[str], language: str = "typescript"):
    """
    Discover the top-level source code folder for a repository.

    Supports both Python and TypeScript project layouts.
    """
    cwd = os.getcwd()
    try:
        os.chdir(repo_path)
        code_path = None
        if top_level_folder:
            for tlf in top_level_folder:
                if os.path.exists(tlf):
                    code_path = tlf
        else:
            print("Trying to automatically find top level code folder...")
            if language != "python":
                # TypeScript-typical layouts
                if os.path.exists("src"):
                    code_path = "src"
                elif os.path.exists("lib"):
                    code_path = "lib"
                elif os.path.exists("packages"):
                    # Monorepo -- use packages as top-level
                    code_path = "packages"
                elif os.path.exists(repo_last_name):
                    code_path = repo_last_name
                elif os.path.exists(repo_last_name.lower()):
                    code_path = repo_last_name.lower()
                else:
                    print("Failed to find top level code folder, quitting now...")
            else:
                # Original Python heuristics
                if os.path.exists(repo_last_name):
                    code_path = repo_last_name
                elif os.path.exists(repo_last_name.lower()):
                    code_path = repo_last_name.lower()
                elif os.path.exists(os.path.join("src", repo_last_name)):
                    code_path = os.path.join("src", repo_last_name)
                elif os.path.exists("src"):
                    code_path = "src"
                else:
                    print("Failed to find top level code folder, quitting now...")

        if code_path:
            wildcards = get_folder_wildcards(code_path, language=language)
        else:
            wildcards = []
    except Exception as e:
        raise
    finally:
        os.chdir(cwd)
    return wildcards


##############################
# code2flow (Python -- legacy)

def split_function_path(func_path):
    """
    Expects in the form of lib/matplotlib/inset::InsetIndicator.set_alpha.
    Returns the file path and the function path. Here it would be lib/matplotlib/inset.py
    and InsetIndicator.set_alpha.
    """
    file_path, fn = func_path.split("::")
    module_name = os.path.split(file_path)[-1]
    if module_name.endswith(".py"):
        module_name = module_name[:-3]

    if not file_path.endswith(".py"):
        file_path += ".py"
    return file_path, fn

def get_full_path(folders, func, code_dict):
    file_name, func_name = split_function_path(func)
    found_file_path = ""
    found_times = 0
    # Search for the file containing the function
    func_components = func_name.split(".")
    for folder in folders:
        glob_path = os.path.join(folder, file_name)
        matches = [n for n in glob.glob(glob_path) if os.path.isfile(n)]
        for file_path in matches:
            if file_path in code_dict:
                file_text = code_dict[file_path]
            else:
                with open(file_path, "r") as f:
                    file_text = f.read()
                code_dict[file_path] = file_text
            if len(func_components) == 2 and func_components[0] in file_text and func_components[1] in file_text:
                found_file_path = file_path
                found_times += 1
            elif len(func_components) == 1 and func_components[0] in file_text:
                found_file_path = file_path
                found_times += 1

    # Only return successfully if we found a unique occurence of the function
    if found_times == 1:
        return found_file_path + "::" + func_name
    else:
        return None

def convert_to_file_path(call_graph, folders, node_id_to_name, nodes):
    func_to_path_map = {}
    line_mappings = {}
    code_dict = {}
    # Reverse the node_id_to_name dictionary to get the ID of any name
    node_name_to_id = {}
    for k, v in node_id_to_name.items():
        node_name_to_id[v] = k

    # Get line mappings of each function file path and create map of func name to func path
    remove_nodes = set()
    for i, func in enumerate(call_graph.keys()):
        file_path = get_full_path(folders, func, code_dict)
        if file_path is None:
            remove_nodes.add(func)
        else:
            func_to_path_map[func] = file_path
            line_mappings[file_path] = int(nodes[node_name_to_id[func]]["label"].split(":")[0])

    new_call_graph = {}
    for func, neighbors in call_graph.items():
        # Skip removed nodes
        if func in remove_nodes:
            continue

        # Filter neighbors and map to paths in one step
        valid_neighbors = [
            func_to_path_map[nbr]
            for nbr in neighbors
            if nbr not in remove_nodes
        ]

        new_call_graph[func_to_path_map[func]] = valid_neighbors
    return new_call_graph, line_mappings

def convert_code2flow_to_adj(loaded_json):
    new_call_graph = {}
    id_to_name = {}
    nodes = loaded_json["nodes"]
    edges = loaded_json["edges"]
    # Fill in adjacency list
    for uid in nodes:
        new_call_graph[nodes[uid]["name"]] = set()
        id_to_name[uid] = nodes[uid]["name"]
    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        new_call_graph[id_to_name[source]].add(id_to_name[target])
    # Convert sets to lists
    for node_name in new_call_graph:
        new_call_graph[node_name] = list(new_call_graph[node_name])
    return new_call_graph, id_to_name, nodes


##############################
# Jelly (TypeScript call graph via @cs-au-dk/jelly)

def _find_ts_entry_files(repo_path: str, relevant_folders: List[str]) -> List[str]:
    """
    Find TypeScript entry files for Jelly analysis.

    Jelly needs entry files to start its whole-program analysis. We look for
    common entry points: index.ts, main.ts, app.ts, or fall back to all .ts
    files matched by the relevant_folders wildcards.
    """
    entry_files = []

    # First, check for common entry point files at the repo root or src/
    common_entries = [
        "src/index.ts", "src/index.tsx", "src/main.ts", "src/main.tsx",
        "src/app.ts", "src/app.tsx", "index.ts", "index.tsx",
        "lib/index.ts", "lib/index.tsx",
    ]
    for entry in common_entries:
        full_path = os.path.join(repo_path, entry)
        if os.path.isfile(full_path):
            entry_files.append(entry)
            break  # Use the first match as primary entry

    # If no common entry found, collect all TS files from relevant folders
    if not entry_files:
        for folder_pattern in relevant_folders:
            matches = glob.glob(os.path.join(repo_path, folder_pattern))
            for match in matches:
                if os.path.isfile(match) and match.endswith(TS_PRIMARY_EXTENSIONS):
                    rel_path = os.path.relpath(match, repo_path)
                    entry_files.append(rel_path)

    return entry_files


def _check_jelly_installed() -> bool:
    """Check if Jelly (@cs-au-dk/jelly) is available on PATH."""
    try:
        result = subprocess.run(
            ["jelly", "--version"],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _run_jelly(repo_path: str, entry_files: List[str], output_path: str) -> bool:
    """
    Run Jelly to produce a call graph JSON file.

    Jelly CLI usage: jelly [options] [files...]
      -j <file>   Write call graph as JSON
      -m <file>   Write call graph as HTML
      --ignore-dependencies   Analyze only entry files and base directory files
      --timeout <ms>          Analysis timeout

    Returns True on success, False on failure.
    """
    if not entry_files:
        logging.warning("No TypeScript entry files found for Jelly analysis")
        return False

    cmd = [
        "jelly",
        "--ignore-dependencies",  # Skip node_modules analysis for speed
        "-j", output_path,        # JSON output
    ] + entry_files

    logging.info(f"Running Jelly: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        if result.returncode != 0:
            logging.error(f"Jelly failed (exit {result.returncode}): {result.stderr[:500]}")
            return False
        return os.path.isfile(output_path)
    except subprocess.TimeoutExpired:
        logging.error("Jelly analysis timed out after 5 minutes")
        return False
    except FileNotFoundError:
        logging.error(
            "Jelly not found. Install with: npm install -g @cs-au-dk/jelly"
        )
        return False


def _strip_ts_extension(file_path: str) -> str:
    """Strip TypeScript/JavaScript extension from a file path, if present."""
    for ext in TS_EXTENSIONS:
        if file_path.endswith(ext):
            return file_path[:-len(ext)]
    return file_path


def convert_jelly_to_adj(jelly_json: dict, repo_path: str) -> Dict[str, List[str]]:
    """
    Convert Jelly's JSON call graph output into the adjacency list format
    expected by the rest of the SERA pipeline.

    Expected pipeline format (same as code2flow output after processing):
        {
            "path/to/file.ts::functionName": ["path/to/other.ts::calledFn", ...],
            "path/to/file.ts::ClassName.methodName": [...],
        }

    Jelly JSON output format (based on @cs-au-dk/jelly documentation):
    The JSON contains two main sections:

    1. "entries" - list of entry point functions
    2. "files" - mapping of file IDs to file paths
    3. "functions" - mapping of function IDs to function metadata:
       {
         "<id>": {
           "name": "<function name>",
           "file": "<file path or file ID>",
           "loc": { "start": {"line": N, "column": N}, "end": {...} }
         }
       }
    4. "calls" or "fun2fun" - list of call edges:
       [{"source": "<caller_id>", "target": "<callee_id>"}]
       OR mapping from caller to list of callees

    NOTE: The exact Jelly JSON schema may vary between versions. This function
    handles multiple known formats with fallback heuristics. If the format
    changes, update the parsing logic below.

    TODO: Pin to a specific Jelly version and validate against its exact schema.
    """
    adj_list: Dict[str, Set[str]] = {}

    # --- Format A: Jelly produces {"functions": {...}, "calls": [...]} ---
    # This is the most common format from `jelly -j output.json`
    if "functions" in jelly_json and "calls" in jelly_json:
        return _parse_jelly_format_functions_calls(jelly_json, repo_path)

    # --- Format B: Jelly produces {"fun2fun": [[src, tgt], ...], "call2fun": [...]} ---
    # Some versions use this adjacency-pair format
    if "fun2fun" in jelly_json:
        return _parse_jelly_format_fun2fun(jelly_json, repo_path)

    # --- Format C: Jelly produces {"entries": [...], "cg": {...}} ---
    # Older versions may nest the call graph under "cg"
    if "cg" in jelly_json:
        return convert_jelly_to_adj(jelly_json["cg"], repo_path)

    # --- Format D: Flat node/edge format similar to code2flow ---
    # {"nodes": {...}, "edges": [...]}
    if "nodes" in jelly_json and "edges" in jelly_json:
        return _parse_jelly_format_nodes_edges(jelly_json, repo_path)

    logging.warning(
        f"Unrecognized Jelly JSON format. Top-level keys: {list(jelly_json.keys())}. "
        f"Returning empty call graph. Please check Jelly version and update parser."
    )
    return {}


def _make_function_key(file_path: str, func_name: str, repo_path: str = "") -> str:
    """
    Create a function key in the pipeline-expected format: "relative/path.ts::funcName"

    This mirrors the code2flow convention of "module/path::ClassName.method"
    """
    # Normalize to relative path
    if repo_path and os.path.isabs(file_path):
        file_path = os.path.relpath(file_path, repo_path)
    # Remove leading ./ if present
    if file_path.startswith("./"):
        file_path = file_path[2:]
    return f"{file_path}::{func_name}"


def _parse_jelly_format_functions_calls(
    jelly_json: dict, repo_path: str
) -> Dict[str, List[str]]:
    """
    Parse Jelly format with "functions" dict and "calls" list.

    Expected structure:
    {
      "functions": {
        "<id>": {
          "name": "functionName",
          "file": "path/to/file.ts",
          "loc": {"start": {"line": 10, "column": 0}, "end": {"line": 20, "column": 1}}
        }
      },
      "calls": [
        {"source": "<caller_id>", "target": "<callee_id>"},
        ...
      ]
    }
    """
    functions = jelly_json["functions"]
    calls = jelly_json["calls"]

    # Build function ID -> key mapping
    id_to_key: Dict[str, str] = {}
    adj_list: Dict[str, Set[str]] = {}

    for func_id, func_info in functions.items():
        func_name = func_info.get("name", f"anonymous_{func_id}")
        file_path = func_info.get("file", "unknown")

        # Skip functions from node_modules or excluded directories
        if _is_excluded_dir(file_path):
            continue
        # Skip test files
        if _is_test_file(file_path):
            continue
        # Skip anonymous/lambda if they have no meaningful name
        if func_name in ("<anonymous>", "(anonymous)", ""):
            func_name = f"anonymous_{func_id}"

        key = _make_function_key(file_path, func_name, repo_path)
        id_to_key[func_id] = key
        if key not in adj_list:
            adj_list[key] = set()

    # Process call edges
    for edge in calls:
        source_id = str(edge.get("source", ""))
        target_id = str(edge.get("target", ""))

        if source_id in id_to_key and target_id in id_to_key:
            adj_list[id_to_key[source_id]].add(id_to_key[target_id])

    # Convert sets to lists
    return {k: list(v) for k, v in adj_list.items()}


def _parse_jelly_format_fun2fun(
    jelly_json: dict, repo_path: str
) -> Dict[str, List[str]]:
    """
    Parse Jelly format with "fun2fun" adjacency pairs.

    Expected structure:
    {
      "fun2fun": [
        ["path/to/file.ts:funcA", "path/to/other.ts:funcB"],
        ...
      ],
      "files": {"<id>": "path/to/file.ts", ...}  // optional
    }

    Function references may be in format "file:line:col:name" or "file:name".
    """
    fun2fun = jelly_json["fun2fun"]
    adj_list: Dict[str, Set[str]] = {}

    for pair in fun2fun:
        if len(pair) < 2:
            continue
        source_ref = pair[0]
        target_ref = pair[1]

        source_key = _normalize_jelly_func_ref(source_ref, repo_path)
        target_key = _normalize_jelly_func_ref(target_ref, repo_path)

        if source_key is None or target_key is None:
            continue

        # Skip excluded/test files
        source_file = source_key.split("::")[0] if "::" in source_key else ""
        target_file = target_key.split("::")[0] if "::" in target_key else ""
        if _is_excluded_dir(source_file) or _is_excluded_dir(target_file):
            continue
        if _is_test_file(source_file) or _is_test_file(target_file):
            continue

        if source_key not in adj_list:
            adj_list[source_key] = set()
        if target_key not in adj_list:
            adj_list[target_key] = set()
        adj_list[source_key].add(target_key)

    return {k: list(v) for k, v in adj_list.items()}


def _normalize_jelly_func_ref(ref: str, repo_path: str) -> Optional[str]:
    """
    Normalize a Jelly function reference into the pipeline's "file::func" format.

    Jelly references can be in various forms:
      - "path/to/file.ts:line:col:funcName"
      - "path/to/file.ts:funcName"
      - "funcName@path/to/file.ts"
      - Just a function ID string
    """
    if not ref:
        return None

    # Format: "path/to/file.ts:line:col:funcName"
    # Split on colon, but be careful about Windows paths (C:\...)
    parts = ref.split(":")

    # Try to detect "file:line:col:name" format (4+ parts where parts[1] is numeric)
    if len(parts) >= 4 and parts[1].isdigit():
        file_path = parts[0]
        func_name = parts[-1] if parts[-1] else f"anonymous"
        return _make_function_key(file_path, func_name, repo_path)

    # Try "file:funcName" format (2 parts, first looks like a path)
    if len(parts) == 2 and ("/" in parts[0] or parts[0].endswith(TS_EXTENSIONS)):
        file_path = parts[0]
        func_name = parts[1]
        return _make_function_key(file_path, func_name, repo_path)

    # Try "funcName@file" format
    if "@" in ref:
        func_name, file_path = ref.split("@", 1)
        return _make_function_key(file_path, func_name, repo_path)

    # Fallback: treat entire ref as a key
    if "::" in ref:
        return ref  # Already in expected format

    return None


def _parse_jelly_format_nodes_edges(
    jelly_json: dict, repo_path: str
) -> Dict[str, List[str]]:
    """
    Parse a node/edge format (similar to code2flow's graph structure).

    Expected structure:
    {
      "nodes": {
        "<id>": {"name": "file.ts::func", "file": "path/to/file.ts", ...}
      },
      "edges": [
        {"source": "<id>", "target": "<id>"},
        ...
      ]
    }
    """
    nodes = jelly_json["nodes"]
    edges = jelly_json["edges"]

    id_to_key: Dict[str, str] = {}
    adj_list: Dict[str, Set[str]] = {}

    for node_id, node_info in nodes.items():
        if isinstance(node_info, dict):
            name = node_info.get("name", "")
            file_path = node_info.get("file", "")
        elif isinstance(node_info, str):
            # Simple string node
            name = node_info
            file_path = ""
        else:
            continue

        # If name already has :: separator, use it directly
        if "::" in name:
            key = name
        elif file_path:
            key = _make_function_key(file_path, name, repo_path)
        else:
            key = name

        # Skip excluded
        file_part = key.split("::")[0] if "::" in key else ""
        if _is_excluded_dir(file_part) or _is_test_file(file_part):
            continue

        id_to_key[node_id] = key
        if key not in adj_list:
            adj_list[key] = set()

    for edge in edges:
        source_id = str(edge.get("source", ""))
        target_id = str(edge.get("target", ""))
        if source_id in id_to_key and target_id in id_to_key:
            adj_list[id_to_key[source_id]].add(id_to_key[target_id])

    return {k: list(v) for k, v in adj_list.items()}


##############################
# Unified entry point

def get_adj_list(repo_path: str,
                 repo_last_name: str,
                 base_commit: str,
                 relevant_folders: List[str],
                 metadata_dir: str,
                 overwrite: bool = False,
                 language: str = "typescript") -> Optional[Dict[str, List[str]]]:
    """
    Generate a function-level call graph as an adjacency list.

    For Python repos, uses code2flow (legacy behavior).
    For TypeScript repos, uses Jelly (@cs-au-dk/jelly).

    Returns:
        Dictionary mapping "file_path::function_name" -> ["file_path::called_fn", ...]
        or None on failure.
    """
    if language == "python":
        return _get_adj_list_code2flow(
            repo_path=repo_path,
            repo_last_name=repo_last_name,
            base_commit=base_commit,
            relevant_folders=relevant_folders,
            metadata_dir=metadata_dir,
            overwrite=overwrite,
        )
    else:
        return _get_adj_list_jelly(
            repo_path=repo_path,
            repo_last_name=repo_last_name,
            base_commit=base_commit,
            relevant_folders=relevant_folders,
            metadata_dir=metadata_dir,
            overwrite=overwrite,
        )


def _get_adj_list_jelly(repo_path: str,
                        repo_last_name: str,
                        base_commit: str,
                        relevant_folders: List[str],
                        metadata_dir: str,
                        overwrite: bool = False) -> Optional[Dict[str, List[str]]]:
    """
    Generate call graph for TypeScript projects using Jelly (@cs-au-dk/jelly).

    Jelly is a whole-program static analyzer for JavaScript/TypeScript that produces
    function-level call graphs. It analyzes imports, re-exports, class hierarchies,
    and dynamic dispatch to build accurate cross-file call graphs.

    Install: npm install -g @cs-au-dk/jelly
    CLI:     jelly -j callgraph.json [--ignore-dependencies] <entry-files...>

    The output JSON is parsed and transformed into the same adjacency list format
    that the rest of the SERA pipeline expects (file_path::func_name keys).
    """
    if not _check_jelly_installed():
        logging.error(
            "Jelly is not installed or not on PATH. "
            "Install with: npm install -g @cs-au-dk/jelly\n"
            "See: https://github.com/cs-au-dk/jelly"
        )
        return None

    cwd = os.getcwd()
    try:
        os.chdir(repo_path)
        cg_save_path = os.path.join(metadata_dir, f"{repo_last_name}_{base_commit[:5]}_jelly.json")

        if not os.path.exists(cg_save_path) or overwrite:
            # Find TypeScript entry files
            entry_files = _find_ts_entry_files(repo_path, relevant_folders)
            if not entry_files:
                logging.error(f"No TypeScript entry files found in {repo_path}")
                return None

            # Run Jelly
            success = _run_jelly(repo_path, entry_files, cg_save_path)
            if not success:
                logging.error(f"Jelly analysis failed for {repo_last_name}")
                return None

        # Load and parse the Jelly JSON output
        with open(cg_save_path, "r") as f:
            jelly_output = json.load(f)

        # Convert Jelly's JSON format to our adjacency list format
        adj_list = convert_jelly_to_adj(jelly_output, repo_path)

        if not adj_list:
            logging.warning(f"Jelly produced an empty call graph for {repo_last_name}")
            return None

        logging.info(
            f"Jelly call graph for {repo_last_name}: "
            f"{len(adj_list)} functions, "
            f"{sum(len(v) for v in adj_list.values())} edges"
        )
        return adj_list

    except FileNotFoundError:
        logging.error(f"\t\tFailed: {repo_last_name}, repo_path: {repo_path}, cur_dir: {os.getcwd()}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse Jelly JSON output: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error in Jelly call graph generation: {e}")
        return None
    finally:
        os.chdir(cwd)


def _get_adj_list_code2flow(repo_path: str,
                            repo_last_name: str,
                            base_commit: str,
                            relevant_folders: List[str],
                            metadata_dir: str,
                            overwrite: bool = False) -> Optional[Dict[str, List[str]]]:
    """
    Generate call graph for Python projects using code2flow (legacy).

    This is the original implementation preserved for backward compatibility
    with Python repository analysis.
    """
    cwd = os.getcwd()
    try:
        os.chdir(repo_path)
        cg_save_path = os.path.join(metadata_dir, f"{repo_last_name}_{base_commit[:5]}.json")
        if not os.path.exists(cg_save_path) or overwrite:
            # Generate call graph
            file_input = " ".join(relevant_folders)
            cmd = ["code2flow"] + [file_input] + ["--o", cg_save_path] + ["--quiet"]
            os.system(" ".join(cmd))
        with open(cg_save_path, "r") as f:
            call_graph = json.load(f)
        # Get call graph in adjacency list format
        adj_list, node_id_to_name, nodes = convert_code2flow_to_adj(call_graph["graph"])
        # Convert call graph to use full file paths, by default code2flow uses file_name::func_name
        adj_list, _ = convert_to_file_path(adj_list, [os.path.split(p)[0] for p in relevant_folders], node_id_to_name, nodes)
    except FileNotFoundError:
        # Sometimes codeflow fails, and then there is no file to read.
        logging.error(f"\t\tFailed: {repo_last_name}, repo_path: {repo_path}, cur_dir: {os.getcwd()}")
        adj_list = None
    except Exception as e:
        print(e)
        adj_list = None
    finally:
        os.chdir(cwd)
    return adj_list
