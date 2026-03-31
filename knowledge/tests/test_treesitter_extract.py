"""Tests for tree-sitter/regex AST pair extraction."""

import tempfile
from pathlib import Path

from larql_knowledge.ingest.treesitter_extract import (
    SUPPORTED_LANGUAGES,
    LANGUAGE_EXTENSIONS,
    detect_language,
    extract_pairs_from_source,
    extract_pairs_from_directory,
)


# ------------------------------------------------------------------
# Language metadata
# ------------------------------------------------------------------

def test_supported_languages():
    assert len(SUPPORTED_LANGUAGES) >= 15


def test_language_extensions():
    for lang in SUPPORTED_LANGUAGES:
        assert lang in LANGUAGE_EXTENSIONS, f"Missing extensions for {lang}"
        assert len(LANGUAGE_EXTENSIONS[lang]) > 0, f"Empty extensions for {lang}"


# ------------------------------------------------------------------
# detect_language
# ------------------------------------------------------------------

def test_detect_language_py():
    # .py is handled by ast_extract, not treesitter_extract
    assert detect_language(Path("foo.py")) is None


def test_detect_language_rs():
    assert detect_language(Path("foo.rs")) == "rust"


def test_detect_language_js():
    assert detect_language(Path("foo.js")) == "javascript"


def test_detect_language_ts():
    assert detect_language(Path("foo.ts")) == "typescript"


def test_detect_language_go():
    assert detect_language(Path("foo.go")) == "go"


def test_detect_language_java():
    assert detect_language(Path("foo.java")) == "java"


def test_detect_language_unknown():
    assert detect_language(Path("foo.xyz")) is None


# ------------------------------------------------------------------
# extract_pairs_from_source -- Rust
# ------------------------------------------------------------------

def test_extract_rust_function():
    pairs = extract_pairs_from_source("fn main() { }", "rust")
    assert "fn" in pairs
    assert ["fn", "main"] in pairs["fn"]


def test_extract_rust_struct():
    pairs = extract_pairs_from_source("struct Point { x: i32 }", "rust")
    assert "struct" in pairs
    assert ["struct", "Point"] in pairs["struct"]


def test_extract_rust_impl():
    pairs = extract_pairs_from_source("impl Foo { fn bar() {} }", "rust")
    assert "impl" in pairs
    assert ["impl", "Foo"] in pairs["impl"]


# ------------------------------------------------------------------
# extract_pairs_from_source -- JavaScript
# ------------------------------------------------------------------

def test_extract_js_function():
    pairs = extract_pairs_from_source("function hello() {}", "javascript")
    assert "function" in pairs
    assert ["function", "hello"] in pairs["function"]


def test_extract_js_const():
    pairs = extract_pairs_from_source("const x = 5;", "javascript")
    assert "const" in pairs
    assert any(p[1] == "x" for p in pairs["const"])


def test_extract_js_class():
    pairs = extract_pairs_from_source("class MyClass {}", "javascript")
    assert "class" in pairs
    assert ["class", "MyClass"] in pairs["class"]


# ------------------------------------------------------------------
# extract_pairs_from_source -- TypeScript
# ------------------------------------------------------------------

def test_extract_ts_interface():
    pairs = extract_pairs_from_source("interface Foo { bar: string }", "typescript")
    assert "interface" in pairs
    assert ["interface", "Foo"] in pairs["interface"]


# ------------------------------------------------------------------
# extract_pairs_from_source -- Go
# ------------------------------------------------------------------

def test_extract_go_func():
    pairs = extract_pairs_from_source("func main() {}", "go")
    assert "func" in pairs
    assert ["func", "main"] in pairs["func"]


# ------------------------------------------------------------------
# extract_pairs_from_source -- Java
# ------------------------------------------------------------------

def test_extract_java_class():
    pairs = extract_pairs_from_source("public class Main {}", "java")
    assert "class" in pairs
    assert ["class", "Main"] in pairs["class"]


# ------------------------------------------------------------------
# extract_pairs_from_source -- C
# ------------------------------------------------------------------

def test_extract_c_function():
    pairs = extract_pairs_from_source("int main() { return 0; }", "c")
    assert "int" in pairs
    assert ["int", "main"] in pairs["int"]


# ------------------------------------------------------------------
# extract_pairs_from_source -- Bash
# ------------------------------------------------------------------

def test_extract_bash_function():
    pairs = extract_pairs_from_source("function foo() { echo hi; }", "bash")
    assert "function" in pairs
    assert ["function", "foo"] in pairs["function"]


# ------------------------------------------------------------------
# extract_pairs_from_source -- SQL
# ------------------------------------------------------------------

def test_extract_sql_select():
    pairs = extract_pairs_from_source("SELECT name FROM users;", "sql")
    # SQL keywords are stored upper-case in _REGEX_PATTERNS
    assert "SELECT" in pairs
    assert any(p[1] == "name" for p in pairs["SELECT"])


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------

def test_extract_empty_source():
    pairs = extract_pairs_from_source("", "rust")
    assert pairs == {}


# ------------------------------------------------------------------
# extract_pairs_from_directory
# ------------------------------------------------------------------

def test_extract_from_directory_returns_expected_format():
    with tempfile.TemporaryDirectory() as td:
        rust_file = Path(td) / "sample.rs"
        rust_file.write_text("fn hello() {}\nstruct Foo {}\n")

        result = extract_pairs_from_directory(Path(td), "rust")
        assert "language" in result
        assert "source" in result
        assert "num_files" in result
        assert "relations" in result
        assert result["language"] == "rust"
        assert result["num_files"] >= 1
        assert isinstance(result["relations"], dict)
