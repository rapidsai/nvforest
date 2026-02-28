# AI Code Review Guidelines for CodeRabbit - nvForest Python/Cython

**Role**: Act as a principal engineer with 10+ years experience in Python, Cython, and machine learning APIs. Focus ONLY on CRITICAL and HIGH issues.

**Target**: Sub-3% false positive rate. Be direct, concise, minimal.

**Context**: nvForest is a GPU-accelerated random forest inference library. This file covers Python and Cython code review guidelines for the nvforest Python package.

## IGNORE These Issues

- Style/formatting (linters handle this)
- Minor naming preferences (unless truly misleading)
- Personal taste on implementation (unless impacts maintainability)
- Nits that don't affect functionality
- Already-covered issues (one comment per root cause)

## CRITICAL Issues (Always Comment)

### Algorithm Correctness
- Logic errors that produce incorrect predictions from the Python API
- Incorrect data type conversions between Python and C++ (numpy arrays, GPU buffers)
- Breaking changes to inference behavior without versioning
- **Data layout errors** (passing wrong array order to C++ layer, C vs F order confusion)
- **Memory ownership bugs** (Python object freed while C++ still holds reference)

### Memory Management (Cython)
- Memory leaks in Cython code (malloc without free, missing destructor calls)
- Incorrect use of typed memoryviews (wrong dtype, incorrect strides)
- GIL handling errors (releasing GIL while accessing Python objects)
- Buffer protocol violations (returning invalid buffer info)
- Reference counting errors (missing Py_INCREF/DECREF in manual memory management)

### API Breaking Changes
- Python API changes breaking backward compatibility
- Cython API changes affecting downstream packages
- Changes to function signatures without deprecation warnings
- Removing or renaming public functions/classes without migration path

### Security
- Unsafe deserialization (pickle.load on untrusted data)
- Path traversal vulnerabilities in model loading
- Lack of input validation allowing resource exhaustion

## HIGH Issues (Comment if Substantial)

### Type Safety
- Missing or incorrect type hints on public APIs
- Type coercion that silently loses precision (float64 → float32)
- Incorrect numpy dtype handling
- Missing validation of array shapes and dtypes before passing to C++

### Error Handling
- Bare except clauses hiding real errors
- Missing error messages or unhelpful exception text
- Exceptions that don't propagate correctly from C++ through Cython
- Silent failures that should raise exceptions

### Cython-Specific Issues
- Inefficient Cython code in hot paths (Python object creation in loops)
- Missing `nogil` blocks where safe and beneficial
- Incorrect `cdef`/`cpdef`/`def` usage affecting performance or API exposure
- Memory views with incorrect mode (read-only vs writable)
- Missing `const` qualifiers on read-only data

### RAPIDS Integration
- Incorrect use of RAFT handles or CUDA streams
- Improper RMM memory resource usage (not respecting user-configured memory pools)
- Device memory handling that doesn't follow RAPIDS patterns (e.g., not using rmm::device_buffer)
- Missing support for standard GPU array inputs (cupy arrays, __cuda_array_interface__)

### Dependencies
- **Adding new dependencies without strong justification** (nvForest must remain lightweight)
- **Heavy dependencies that increase install size or add transitive dependencies**
- Adding dependencies that duplicate functionality already in numpy, treelite, or RAPIDS core libs

### Test Quality
- Missing tests for Python API edge cases
- Tests that don't verify prediction correctness (only "runs without error")
- Missing tests for error conditions and exception handling
- Inadequate coverage of different input types (numpy, cupy)

## MEDIUM Issues (Comment Selectively)

- Missing docstrings on public functions (NumPy style preferred)
- Incomplete type hints (partial annotations)
- Minor inefficiencies in non-critical Python code
- Missing input validation for edge cases
- Deprecated API usage without urgent security/correctness concern
- Code duplication that could be refactored

## Review Protocol

1. **Understand intent**: Read PR description, check if this affects Python API or behavior
2. **API correctness**: Do the Python APIs correctly wrap the C++ functionality?
3. **Type safety**: Are types properly validated and converted?
4. **Memory safety**: Is Cython memory management correct? GIL handled properly?
5. **Error handling**: Do errors propagate correctly with helpful messages?
6. **RAPIDS compatibility**: Does this integrate correctly with the RAPIDS ecosystem?
7. **Security**: Any unsafe deserialization or input handling?
8. **Ask, don't tell**: "Have you considered X?" not "You should do X"

## Quality Threshold

Before commenting, ask:
1. Is this actually wrong/risky, or just different?
2. Would this cause a real problem for users?
3. Does this comment add unique value?

**If no to any: Skip the comment.**

## Output Format

- Use severity labels: CRITICAL, HIGH, MEDIUM
- Be concise: One-line issue summary + one-line impact
- Provide code suggestions when you have concrete fixes
- Omit generic explanations and boilerplate
- No preamble or sign-off

## Token Optimization

- Omit explanations for obvious issues
- Omit descriptions of code or design not critical to understanding the changes or issues raised
- Omit listing benefits of standard good practices and other generic information apparent to an experienced developer
- No preamble or sign-off

## Context Awareness

**Skip if**:
- Already handled by CI/linters (ruff, mypy, etc.)
- Same issue exists in codebase (note once if systemic)
- Experimental/prototype code (check PR labels)
- Explicitly marked as technical debt

**Escalate if**:
- Breaking change without discussion
- Security vulnerability
- Memory safety issue in Cython code

## Examples to Follow

**CRITICAL** (memory ownership bug):
```
CRITICAL: Memory ownership error in Cython wrapper

Issue: Python array may be garbage collected while C++ still holds pointer
Why: Causes use-after-free, segfaults, or data corruption

Suggested fix:
# Keep reference alive for duration of C++ operation
cdef np.ndarray arr_ref = np.ascontiguousarray(arr)
self._cpp_obj.set_data(<float*>arr_ref.data)
self._arr_ref = arr_ref  # prevent GC
```

**CRITICAL** (GIL handling error):
```
CRITICAL: Accessing Python object without GIL

Issue: Python dict accessed inside `with nogil` block
Why: Can cause crashes or data corruption

Suggested fix:
# Either hold GIL or extract data before releasing
cdef int value
with gil:
    value = self.config['batch_size']
with nogil:
    self._process(value)
```

**HIGH** (type safety):
```
HIGH: Missing dtype validation before C++ call

Issue: Array dtype not validated before passing to C++ layer
Why: Wrong dtype causes silent incorrect results or crashes

Suggested fix:
if arr.dtype != np.float32:
    raise TypeError(f"Expected float32 array, got {arr.dtype}")
```

**HIGH** (error handling):
```
HIGH: Bare except clause hiding errors

Issue: `except:` catches and ignores all exceptions including KeyboardInterrupt
Why: Makes debugging difficult, can hide serious errors

Suggested fix:
except Exception as e:
    logger.error(f"Inference failed: {e}")
    raise
```

**CRITICAL** (unsafe deserialization):
```
CRITICAL: Unsafe pickle.load on user-provided file

Issue: pickle.load() on untrusted model file allows arbitrary code execution
Why: Security vulnerability - attacker can execute code via malicious model file

Suggested fix:
# Use safe model loading through Treelite instead
model = treelite.Model.load(filepath, format='xgboost_json')
```

**Good, concise summary**:
- Add type hints to all public API functions
- Validate array dtypes before passing to C++ layer
- Add deprecation warnings for renamed parameters

## Examples to Avoid

**Boilerplate and generic descriptions** (avoid):
- "Type hints improve code readability and enable better IDE support."
- "Docstrings help users understand how to use the API."
- "Input validation prevents errors downstream."

**Subjective style preferences** (ignore):
- "Consider using f-strings instead of .format()"
- "This could be a list comprehension"
- "Consider adding more comments"

---

## nvForest Python Specific Considerations

**Python API**:
- All public functions should have type hints
- All public functions should have NumPy-style docstrings
- API changes require deprecation warnings (at least one release cycle)
- Error messages should be helpful and include expected vs actual values
- Support standard input types: numpy arrays, cupy arrays (where applicable)

**Cython Code**:
- Use typed memoryviews for array access (not raw pointers when possible)
- Release GIL (`with nogil:`) for long-running C++ operations
- Ensure Python object references are held while C++ uses the data
- Use `cdef` for internal functions, `cpdef` for functions callable from Python and Cython
- Proper error handling that converts C++ exceptions to Python exceptions

**RAPIDS Integration**:
- Use RAFT handles consistently for GPU resource management and CUDA stream access
- Use RMM for device memory allocation (respect user-configured memory pools)
- Support device memory inputs via __cuda_array_interface__ (cupy arrays, rmm DeviceBuffer)
- Pass CUDA streams through the API to enable asynchronous operation

**Lightweight Design Philosophy**:
nvForest must remain a lean, focused inference library. When reviewing changes that add dependencies:
- **Question every new dependency**: Is it absolutely necessary? Can we achieve the same with existing deps?
- **Allowed dependencies**: numpy, treelite, and RAPIDS core libs (rmm, pylibraft)
- **Avoid**: Large ML frameworks, libraries with heavy transitive dependencies, optional "nice-to-have" deps
- **Install size matters**: New deps should not significantly increase wheel/conda package size
- **Runtime dependencies are costly**: Each new import adds startup time and potential version conflicts
- If functionality exists in numpy or an allowed dependency, use it rather than adding a new one

**Testing**:
- Test with various input types (numpy, cupy if supported)
- Test edge cases (empty arrays, single-sample batches, large batches)
- Verify prediction correctness against known-good values
- Test error conditions raise appropriate exceptions with helpful messages

**Documentation**:
When reviewing code changes that affect public APIs:
- Check if docstrings need updating
- Verify examples in docstrings still work
- Flag missing parameter descriptions
- Suggest documenting any performance characteristics or GPU requirements

Example documentation suggestion:
```
HIGH: Missing docstring for new public function

Issue: New `optimize_layout()` function has no docstring
Why: Users won't understand how to use it

Suggest: Add NumPy-style docstring with:
  - One-line summary
  - Parameters section with types and descriptions
  - Returns section
  - Example usage
```

---

## Common Bug Patterns in nvForest Python (Watch For These)

### 1. Array Layout Mismatch
**Pattern**: Passing arrays to C++ with wrong memory layout

**Red flags**:
- No explicit `np.ascontiguousarray()` or `np.asfortranarray()` call
- Assuming input arrays are already contiguous
- Not checking array flags before passing to C++

**Example bug**: Passing F-order array to C++ expecting C-order

### 2. Memory Lifetime Issues
**Pattern**: Python object garbage collected while C++ holds reference

**Red flags**:
- Passing `.data` pointer to C++ without keeping Python object alive
- Temporary array expressions passed directly to Cython functions
- Missing instance variable to hold array reference

**Example bug**: `self._cpp.set_input(np.ascontiguousarray(x).data)` - temporary is freed immediately

### 3. GIL Handling Errors
**Pattern**: Accessing Python objects without GIL, or holding GIL unnecessarily

**Red flags**:
- Python method calls inside `with nogil` blocks
- Dictionary/list access inside `with nogil` blocks
- Long-running C++ operations without releasing GIL

**Example bug**: Accessing `self.config` dict inside nogil block

### 4. Type Coercion Bugs
**Pattern**: Silent type conversion losing precision or causing incorrect results

**Red flags**:
- No explicit dtype check before array operations
- Automatic casting without warning (float64 → float32)
- Integer overflow when converting sizes/indices

**Example bug**: Model trained with float64 silently converted to float32 on inference

### 5. Exception Handling Gaps
**Pattern**: C++ exceptions not properly translated to Python exceptions

**Red flags**:
- C++ calls without try/except in Cython
- Generic exception types that lose error information
- Missing error messages or context

**Example bug**: C++ `std::runtime_error` becomes generic Python `RuntimeError` with no message

---

## Code Review Checklists by Change Type

### When Reviewing Python API Changes
- [ ] Are type hints present and accurate?
- [ ] Is there a NumPy-style docstring with parameters, returns, and example?
- [ ] Are breaking changes accompanied by deprecation warnings?
- [ ] Is input validation present for dtypes, shapes, and value ranges?
- [ ] Do error messages include expected vs actual values?

### When Reviewing Cython Code
- [ ] Are typed memoryviews used correctly (dtype, mode, strides)?
- [ ] Is GIL handled correctly (released for long ops, held for Python objects)?
- [ ] Are Python object references kept alive while C++ uses the data?
- [ ] Do C++ exceptions translate to appropriate Python exceptions?
- [ ] Is memory properly freed in all code paths (including exceptions)?

### When Reviewing Model Loading/Inference
- [ ] Are input array dtypes and layouts validated?
- [ ] Is the model file safely loaded (no pickle on untrusted files)?
- [ ] Are inference results the correct dtype and shape?
- [ ] Are edge cases handled (empty inputs, mismatched shapes)?

### When Reviewing Tests
- [ ] Do tests verify prediction correctness, not just "runs without error"?
- [ ] Are edge cases covered (empty, single-sample, large batches)?
- [ ] Are error conditions tested (wrong dtype, invalid shape, missing features)?
- [ ] Is test data deterministic for reproducibility?

---

**Remember**: Focus on objective correctness, not subjective preference. Catch real bugs and API issues, ignore style preferences. For nvForest Python: correct predictions, type safety, and memory safety are the priorities.
