---
name: systematic-debugging
description: 4-phase root cause debugging process. Reproduce, Isolate, Identify root cause, Fix and verify. Use when encountering persistent errors or when asked to debug an issue.
user-invocable: true
allowed-tools: "Read, Write, Edit, Bash, Glob, Grep"
---

# Systematic Debugging

A structured approach to finding and fixing bugs.

## The 4 Phases

### Phase 1: Reproduce

Goal: Create a minimal, reliable reproduction of the bug.

Steps:
1. Write a test case that demonstrates the bug
2. Run it to confirm the bug exists
3. Document exact steps to reproduce
4. Save reproduction script for regression testing

### Phase 2: Isolate

Goal: Narrow down where the bug could be.

Techniques:
1. Binary search: Comment out code until bug disappears
2. Logging: Add detailed logs around suspect area
3. Mocking: Replace dependencies with mocks
4. Simplification: Remove non-essential code paths

### Phase 3: Identify Root Cause

Goal: Understand WHY the bug happens.

Questions to answer:
1. What is the expected behavior?
2. What is the actual behavior?
3. What assumptions does the code make?
4. Which assumption is wrong?

### Phase 4: Fix and Verify

Goal: Implement a targeted fix and prove it works.

Steps:
1. Write the fix that addresses root cause
2. Run reproduction test
3. Run full test suite
4. Verify fix doesn't break anything else

## Debugging Log Template

Create `debug_log.md` in your project:

```markdown
# Debug Log: [Issue Description]

## Bug Description
[Clear description of the bug]

## Reproduction Steps
1. [Step 1]
2. [Step 2]
3. [Expected]: [What should happen]
4. [Actual]: [what actually happens]

## Isolation Attempts
| Technique | Result | Notes |
|-----------|--------|-------|
| [technique] | [result] | [notes] |

## Root Cause
[Description of the root cause]

## Fix
[Description of the fix applied]

## Verification
- [ ] Reproduction test passes
- [ ] Full test suite passes
- [ ] No regressions introduced
```

## Error Patterns to Watch For

| Pattern | Likely Cause | Solution |
|---------|--------------|----------|
| Null reference | Missing initialization | Add null checks |
| Index out of bounds | Wrong array access | Validate indices |
| Race condition | Concurrent access | Add synchronization |
| Memory leak | Unclosed resources | Use try/finally |
| Infinite loop | Wrong termination condition | Fix loop logic |

## Anti-Patterns

| Don't | Do Instead |
|-------|------------|
| Guess at solution without understanding | Follow 4 phases |
| Skip isolation phase | Always isolate first |
| Fix symptoms instead of root cause | Find underlying issue |
| Add logging without purpose | Target logging to suspect area |
| Change multiple things at once | Change one thing at a time |

## Integration with planning-with-files

1. Create debug_log.md at starting debugging
2. Update progress.md after each phase
3. Log root cause in findings.md
4. Mark debugging task in task_plan.md
