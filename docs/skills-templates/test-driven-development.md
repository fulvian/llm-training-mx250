---
name: test-driven-development
description: Enforces RED-GREEN-REFACTOR cycle. Write failing test first, watch it fail, write minimal code, watch it pass, then refactor. Never write production code before tests.
user-invocable: true
allowed-tools: "Read, Write, Edit, Bash, Glob, Grep"
hooks:
  PreToolUse:
    - matcher: "Write|Edit"
      hooks:
        - type: command
          command: "if [ -f task_plan.md ]; then echo '[TDD] Before writing code, verify tests exist or are being written first.'; fi"
  PostToolUse:
    - matcher: "Bash"
      hooks:
        - type: command
          command: "if echo '$KILOCODE_TOOL_INPUT' | grep -q 'test\\|spec\\|describe'; then echo '[TDD] Test detected. Ensure you are following RED-GREEN-REFACTOR.'; fi"
metadata:
  version: "1.0.0"
---

# Test-Driven Development

Strict adherence to the RED-GREEN-REFACTOR cycle for all production code.

## The Cycle

```
┌─────────────────────────────────────────────────────────┐
│  RED: Write a failing test                              │
│  ├── Define expected behavior                           │
│  ├── Run test, watch it FAIL                            │
│  └── Commit: "test: describe expected behavior"         │
├─────────────────────────────────────────────────────────┤
│  GREEN: Write minimal code to pass                      │
│  ├── Implement ONLY what test requires                  │
│  ├── Run test, watch it PASS                            │
│  └── Commit: "feat: implement X"                        │
├─────────────────────────────────────────────────────────┤
│  REFACTOR: Clean up the code                            │
│  ├── Improve structure/naming                           │
│  ├── Tests still pass                                   │
│  └── Commit: "refactor: improve X"                      │
└─────────────────────────────────────────────────────────┘
```

## Non-Negotiable Rules

1. **NO CODE WITHOUT TESTS** - Production code requires tests first
2. **MINIMAL GREEN** - Write only what's needed to pass
3. **COMMIT EACH PHASE** - Separate commits for RED, GREEN, REFACTOR
4. **DELETE UNTESTED CODE** - If code exists without tests, question its necessity

## Anti-Patterns to Avoid

| Don't | Do Instead |
|-------|------------|
| Write code then add tests | Write tests first |
| Skip running failing test | Always verify RED fails |
| Implement extra features | Only what test requires |
| Big refactors | Small, incremental changes |
| Skip REFACTOR phase | Always clean up after GREEN |

## Testing Checklist

Before marking code complete:
- [ ] Tests exist for new functionality
- [ ] Tests were written BEFORE implementation
- [ ] All tests pass
- [ ] Code coverage acceptable
- [ ] Edge cases covered

## When to Skip TDD

- Prototyping/exploration
- One-off scripts
- Configuration files
- Documentation

## Integration with planning-with-files

After completing each phase:
1. Update progress.md with test results
2. Mark task as complete in task_plan.md only after tests pass
3. Log any test failures in Errors Encountered section
