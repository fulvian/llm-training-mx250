---
name: code-review
description: Structured code review against plan and best practices. Reviews after implementation and before commits. Reports issues by severity with critical issues blocking progress.
user-invocable: true
allowed-tools: "Read, Glob, Grep"
---
# Code Review

Systematic code review against implementation plan and coding standards.

## Review Checklist

### Correctness
- [ ] Does code match the plan specification?
- [ ] Are all requirements from plan implemented?
- [ ] Are there any missing features?

### Code Quality
- [ ] Is code readable and well-organized?
- [ ] Are functions/methods appropriately sized?
- [ ] Is there code duplication?
- [ ] are names clear and meaningful?

### Testing
- [ ] are there unit tests?
- [ ] do tests cover the happy path?
- [ ] are edge cases tested?
- [ ] is there adequate test coverage?

### Security
- [ ] are there any security issues?
- [ ] is input validated?
- [ ] are there any hardcoded secrets?
- [ ] is error handling appropriate?

### Performance
- [ ] are there obvious performance issues?
- [ ] is algorithm complexity appropriate?
- [ ] are there any N+1 queries?

## Severity Levels

| Level | Description | Action |
|-------|-------------|--------|
| **CRITICAL** | Blocks progress, must fix immediately |
| **HIGH** | Should be fixed | Fix before merge |
| **medium** | Minor issues | Fix when convenient |
| **low** | Style suggestions | Consider for future |

## Review Template

After reviewing code, create `review_report.md`:

```markdown
# Code Review Report

## Summary
[Brief summary of changes]

## Files Reviewed
| File | Lines Changed | Type |
|------|---------------|------|
|      |               |      |

## Checklist Results

| Category | Status | Notes |
|----------|--------|-------|
| Correctness | ✅/❌ |       |
| Code Quality | ✅/❌ |       |
| Testing | ✅/❌ |       |
| Security | ✅/❌ |       |
| Performance | ✅/❌ |       |

## Issues Found

| Severity | File | Line | Issue | Suggestion |
|----------|------|------|-------|-------------|
|          |      |      |       |             |

## Recommendations
1. [Recommendation 1]
2. [Recommendation 2]
3. [Recommendation 3]

## Integration with planning-with-files

1. Run review after completing implementation tasks
2. Update task_plan.md with review status
3. Log issues in Errors Encountered section
4. Critical issues block progress until resolved
