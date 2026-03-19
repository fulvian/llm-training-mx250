---
name: brainstorming
description: Socratic design refinement. Explores alternatives through questions before writing code. Validates design in sections. Use when starting a new feature or when requirements are unclear.
user-invocable: true
allowed-tools: "Read, Write, Bash, Glob, Grep"
---
# Brainstorming

Socratic approach to design refinement. Never write code during this phase.

## The Process

### Step 1: Understand Context
Ask clarifying questions:
1. What problem are we trying to solve?
2. Who are the users?
3. What constraints exist?
4. What's the timeline?
5. What's already working?

6. What's the desired outcome?

### Step 2: Explore Alternatives
For each possible approach:
1. What are the trade-offs?
2. What are the risks?
3. What are the dependencies?
4. What's the learning curve?
5. What's the long-term maintainability?

6. How does this scale?

### Step 3: Present Design
Break design into sections:
1. **Overview** - High-level architecture
2. **Components** - Key modules and their responsibilities
3. **Data Flow** - How data moves through the system
4. **APIs** - External interfaces
5. **Security** - Authentication and authorization
6. **Error Handling** - How errors are managed

### Step 4: Validate
For each section:
1. Present to user for review
2. Address questions and concerns
3. Get explicit approval before proceeding
4. Document any decisions made

5. Note any open questions

## Design Document Template

After approval, create `design.md`:

```markdown
# Design: [Feature Name]

## Problem Statement
[What problem does this solve?]

## Proposed Solution
[High-level description of approach]

## Architecture Overview
[Diagram or description of main components]

## Components

### [Component 1]
- Responsibility: [what it does]
- Dependencies: [what it needs]
- API: [how it interacts]

- [Open questions: [what's undecided]

### [Component 2]
- Responsibility: [what it does]
- Dependencies: [what it needs]
- API: [how in interacts]
- [Open questions: [what's undecided]

## Data Flow
[How data moves through the system]

## Security Consider
- Authentication: [how users authenticate]
- Authorization: [what users can do]
- Data Protection: [how data is secured]

- [Open questions]: [what's undecided]

## Error Handling Strategy
[How errors are managed at different levels]

## Open Questions
| Question | Status | Notes |
|----------|--------|-------|
|          |        |       |

## Decisions Log
| Decision | Rationale | Alternatives Considered |
|----------|-----------|--------------------------|
|          |           |                          |

## Next Steps
1. [ ] Create implementation plan
2. [ ] Set up development environment
3. [ ] Start implementation

## Integration with planning-with-files

1. Save design.md after approval
2. Update task_plan.md with design phase complete
3. Log key decisions in findings.md
4. Proceed to writing-plans skill
