# Contributing Guidelines

To keep our development organized and the codebase clean, please follow these simple steps when working on the project.

## 1. Pick an Issue
* Go to the **Issues** tab.
* Assign yourself to the task you want to work on (**Assignees** section on the right).
* If the task doesn't exist yet, create a new Issue first.

## 2. Branching Strategy
**Never push directly to `main`.** Always create a new branch for your task from devel.
Use the following naming convention:
`type/issue-number-short-description`

**Examples:**
* `feature/` -  for new functionality.
* `fix/` - for bug fixes.
* `refactor/` - for code improvements.
* `prototype/` - for PoCs

## 3. Commits

Keep your commit messages clear and concise. Try to group your changes into logical steps.

## 4. Pull Requests (PR)

Once your code is ready and tested:

1. **Push** your branch
2. Open a **Pull Request** on Github
3. Link the **Issue**: In the PR description, include the keyword `Closes #X`(where X is the issue number). This will automatically close the issue when the PR is merged.
4. Request Review/Merge

