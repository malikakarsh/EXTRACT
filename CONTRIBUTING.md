# Contributing to this project:

Follow the guidelines to contribute to the project

## Bug Reports:

A bug is a demonstrable problem that is caused by the code in the repository. Good bug reports are extremely helpful - thank you!

Guidelines for bug reports:

**1) Use the GitHub issue search**  — check if the issue has already been reported.

**2) Check if the issue has been fixed** — try to reproduce it using the latest master or development branch in the repository.

**3) Isolate the problem** — create a reduced test case and a live example.

## Pull Requests:

Follow this process if you'd like your work considered for inclusion in the project:

1) Fork the project, clone your fork, and configure the remotes:
```
# Clone your fork of the repo into the current directory
git clone https://github.com/<your-username>/<repo-name>

# Navigate to the newly cloned directory
cd <repo-name>

# Assign the original repo to a remote called "upstream"
git remote add upstream https://github.com/<upstream-owner>/<repo-name>
```
2) If you cloned a while ago, get the latest changes from upstream:
```
git checkout <dev-branch>
git pull upstream <dev-branch>
```
3) Create a new topic branch (off the main project development branch) to contain your feature, change, or fix:
```
git checkout -b <topic-branch-name>
```
4) Commit your changes in logical chunks. Please adhere to these git commit message guidelines or your code is unlikely be merged into the main project. Use Git's interactive rebase feature to tidy up your commits before making them public.

5) Locally merge (or rebase) the upstream development branch into your topic branch:
```
git pull [--rebase] upstream <dev-branch>
```
6) Push your topic branch up to your fork:
```
git push origin <topic-branch-name>
```
7) **Open a Pull Request with a clear title and description.**
