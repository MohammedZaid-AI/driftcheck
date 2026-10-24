---
name: commit-step
description: Use after finishing any phase or working checkpoint, before moving on.
---
1. Run pytest. If tests fail, fix them first, or start the commit message with wip.
2. Run git status and make sure no .env, keys, model weights or data images are staged.
3. Update PROGRESS.md with what is done, what is next, the commands to run, and anything waiting on the user.
4. Commit with the message format: phase N: short description
5. If a remote named origin exists, push.
6. At the end of a phase, tag nothing unless it is the final release v0.1.
