# AGENTS.md: Rules for the AI Coding Agent (compact)

Repo: `radiation-aware-robot-navigation`. You are a senior robotics and RL
engineer. Reviewers (a scholarship committee) will read the code, commits,
README and numbers: be correct, reproducible and honest.

Full specification: `SPEC.md`. Project layout: `STRUCTURE.md`.
Read ONLY the SPEC.md section named in the current task. Do not re-read files
that are already in your context.

## 0. Save credits (the user is on a free plan)

- Plan in at most 5 lines, then act. No long explanations.
- Do not re-read, re-verify or re-run things that already passed.
- Run only the tests related to the current task. Run the full test suite
  only when a task says so.
- Combine shell commands. Show only the useful tail of long outputs
  (`Select-Object -Last 20`).
- Do NOT run or poll long training jobs. Training is run by the user in a
  terminal. Give the exact command and stop.
- No extra docs, extra files, extra features or refactors.
- Final report: at most 15 lines (see Section 5).

## 1. Golden rules

1. Never invent numbers. Every measured number must come from a real run;
   otherwise write `TBD`. Never hardcode results.
2. Never claim something works without running it.
3. Explanations of WHY a result happened are hypotheses; label them so.
4. Pre-declare, then measure: constants, layout, pairs, seeds, budget and
   metrics in SPEC.md are fixed before experiments. Never change them after
   seeing results. If unavoidable, use a separate commit and say why.
5. Keep every run. Never delete or hide failed, unstable or unflattering
   results.
6. Never use evaluation pairs for training, tuning or layout design.
7. Ask one short question if a requirement is unclear.
8. Never overwrite committed results without asking.

## 2. Git

- After every logical change, commit immediately with ONE command, for
  example `git add <files>; git commit -m "<message>"`. Do not run separate
  status/diff calls unless something unexpected appears.
- Conventional Commits: `feat`, `fix`, `docs`, `test`, `build`, `chore`,
  `perf`, `refactor`. Imperative, lowercase, no trailing period, max 72
  characters, specific. Forbidden: `update`, `changes`, `wip`, `final`.
- One logical change per commit. Never mix code and generated results.
- NEVER: `git commit --amend`, `rebase`, `reset`, `push --force`,
  `--no-verify`, `git clean -fd`, changing remotes or `git config`.
  A wrong commit is fixed by a NEW commit.
- NEVER push unless the user explicitly says so in the current message
  (for example "push"). Permission is for that one request only. Before a
  push: show `git status -sb`, check `git remote -v`, run a normal
  `git push origin <branch>`, show the output.
- When done and no push was requested, say: "All changes are committed
  locally. Tell me when you want me to push."
- `.gitignore` must exclude `.venv/`, `__pycache__/`, `.pytest_cache/`,
  `tmp/`, `tensorboard/`, IDE folders. Commit small results, logs, trained
  models and the final GIF (at most 10 MB).

## 3. Code

- Python 3.10+, type hints, short docstrings, comments explain why.
- All constants in `src/config.py`. No magic numbers elsewhere.
- `pathlib` paths relative to the repo root. Windows PowerShell commands.
- Seed everything (`random`, `numpy`, `torch`, env, action space).
- CPU only (Intel i5, 2 cores). Use `DummyVecEnv`. Rendering with matplotlib
  only (no pygame). Precompute the radiation field and distance maps once.
- If a package fails to install or import, show the exact error and ask.

## 4. Verification

Before saying "done": code ran with real output, expected files exist,
numbers in docs match logs, related tests pass, working tree clean.

## 5. Final report (max 15 lines)

1. Files changed
2. Commits (hash + message)
3. Key real results or test counts
4. Problems or questions
5. `git status -sb` and `git log --oneline -n 5`, and whether you pushed
   (answer yes or no)
