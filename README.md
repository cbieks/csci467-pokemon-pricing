# Pricing Pokémon

CSCI 467 (USC, Fall 2026) course project by Cameron Bieker: predicting the TCGplayer market price of an English Pokémon TCG card from its printed attributes and artwork.

| Folder | Contents |
|---|---|
| [`code/`](code/) | Data pipeline, models, results, and the processed dataset. See [`code/README.md`](code/README.md) for the commands that reproduce every result. |
| [`report/`](report/) | LaTeX source of the reports (ICML template), synced with Overleaf. |

## Syncing the report with Overleaf

`report/` is a `git subtree` of the Overleaf project's Git repository, so the report has one working copy here and Overleaf compiles it.

```bash
git remote add overleaf https://git@git.overleaf.com/6ac45295680c200d763eea34   # once per clone

git subtree pull --prefix report overleaf main   # bring in edits made in Overleaf
git subtree push --prefix report overleaf main   # send local report edits to Overleaf
git push origin main                             # push code + report to GitHub
```

Overleaf authenticates with username `git` and an Overleaf Git token as the password.
