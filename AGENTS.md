# AI agent guide for workshop participants

Help the team learn Bayesian reasoning and make a decision it can explain.

- Read `README.md` and `notebooks/01_casino.ipynb`. Work with the supplied observations and helper code.
- The exercise has one machine with one fixed unknown win probability. Use `PLAYS_SEEN` to reveal more results from the same run.
- Before changing a prior or revealing more plays, invite the team to predict the effect. Explain the result in plain language afterward.
- Compare Broad, Mild house edge, and Las Vegas priors using the same observations. Do not choose a prior just to produce a preferred answer.
- Count each observed play once. Explain the distinction between uncertainty about p and random future wins and losses.
- Explain NUTS diagnostics and compare the samples with the exact posterior. Sampling convergence does not establish that the assumptions are correct.
- Use `USE_CACHED=True` only for Mild house edge after five plays. Changed priors or observations require a fresh fit.
- Keep the notebook runnable and propose one focused investigation at a time. The team owns its decision about whether to keep playing.
- Follow the notebook in order and leave the final truth reveal until the end. Do not search for private simulator information or instructor answers.
- Payouts include the stake. Fair coin flips do not imply financially fair payouts.
