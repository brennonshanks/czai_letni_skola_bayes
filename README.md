# Bayesian decisions: how much do you trust your prediction?

A model can give you a prediction without telling you how much evidence supports it. If you are going to act on that prediction, that distinction matters. Would you play a slot machine after watching it win three times? Would you invest your money because a model predicts that a stock price will rise?

In this workshop, you will explore those questions through two Python exercises. You will build models that describe a range of possible outcomes, see how their predictions depend on your assumptions, and use them to make decisions. Along the way, you will look for reasons to question your own results.

You do not need any previous experience with Bayesian methods. If you are comfortable with Python and have worked with AI or machine learning, you have what you need to get started. The notebooks introduce the ideas as you use them and include working code for each step.

## Working with your team

Work in a team of two or three. One person can drive the notebook while the others interpret the results and question the choices you are making. Swap roles as you go. You can share a laptop, or run your own copies to compare different assumptions.

AI coding assistants are welcome. They can help explain unfamiliar code, make changes, and investigate unexpected results. Before asking an agent to change something, take a moment to predict what you think will happen. Afterward, check whether the result makes sense to your team. For example:

> “We think a stronger prior will have more influence on machines with fewer observations. Help us change the prior and compare the results.”

The goal is to leave with a decision you can explain. A useful question to keep asking each other is: **What would make us change our minds?**

## First exercise: one slot machine

You are sitting in front of a slot machine whose chance of winning you do not know. You will reveal its results a few plays at a time and see how your beliefs change as you gather more evidence. The machine has one fixed win probability throughout the exercise; it is your knowledge of that probability that changes.

Each play costs one token. A win returns 2.4 tokens in total, so you gain 1.4 tokens on a winning play and lose one token on a losing play. These are fictional tokens throughout the workshop.

Before seeing any results, choose a starting belief, called a **prior**. You can be very unsure, lean gently toward a win probability around 40%, or choose a stronger “Las Vegas” prior that expresses a belief that the house probably has an edge. These are three choices of a Beta distribution, which describes possible probabilities between zero and one.

Start with five plays. Combine your prior with those results to obtain your **posterior**: your updated belief about the machine's win probability. Then reveal 20, 100, and 500 plays from the same run. You will see how the posterior moves and how much uncertainty remains, and compare what would have happened if you had chosen another prior.

Finally, use your posterior to predict the next 100 plays and decide whether you would keep playing. Even if you become confident about the machine's win probability, future wins and losses are still random.

An additional section introduces the no-U-turn-sampler (NUTS), a sampling method based on Hamiltonian Monte Carlo. You can compare its samples with the exact posterior you have already calculated, so you have a way to check the result.

Open [the casino notebook](notebooks/01_casino.ipynb) to begin.

## Second exercise: build a stock portfolio

Your team has 10,000 tokens and price histories for three fictional stocks: A, B, and C. You need to decide how much to put into each stock and how much to keep as cash.

You will use a **Gaussian process**, or GP, to describe possible future price curves. A GP lets you express assumptions about how a curve behaves—for example, how smoothly it changes over time. These assumptions are encoded partly through a function called a **kernel**. You will compare kernels and see how changing them affects both the forecast and its uncertainty.

Before making your final allocation, you will test forecasts against later observations within the historical dataset. You will also compare your model with a simple forecast based on the latest observed price. This gives you evidence to discuss before committing your tokens.

Your portfolio follows a few simple rules:

- Divide your tokens among A, B, C, and cash. The four percentages must total 100%.
- You can put everything into one choice, including cash, or spread your tokens across several choices.
- You hold the portfolio for 20 trading days without changing it.
- You cannot borrow tokens or bet on a stock falling. Cash earns no interest, and there are no trading fees.
- You can buy fractions of shares, so you do not need to round your allocation to whole shares.

When your team is ready, use the QR code or link shown during the session. One teammate submits your allocation, your estimated probability of losing money, and a short explanation. Keep the same team name and browser if you want to revise your submission before it closes.

We will then reveal what happened and watch the portfolio leaderboard. Making money is part of the game, but we will also discuss how much of the outcome came from a well-supported decision and how much came from luck.

Open [the stock portfolio notebook](notebooks/02_stock_portfolio.ipynb) for this exercise.

## Getting the notebooks running

You will need Python **3.11 or 3.12**. If you downloaded a ZIP, extract it first. Open a terminal in the extracted folder—the one containing this README and `requirements.txt`.

On macOS or Linux, run:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyterlab
```

If you use Python 3.12, replace `python3.11` in the first command with `python3.12`.

On Windows, using PowerShell, run:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m jupyterlab
```

For Python 3.12, use `py -3.12` in the first command. If PowerShell blocks activation, you can run the environment's Python directly:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m jupyterlab
```

These commands create a separate Python environment, install the packages, and open JupyterLab in your browser. In JupyterLab, open the `notebooks` folder and start with `01_casino.ipynb`. You can run each cell with **Shift+Enter**. Work from top to bottom, since later cells use results from earlier ones.

If you prefer VS Code, you can open the notebooks there instead. Select the Python environment in `.venv` as the notebook kernel.

Try to install the packages and run the first setup cell before the session. Installation requires internet access; the datasets are already included in your download.

## If something takes a while or goes wrong

The prior and posterior plots in the main casino exercise run immediately. The additional NUTS section can take a few minutes on its first run because PyMC needs to prepare the calculations before sampling. Later fits may be faster.

If you need to keep moving, find `USE_CACHED = False` in the casino notebook and change it to `True`. This loads samples for the Mild house edge prior after the first five plays. If you choose a different prior or reveal more plays, leave `USE_CACHED = False` so the sampler fits your current observations. The saved samples cannot reflect those changes.

If Python says a package is missing, check that your notebook is using the same `.venv` environment where you installed the packages. If you have run cells out of order and the results seem inconsistent, restart the notebook kernel and run the cells again from the top.

Each notebook has optional extensions. It is fine to finish the main exercise first and return to those if you have time. For both exercises, aim to explain what you chose, what evidence supported it, and which assumption you would want to investigate next.
