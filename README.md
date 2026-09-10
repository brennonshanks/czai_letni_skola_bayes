# Bayesian decisions: how much do you trust your prediction?

A model can give you a prediction without telling you how much evidence supports it. If you are going to act on that prediction, that distinction matters. Would you play a slot machine after watching it win three times?

In this workshop, you will explore those questions through a Python exercise. You will compare beliefs that describe a range of possible outcomes, see how their predictions depend on your assumptions, and use them to make decisions. Along the way, you will look for reasons to question your own results.

You do not need any previous experience with Bayesian methods. If you are comfortable with Python and have worked with AI or machine learning, you have what you need to get started. The notebook introduces the ideas as you use them and include working code for each step.

## Working with your team

Work in a team of two or three. One person can drive the notebook while the others interpret the results and question the choices you are making. Swap roles as you go. You can share a laptop, or run your own copies to compare different assumptions.

AI coding assistants are welcome. They can help explain unfamiliar code, make changes, and investigate unexpected results. Before asking an agent to change something, take a moment to predict what you think will happen. Afterward, check whether the result makes sense to your team. For example:

> “We think a stronger prior will have more influence on our estimate when we have fewer observations. Help us change the prior and compare the results.”

The goal is to leave with a decision you can explain. A useful question to keep asking each other is: **What would make us change our minds?**

## The Bayesian Casino

You are sitting in front of one slot machine with a fixed, unknown chance of winning. Choose a starting belief, reveal its results a few plays at a time, and see how your knowledge changes.

Each play costs one token. A win returns 2.4 tokens in total, so you gain 1.4 tokens on a winning play and lose one token on a losing play. These are fictional tokens.

Your starting belief is called a **prior**. You can choose a broad prior, a mild house-edge prior, or a stronger “Las Vegas” prior. All three use a Beta distribution to describe possible win probabilities.

Start with five plays, then reveal 20, 100, and 500 plays from the same run. Combine your prior with the observations to get your **posterior**, your updated belief about the machine's win probability. Compare the three priors and watch how much uncertainty remains as the evidence grows.

Use the posterior to predict the next 100 plays and decide whether you would keep playing. Even when you become confident about the machine's win probability, future wins and losses remain random.

An additional section introduces the no-U-turn-sampler (NUTS), a sampling method for posterior probability distributions based on Hamiltonian Monte Carlo. Compare its samples with the exact posterior you have already calculated, so you have a way to check the result.

Open [the casino notebook](notebooks/01_casino.ipynb) to begin.

## AI/ML extension: Bayesian optimization

After the casino exercise, try [the Gaussian-process Bayesian optimization
notebook](notebooks/02_gp_bayesian_optimization.ipynb). It asks which learning
rate to evaluate next when model-training runs are expensive. A Gaussian
process represents both predicted validation loss and uncertainty at untried
learning rates, and an acquisition rule turns that uncertainty into a decision.

The ML problem stays fixed while you design the optimizer itself. Change the
GP's prior level, plausible variation, smoothness, and observation noise, then
change how strongly the acquisition rule explores. You can compare how those
choices alter the next proposed experiment and the complete search path before
mapping the same sequential loop to your own project. A final optional section
then refits the GP hyperparameters before every decision by maximizing the
marginal likelihood, and contrasts that empirical-Bayes approach with keeping
your assumptions fixed.

The exercise uses a fixed table of simulated training results, so it runs
quickly and consistently without a GPU. Each cached run is a noisy draw around
a hidden smooth performance curve. That curve and the simulator's noise scale
are kept in a separate reveal table and loaded only after the search. The final
comparison scores GP-guided and random experiment selection by the true average
loss at each method's posterior-mean recommendation, under the same eight-run
budget, and contrasts the GP recommendation with the lowest observed run.

`scripts/generate_gp_tuning_results.py` regenerates both tables from the
documented curve, noise model, and random seed.

## Getting the notebook running

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

If you prefer VS Code, install the **Python** and **Jupyter** extensions, open the notebook there, and select the Python environment in `.venv` as the notebook kernel.

Try to install the packages and run the first setup cell before the session. Installation requires internet access; the datasets are already included in your download.

## If something takes a while or goes wrong

If Python says a package is missing, check that your notebook is using the same `.venv` environment where you installed the packages. If you have run cells out of order and the results seem inconsistent, restart the notebook kernel and run the cells again from the top.

For `ModuleNotFoundError: No module named 'nutpie'`, rerun `python -m pip install -r requirements.txt` in the activated environment. `nutpie` is required for the NUTS sampler and is included in that file. Alternatively, uncomment the `%pip install` line in the casino notebook's first setup cell and run it to install dependencies into the active notebook kernel. Restart the kernel afterward, then run from the top. Use `%pip`, rather than `!pip`, to target the notebook's Python environment.

The notebook has optional extensions. It is fine to finish the main exercise first and return to those if you have time. Aim to explain what you chose, what evidence supported it, and which assumption you would want to investigate next.
