# chpc_autoresearch

Welcome to my "chpc_autoresearch" repo where we are going to setup a repo such that coding agents on Github Copilot, can execute research on my behalf. Doing the research, implementing, running experiments and analysing resuls. All, completely automated using an agentic flow.

## Glossary

CHPC = Center For High Performanc Computing (in South Africa)
AUTORESEARCH = Agentic-driven research assistant, that continually does automatic research in this repo.

## Background

I am currently a PhD student in Computer Science, specialising in AI.
In the modern age of Agentic Coding AI Agents and LLMs, an interesting proposition is that of "recursive self-improvement".
This is now possible given the technology that we have.
I would like to setup this repo so that I can just as a remote coding agent on github what I want, and it will go and do the research for my, and help me improve it over time.
There is both a direction and automation component. The human researcher gives direction for experiments, or ideas, and the coding agent runs with it, doing the actual work.

Suppose the researcher has a simple goal like: "Let's research different NN architectures on image processing capabilities". That speaks to a typical ML research process where the researcher:

1. Does literature studies to get an idea of where the field sits today.
2. Makes a proposal for improvements or research contribution
3. Implements it
4. Does experimentation
5. Analyses results
6. Writes up findings
7. Repeats

One can them imagine a process where a coding agent starts with a simple FFNN and mnist, and gradually improves itself to the latest SOTA yolo/efficientnet models etc.

## What "defines" a good AI research assistant?

One that is an expert at following the scientific process.
That is effectively it.
That means that this researcher is good at surveying and gathering knowledge about the field/problem in question.
That means that this researcher is able to think outside the box, or to consider cross-domain techniques as a way of improving.
That means that this researcher collects and tracks literature, references (bibtex or similar), code, configurations, benchmarks, datasets. etc.
That means that this researcher is able to implement this things, or to build upon other's previous work.
That means that this researcher is able to run experiments, even if that requires CHPC-like resources.
That means that this researcher is able to analyse results, especially for statistical significance.
That means that this researcher can conclude findings from results.
That means that this researcher can propos future research directions.
That means that this researcher can continue this loop until it can no longer improve.

## Autoresearch loop

Since Coding agents lose context over time (context length limits), we need to define an "autoresearch loop" where a new fresh session of an AI agent can pick up where it left off so that it can continue on with its implementation. 
That means keeping track of what was done, what was implemented, what was collected, the outcomes, where everything is, how to use everything, what to do if you run into issues, a well-defined workflow.

I imagine something like this. There will be a main and develop branch like git flow. Then, a human researcher will start a new "project". That project is a high level topic, like "image processing with llms". Each project can have one or more "iterations"/sprints. Each sprint is like a "feature"/enhancement of the one before. So a first iteration could be, get a basic training loop going. The AI research agent must then be invoked. The AI agent makes a feature branch for that project. The agent keeps track of a new project. The agent then keeps track of iterations, and their status. Each iteration is a full research loop as discussed above. The outcome of a coding run would be the final results. That's it. 
If the researcher is happy, they will merge that to dev, and start the next iteration, by either saying you can continue improving or doing autoresearch on an existing project, or the human researcher will give specific direction.

## Some "cheat-codes"

To simplify some technical challenges:

1. We should provide at least some dependencies already installed and configured
2. An environment variables scheme so researchers can customise (CHPC config details, WANDB details etc.)
3. Access to and documentation of how to execute and run something on the CHPC.

## CHPC

The chpc is where we run our experiments.
In short this is what I always do:

1. ssh into "ssh xxxxxx@lengau.chpc.ac.za" -> where xxxxxx is my username. I already have an SSH key and configured, so I dont need to password authenticate. But if users do not have ssh keys configured, it would just be normal password authentication (which could be a secret in github repo secrets?)
2. This lands us at the login node.
3. Now, at this point if I have not cloned or configured my repo on the chpc, I do this now. To do this, we first need to ssh into an "internet" node. This is just a node that has keep-alive internet connection for long periods. Ideal for cloning repos and installing dependencies.
This is done by:

```
ssh chpclic1
```

once I have ssh'ed into the login node.
4. At this point, I navigate to "luster". Luster is a working storage volume where each user gets their own directory to work in. This is where our code, data, configs everything sits.

See below


```
ssh xxxxxx@lengau.chpc.ac.za                                                                                                                                                                                          12:14:48
xxxxxx@lengau.chpc.ac.za's password:

Welcome to LENGAU

################################################################################
#                                                                              #
# In order to receive notifications via email from the CHPC all users should   #
# be subscribed to the CHPC user distribution list. If you are not part of the #
# distribution list you can subscribe at the following link:                   #
# https://lists.chpc.ac.za/sympa/info/chpc-users                               #
#                                                                              #
################################################################################
[xxxxxx@login2 ~]$ cd lustre/
[xxxxxx@login2 lustre]$ ls -als
total 4372
   4 drwxr-x---    3 xxxxxx xxxxxx    4096 Feb 27 12:16 .
 392 drwxrwxr-x 7534 root       root        397312 Apr  9 15:54 ..
   4 drwxrwxr-x   14 xxxxxx xxxxxx    4096 Feb 27 12:16 LTHNS
3944 -rw-r--r--    1 root       root       4033728 Feb 16 18:50 lustre-clean-2026-02-16_15h55m02s.log
  28 -rw-r--r--    1 root       root         28424 Feb 27 12:16 lustre-clean-2026-02-27_09h41m13s.log
[xxxxxx@login2 lustre]$ pwd
/home/xxxxxx/lustre
[xxxxxx@login2 lustre]$ ssh chpclic1
\S
Kernel \r on an \m
Last login: Thu Mar  5 14:39:13 2026 from login2.cm.cluster
[xxxxxx@chpclic1 ~]$ ls -als
total 316
  4 drwxr-x---   10 xxxxxx xxxxxx   4096 Jan  8 22:32 .
240 drwxr-xr-x 6633 root       root       159744 Apr  9 15:53 ..
 24 -rw-------    1 xxxxxx xxxxxx  23177 Mar  5 15:19 .bash_history
  4 -rwx------    1 xxxxxx xxxxxx     18 Mar  3  2017 .bash_logout
  4 -rwx------    1 xxxxxx xxxxxx    193 Mar  3  2017 .bash_profile
  4 -rwx------    1 xxxxxx xxxxxx    248 Oct 18  2021 .bashrc
  0 drwxrwxr-x    7 xxxxxx xxxxxx     73 Dec 11 06:05 .cache
  0 drwxrwxr-x    5 xxxxxx xxxxxx     46 Dec 11 06:05 .config
  4 -rwx------    1 xxxxxx xxxxxx    334 Mar  3  2017 .emacs
  4 -rw-rw-r--    1 xxxxxx xxxxxx     65 Nov 14 20:34 .gitconfig
  0 drwxrwxr-x    2 xxxxxx xxxxxx     23 Oct 18  2021 .keras
  4 -rwx------    1 xxxxxx xxxxxx    172 Mar  3  2017 .kshrc
  0 drwxrwxr-x    5 xxxxxx xxxxxx     38 Nov 14 19:23 .local
  0 drwx------    4 xxxxxx xxxxxx     37 Mar  3  2017 .mozilla
  4 -rw-------    1 xxxxxx xxxxxx     86 Jan  8 23:17 .netrc
  0 drwx------    3 xxxxxx xxxxxx     25 Nov 14 19:51 .nv
  0 drwxrw----    3 xxxxxx xxxxxx     18 Oct 18  2021 .pki
  4 -rw-rw-r--    1 xxxxxx xxxxxx     15 Oct 18  2021 .profile
  4 -rw-------    1 xxxxxx xxxxxx    350 Nov 14 20:46 .python_history
  0 drwx------    2 xxxxxx xxxxxx     80 Jan  5 13:29 .ssh
 12 -rw-------    1 xxxxxx xxxxxx  10043 Jan  8 22:32 .viminfo
  0 lrwxrwxrwx    1 xxxxxx xxxxxx     29 Mar  3  2017 lustre -> /mnt/lustre/users/xxxxxx/
[xxxxxx@chpclic1 ~]$ cd lustre/
[xxxxxx@chpclic1 lustre]$ ls -als
total 4372
   4 drwxr-x---    3 xxxxxx xxxxxx    4096 Feb 27 12:16 .
 392 drwxrwxr-x 7534 root       root        397312 Apr  9 15:54 ..
   4 drwxrwxr-x   14 xxxxxx xxxxxx    4096 Feb 27 12:16 LTHNS -> this is a repo I have cloned into the chpc, and is also in the "example" folder.
3944 -rw-r--r--    1 root       root       4033728 Feb 16 18:50 lustre-clean-2026-02-16_15h55m02s.log
  28 -rw-r--r--    1 root       root         28424 Feb 27 12:16 lustre-clean-2026-02-27_09h41m13s.log
```
5. I then ensure that I have an experiment file (pbs script) that defines the job script for my experiment. This is a que-based processing system. So a file like this:

```
#!/bin/sh
#PBS -N mnist_ffnn_adam_imp
#PBS -q gpu_1
#PBS -P XXXXXXXX -> VERY IMPORTANT, THIS IS A UNIQUE PROJECT ID THAT A SUPERVISOR GIVES THEIR STUDENTS. IT TRACKS THAT SUPERVISOR's USAGE ACROSS THEIR RESEARCH PROJECTS.
#PBS -l select=1:ncpus=4:mem=64gb:ngpus=1
#PBS -l walltime=12:00:00
#PBS -o /mnt/lustre/users/xxxxxx/LTHNS/logs/mnist_ffnn_adam_imp.out
#PBS -e /mnt/lustre/users/xxxxxx/LTHNS/logs/mnist_ffnn_adam_imp.err
#PBS -m abe -M xxxxxx@icloud.com -> USER's EMAIL

set -euo pipefail

module purge
module load chpc/python/anaconda/3-2024.10.1

cd /mnt/lustre/users/xxxxxx/LTHNS
source .venv/bin/activate
# python -m lthns.imp --multirun experiment=mnist_ffnn_adam_imp
python -m lthns.imp --multirun experiment=mnist_ffnn_adam_imd
python -m lthns.imp --multirun experiment=mnist_ffnn_adam_imr
```

contains the chpc pbs config, dependencies to load, and then my execution script.
Side note: we will say more about he configs given above, below. But quickly, take note, the CHPC can send you an email once a job/experiment has finised.
6. I then submit this job using: `qsub experiments/mnist_ffnn_e2e_sequential.pbs`
7. I can check that a job is in the queue, or if it is running using: `qstat -u xxxxxx` or just qstat, but that includes all users' jobs.

### WIKI

Here: https://wiki.chpc.ac.za/
Start here: https://wiki.chpc.ac.za/survival_guide and here https://wiki.chpc.ac.za/quick:start
GPU Guide: https://wiki.chpc.ac.za/guide:gpu
CPU queues given here: https://wiki.chpc.ac.za/quick:start
Python guide: https://wiki.chpc.ac.za/guide:python

I provide this, because these options are required to define the selections and config given above.

HINT: IT WOULD BE GOOD TO PULL THESE ARTICLES, SAVE THEM AS MD FILES IN docs/ SO THAT WE CAN REFERENCE THEM LATER WITHOUT INTERNET. ALSO, MAKE A SKILL FOR THIS.


### AI AGENT

Naturally, this is a repo that should implement a bunch of agentic ai things.
This includes:

1. The documentation to understand what we are doing here
2. A state maintenance scheme where it keeps track of progress, and artifacts.
3. Coding capabilities
4. Skills
5. Prompts
6. Specialist agents

The idea is to be efficient. if we can do something once to make a run more efficient next time, we do it.

### Experimentation

Now, ofcourse, we can not cater for infinite possibilities for implementations. We suggested that we start with some framework. See the example/ folder I gave. it contains my ML research project for a "Lottery Ticket Hypernetwork". This repo already contains a really nice framework. it contains a config-driven mechanism to experimentation. This helps a lot so that I can maintain a LIBRARY of models, datasets, utilities, training loops, configs, etc. and just run experiments using configs. I can then also interpolate over parameters etc.

I effectively want this harness to be used for our experimentation. That means:

1. Config driven experimentation
2. Library of code, including models, datasets, utilities, training loops, configs, etc
3. A way to ssh into CHPC, insure the code is there, insure dependencies are there, and execute the experimentation.

Take note, we also use WANDB for experiment tracking. But this is more for the human. The AI would not to have data locally so it can read it and analyse it.

Use this example to scaffold out our experimentation hardness. And then start with some inital elements for a typical "image processing with NNs" experimentation seed. Like a simple FFNN, CNN, MNIST, and maybe one pretrained like a small resnet.

### Open Source

We will be open-sourcing this, so no sensitive data can be leaked.

### README

Please ensure you make a readme that should a summary of projects and their high level metrics/outcomes as a "leaderboard" page basically.
I want the readme to also reflect what we are doing here, how it works, how one can use the repo for your own research.

### SKILLS

Again, I can not emphasise this enough, split problems into smaller parts, define skills to tackle specific, specialist tasks.

### FINALLY

Ask if there is any uncertainty.
KEEP IT SIMPLE.