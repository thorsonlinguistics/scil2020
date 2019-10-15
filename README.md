# Unsupervised Formal Grammar Induction with Confidence

The code in this repository was used in Collard (2019): _Unsupervised Formal
Grammar Induction with Confidence_. The code trains the Missing Link algorithm
against a subset of the Billion Words Benchmark corpus and evaluates it against
the Corpus of Linguistic Acceptability.

In order to run the code, you will need the following:

- ccglink 0.1.0 (the code in this repository assumes it is installed to your
  PATH)
- python 3.7.4
- the python packages defined in requirements.txt must be installed in your
  Python path.

Other versions of Python and Python packages may still work as expected, but I
trained and evaluated the system using the above configuration.

To collect the corpora, train, and run the full evaluation pipeline, simply run
`./run.sh` from the command line. This will ultimately output the validation
score (expressed as a Matthews Correlation Coefficient). 

# Hardware

Missing Link does not require specialized hardware such as GPUs. I ran the
evaluation on an AMD Athlon II X2 B22 CPU with 2 cores. 
