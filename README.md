# Post-Quantum Anonymous Credentials

This implementation is based off the works of "Lattice-based Commit-Transferrable Signatures and Applications to Anonymous credentials" by Lai, Chen, Liu, Lysyanskaya, and Wang (2023). We have adapted the protocol to include the disclosure of properties of attributes, namely the NOT, the OR, and a non-arbitrary RANGE properties of attributes. We have also tested each of the protocol with correctness tests.

## Components used in setting up this project
```
Python 3.10.14
numpy 2.0.1
SageMath 10.3
Ubuntu 20.04.2 LTS
```

## Repository structure
```
root
|- tests
|  |- all_disclose_tests.py
|  |- all_size_disclose_tests.py
|  |- combo_test.py
|  |- extra_functions.py
|  |- issue_test.py
|  |- verify_test.py
|- ac.py
|- latticezk.py
|- LICENSE
|- README.md
```

## Commands
```
Run tests:
sage --python [test_name]
(Note: when running the issue or the verify test, uncomment the appropriate parameters at the top of latticezk.py)

```