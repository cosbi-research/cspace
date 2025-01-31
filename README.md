# Quick start with CSpace

go to the `examples` folder, 
activate the python virtualenv and run

```
pip install -r requirements.txt
```

then run in the interactive prompt the script `example.py`

for a quick tour of CSpace capabilities.

# Run the performance tests

Go to the `tests` folder,
activate the python virtualenv and run

```
pip install -r requirements.txt
```

then run the `test.py` script in the command line
to get the correlation between CSpace and human judgement on MayoSRS and UMNSRS word similarity test sets.

```
python test.py cspace.kv.bin
```

run the `test_sentence.py` script in the command line
to get the correlation between CSpace and human judgement on BIOSSES sentence similarity test set.

```
python test_sentence.py cspace.kv.bin cspace.bigrams.pkl cspace.dict.pkl
```


# Training code and hyperparameters

The training code and hyperparameters can be found in the `training` folder.
As in the other folders, dependencies can be installed with 

```
pip install -r requirements.txt
```
