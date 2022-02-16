# AnovatefaceReco
## first install pkgs in requrement.txt
### to prepare your dataset first run 
```python data_prepration.py --help ```
## second to encode or recognize in image/ video / set of (imgs, vedios ) rum 
### ```python test.py --help```
first the data_prepration model have tow main commnds 
1. balance 
2. create_faces <br>
you can ues the create_face if the faces insde dataset is not croped this will allow you to genrate new datast with only corped faces 
it also allow you to apply padding for better recognition for lare face <br>
also first command can allow you to bance dataset if it's imbalace by genratting new samples for mainor classes till there num of samples based on you decision 
