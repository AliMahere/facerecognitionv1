# AnovatefaceReco
##### assesment project for hireing process in anovate.ai
#### to prepare your dataset first run 
if your dataset doesn't contains only face images and contains whole body you need to run this command first 

```python data_prepration.py  --dataset <path_to_your_dataset> create_faces --save_path <path_to_save_new_data_set> --padding_ratior 0.3 -aline ```
## second to encode or recognize in image/ video / set of (imgs, vedios ) rum 
### ```python test.py --help```
first the data_prepration model have tow main commnds 
1. balance 
2. create_faces <br>
you can ues the create_face if the faces insde dataset is not croped this will allow you to genrate new datast with only corped faces 
it also allow you to apply padding for better recognition for lare face <br>
also first command can allow you to bance dataset if it's imbalace by genratting new samples for mainor classes till there num of samples based on you decision 
