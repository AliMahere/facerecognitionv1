# AnovatefaceReco
###### assesment project for hireing process in anovate.ai
### to prepare your dataset first run 
if your dataset doesn't contains only face images and contains whole body you need to run this command first 

```python data_prepration.py  --dataset <path_to_your_dataset> create_faces --save_path <path_to_save_new_data_set> --padding_ratior 0.3 -aline ```<br>
and also if your dataset is impalnced you cand run this command 
``` python data_prepration.py  --dataset <path_to_your_dataset> balance --spamples <int_desired_number_of_samples_per_class>```
> type data_prepration.py --help for more information 
## second to encode or recognize in image/ video / set of (imgs, vedios ) rum 
### ```python test.py --help```
you can yes this model for encode your dataset or for recognize a prsoens in video/image or both of them 