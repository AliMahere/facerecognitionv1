### to prepare your dataset first run 
if your dataset doesn't contains only face images and contains whole body you need to run this command first 

```python data_prepration.py  --dataset <path_to_your_dataset> create_faces --save_path <path_to_save_new_data_set> --padding_ratior 0.3 -aline ```<br>
and also if your dataset is impalnced you cand run this command 
``` python data_prepration.py  --dataset <path_to_your_dataset> balance --spamples <int_desired_number_of_samples_per_class>```
> type data_prepration.py --help for more information 
## second to encode or recognize in image/ video / set of (imgs, vedios ) rum 
to encode your dataset you con run this
```python test.py encode --dataset <path_to_your_dataset> --encodings <path_to_save_your_encodings_ex_enco.pickle>  ```
after encoding your dataset you can use these encodes to recognize faces using this command
```python test.py recognize --input <your_input_can_be_video_file_img_or_directory> --embeddings <path_to_your_saved_embdings> --output <path_to_your_outpu> -aline```
don't forget to add -aline argument this will incese you recogntion accurecy 
<br>
> type  ```python test.py --help```
### you will find output sample in  output/myoutput.mp4
># important nots 
> there is alot of feature work that can be done to enhance accurecy of face recognition 
1. systimaticly chose 
2. invest more work in preprocessing to get best represintaiton of face features and number of samples for eatch class 
3. try different classification algorithms like KNN 
