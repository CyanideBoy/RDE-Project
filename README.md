# RDE in NLP
Rate distortion approach used in explaining NLP task <br>
Important links for implementation of model - <br>
* Innvestigate [https://github.com/albermax/innvestigate]
* LRP information [http://www.heatmapping.org/]
* RAP Method [https://github.com/wjNam/Relative_Attributing_Propagation]
<br>
Knowledge distillation experiments not rigorously done yet <hr>

| Folder 	|  Description  	|
|:----------------:	|:--------------:	|
| AGNews 	|  RDE methods and experiments <br> on AG News dataset  	|
|    `tester.py`   	|   for testing  	|
|    `model.py`    	| contains model 	| 
|`CustomDataset.py` | contains datagenerator |
| `norm.py`         | script to get mean and std of embeddings |
|     `mean_std.npy`    |  contains mean and std of embeddings |