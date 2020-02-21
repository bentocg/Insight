# Birds of a Feather :bird: 

## *Bringing birders together*

>Birds of a Feather is a web app to recommend potential birding partners from a list of over 100.000 active eBird users in the US. It finds good partners by matching the users birding preferences with encodings for other eBird users processed by a siamese neural network trained to distinguish suitable matches from unsuitable ones. With only a few clicks, birders can be pointed to ideal partners which they might otherwise never meet.


### Start web app:
```
streamlit run birds_app.py
```

### Scripts include functions for:
1. Reading and processing raw eBird data
2. Creating a geospatial database for eBird users
3. Training siamese neural network on birding pairs to generate user encodings

### For a detailed overview of the data processing pipeline see *data_processing.ipynb* 

### To train encoding siamese network see:
```
python utils/train_encoder_nn.py -h
```

### Repo includes shapefiles with active US eBird users and saved model weights for best performing architectures.
