# Birds of a Feather :bird: 

## *Bringing birders together*

>Birds of a Feather is a web app to recommend potential birding partners from a list of over 100.000 active eBird users in the US. It finds good partners by matching the users birding preferences with encodings for other eBird users processed by a siamese neural network trained to distinguish suitable matches from unsuitable ones. With only a few clicks, birders can be pointed to ideal partners which they might otherwise never meet.


### Run web app:
```
streamlit run birds_app.py
```

### Scripts include functions for:
1. Reading and processing raw eBird data
2. Creating geospatial databases for eBird users
3. Training siamese neural network on birding pairs

### Repo includes shapefiles with active US eBird users
