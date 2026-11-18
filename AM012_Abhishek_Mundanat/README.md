# Exploratory data analysis and Folium maps: Abhishek Mundanat (PES1UG24AM012)

| File | Purpose |
|---|---|
| `eda.py` | dataset inventory, sampling / gap / speed statistics, missing and invalid values, road-network size and connectivity, data-quality issue list, figures |
| `folium_maps.py` | interactive maps: GPS fixes, nearby road network, reference route, predicted route, layer control and legend |

Run (from the repository root):

```
python -m AM012_Abhishek_Mundanat.eda
python -m AM012_Abhishek_Mundanat.folium_maps                 # EDA maps
python -m AM012_Abhishek_Mundanat.folium_maps --predictions   # after evaluate_hmm
```

Outputs: `results/eda/` and `results/maps/`.
