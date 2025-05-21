# Causally Learned Representations for Science, Art and Sustainability

TBA


## Environment set-up

Note: pandas and torchvision installs will also supply numpy, torch, pillow etc
```
pip install pandas==2.2.3
pip install matplotlib==3.9.3
pip install seaborn==0.13.2
pip install torchvision==0.22.0
pip install jupyter notebook
python -m ipykernel install --name clear-dear
pip install torchvision==0.22.0
```


## Repository structure

Draft at the moment, add / review - not yet sure how things will interact?

* * *
    clear-sas
    │ 
    ├── data                   --> **empty** folder for data e.g. to dev / test tools
    │   │   
    │   └── .gitignore         --> making sure no data is stored on git, just the folder
    │ 
    ├── src                    --> code base
    │   │   
    │   ├── TBD?               --> _To Be Developed_ submodules (e.g. parsing, plotting etc) 
    │   │   │  
    │   │   └── __init__.py 
    │   │
    │   ├── utils.py           --> shared utilities, e.g. log formatting
    │   │   
    │   └── __init__.py
    │
    ├── notebooks              --> notebooks
    │   │   
    │   └── ab-0-0-name        --> e.g. initials, number, descriptive name
    │
    ├── scripts                --> python scripts for console use?
    │
    ├── (env / setup) 
    │
    ├── README.md
    │ 
    └── .gitignore
* * *