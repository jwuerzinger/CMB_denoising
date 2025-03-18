Contains code for reconstructing maria data with the jax implementation of NIFTy.
- [nifty_maria/](nifty_maria/): jax-based rewrite of maria's map sampling as well as general steering code.
- [steering/](steering): Test notebooks for testing improved steering notebooks.
    - [steering/steering_full.py](steering/steering_full.py): Main steering script for all reconstructions.
    - [steering/mustang_atmos_opt.ipynb](steering/mustang_atmos_opt.ipynb): Notebook containing MUSTANG-2 atmosphere fit parameter optimisation (very quick to run).
    - [steering/mustang_map_opt.ipynb](steering/mustang_map_opt.ipynb): Notebook containing MUSTANG-2 map fit parameter optimisation (quick to run).
    - [steering/atlast_atmos_opt.ipynb](steering/atlast_atmos_opt.ipynb): Notebook containing AtLAST atmosphere fit parameter optimisation (very quick to run).
    - [steering/atlast_map_opt.ipynb](steering/atlast_map_opt.ipynb): Notebook containing AtLAST map fit parameter optimisation (slow to run).
