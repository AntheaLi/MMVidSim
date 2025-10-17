# MMVidSim

Yichen Li, Antonio Torralba

-------

Code release for ICCV 2025 'MultiModal Action Conditioned Video Generation'



-----
### set up

`git clone https://github.com/AntheaLi/MMVidSim.git`

`mamba create -f environment.yaml`


-----
### process data

We use opensource *Action Sense* data from https://action-sense.csail.mit.edu

-----
### train

```
cd src
bash scripts/train_history_hyperplane.sh
```

