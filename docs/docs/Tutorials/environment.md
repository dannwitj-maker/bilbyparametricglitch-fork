# Setting Up an Environment



1. Clear any current conda environment:
```
$ conda deactivate
```
2. Create a new environment with an appropriate version of python (3.13 does not work, 3.12 might or might not, I haven't tested, 3.11 certainly does), and activate it:
```
$ mamba create -n bilby-glitch-environment python=3.11
$ mamba activate bilby-glitch-environment
```
Note I am using mamba (which I *believe* is pre-installed on the cluster) because it usually resolves dependencies much faster than native conda does.
3. Install `bilby[gw]` via mamba, which should capture *most* dependencies we'll need
```
$ mamba install bilby[gw]
```
4. At time of writing, `bilby_pipe` does not yet contain necessary updates for compatibility (these will be introduced in `1.6.0`). So, install `bilby_pipe` from main by cloning:
```
$ git clone git@git.ligo.org:lscsoft/bilby_pipe,
$ cd bilby_pipe
$ pip install .
$ cd ..
```
5. Install `bilby_glitch` similarly by cloning:
```
$ git clone git@git.ligo.org:rhiannon.udall/bilbyparametricglitch
$ cd bilbyparametricglitch
$ pip install -e .
```
Where the `-e` is convenient so you don't have to pip install every time you make some development change.
