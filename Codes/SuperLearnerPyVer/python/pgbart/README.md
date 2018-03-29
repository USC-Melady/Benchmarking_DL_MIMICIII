This folder contains the scripts used in the following paper:

**Particle Gibbs for Bayesian Additive Regression Trees**

Balaji Lakshminarayanan, Daniel M. Roy, Yee Whye Teh

*Proceedings of International Conference on Artificial Intelligence and Statistics (AISTATS), 2015.*

[Link to PDF](http://www.gatsby.ucl.ac.uk/~balaji/pgbart_aistats15.pdf)

Please cite the above paper if you use this code.

Code released under MIT license (see LICENSE for more info).

If you have any questions/comments/suggestions, please contact me at 
[balaji@gatsby.ucl.ac.uk](mailto:balaji@gatsby.ucl.ac.uk).

Copyright &copy; 2015 Balaji Lakshminarayanan

----------------------------------------------------------------------------

I ran my experiments using Enthought python (which includes all the necessary python packages).
If you are running a different version of python, you will need the following python packages to run the scripts:

* numpy
* scipy


The datasets are not included here; you can run 
experiments using toy data though. Run **commands.sh** in **process_data** folder for automatically 
downloading and processing the datasets. I have tested these scripts only on Ubuntu, but it should be straightforward to process datasets in other platforms.

----------------------------------------------------------------------------

**List of scripts in the src folder**:

- bart.py (main script that does all the computation)
- bart_utils.py (collection of utilities)
- treemcmc.py (MCMC for single tree using CGM/GrowPrune/Particle Gibbs)
- pg.py (Particle Gibbs algorithm)

**Help on usage** (type the following command on the terminal):

./bart.py -h

**Example usage**:

*CGM*: 
./bart.py --store_every=1 --m_bart=1 --verbose=0 --mcmc_type=cgm --dataset=toy-hypercube-3 --save=1 --n_iterations=200 --init_id=1 --alpha_split=0.95 --beta_split=0.5 --tag=example --n_run_avg=1


*GrowPrune*: ./bart.py --store_every=1 --m_bart=1 --verbose=0 --mcmc_type=growprune --dataset=toy-hypercube-3 --save=1 --n_iterations=200 --init_id=1 --alpha_split=0.95 --beta_split=0.5 --tag=example --n_run_avg=1

*Particle Gibbs*: ./bart.py --store_every=1 --m_bart=1 --verbose=0 --mcmc_type=pg --dataset=toy-hypercube-3 --save=1 --n_iterations=200 --init_id=1 --alpha_split=0.95 --beta_split=0.5 --tag=example --n_run_avg=1 --init_pg=empty

**Example on a real-world dataset**:

*(assuming you have successfully run commands.sh in process_data folder)*

*Particle Gibbs*: ./bart.py --store_every=1 --m_bart=1 --verbose=0 --mcmc_type=pg --dataset=houses_01 --save=1 --n_iterations=200 --init_id=1 --alpha_split=0.95 --beta_split=0.5 --tag=example --n_run_avg=1 --init_pg=empty --data_path='../process_data/'

----------------------------------------------------------------------------
**Running experiments on your dataset**:

- process your dataset into suitable format: see load_rgf_datasets in bart_utils.py for an example
-  modify load_data in bart_utils.py: either add your dataset name or remove the "raise Exception" line if you prefer that
-  Note: you might have to pass data_path as an argument when you call bart.py

 Note that the results (predictions, mse, log predictive probability on training/test data, runtimes) are stored in the pickle files. 
You need to write additional scripts to aggregate the results from these pickle files and generate the plots.
  
----------------------------------------------------------------------------

I generated commands for parameter sweeps using 'build_cmds' script by Jan Gasthaus, available publicly at [https://github.com/jgasthaus/Gitsby/tree/master/pbs/python](https://github.com/jgasthaus/Gitsby/tree/master/pbs/python).

**Example of parameter sweep**:

./build_cmds ./bart.py "--store_every={0}" "--m_bart={200}" "--q_bart={0.9}" "--verbose={0}" "--mcmc_type={cgm,growprune,pg}" "--sample_y={0}" "--dataset={msd_01,ctslices_01,houses_01}" "--save={1}"  "--n_iterations={2000}" "--init_id=1:1:2" "--alpha_split={0.95}" "--beta_split={2.0}" "--n_run_avg={10}" "--data_path={../process_data/}" >> run



