Here we deposit the code associated with the article 
"Computer-generation of readily synthesizable structural analogs: By neural networks or reaction networks?"
by 
Wiktor Beker, Agnieszka Wołos, Martyna Moskal and Bartosz A. Grzybowski 

----
Deposited code consists of two files:

- network_generator.py - the code ilustrates how to perform forward synthetic calculation towards given target.
  Please note it is not final ready-to-use code, if you want to run the code you need to define:
   - BUYABLE_SMILES - list of compounds (in SMILES format) which can be used as a starting materials and/or reactants
   - RX_DATABASE - list of reactions (as RDKit's ChemicalReaction objects) which can be used in network generation

  Moreover in calc_synthetic_generation() function you need to plug-in code for performing reaction from given set of substrates.

- selection.py - the code shows how to perform pruning of computed synthetic generation and select possible substrates for
  next generation (i.e. which combinations are allowed and which are not). The script is ready-to-use, user needs to provide
  proper input data in JSON format (see comments in the script for details)
