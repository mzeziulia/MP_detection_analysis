# Macropinosome tracking (volume and pH measurement) repo

Companion repository for "Proton-gated anion transport governs endocytic vacuole shrinkage" by Mariia Zeziulia, Sandy Blin, Franziska W. Schmitt, Martin Lehmann, Thomas J. Jentsch

# Installation 

To run the Python code in this repo, we recommend building a virtual environment using `venv` and the provided `requirements.txt` file, which 
will install all of the required dependencies for you:

```
python3 -m venv my_env
source my_env/bin/activate (for Mac)
my_env\Scripts\activate (for Windows)
pip install -r requirements.txt
```

# Usage

This program expects all time frames from the same experiment within the same condition (one imaging round) to be in multipage tiff file. If imaging was done is z stacks, 3-D stack of images should be converted to 2-D projection (e.g. maximum projection) along the z-axis separately before conducting further analysis (e.g. Fiji).

You can run the code using either an IPython kernel (e.g. in VSCode or Spyder or Pycharm) or command line. 
Open up the file `run_analysis.py` in your IDE of choice or call it from the command line and then provide the path to a folder with images, last and first frame numbers, number of conditions and enter all conditions in one line separated with spaces

Every step in image analysis has a plotting function included, which can be deleted or commented out for the time reasons