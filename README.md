# Macropinosome tracking (volume and pH measurement) repo

Companion repository for "Proton-gated anion transport governs endocytic vacuole shrinkage" by Mariia Zeziulia, Sandy Blin, Franziska W. Schmitt, Martin Lehmann, Thomas J. Jentsch

# Installation 

To run the Python code in this repo, we recommend building a virtual environment using `venv` and the provided `requirements.txt` file, which 
will install all of the required dependencies for you:

```
python3 -m venv my_env
source my_env/bin/activate (for Mac OS)
my_env\Scripts\activate (for Windows)
pip install -r requirements.txt
```

# Usage

This program expects all time frames from the same experiment within the same condition (one imaging round) to be in multipage tiff file. If imaging was done is z stacks, 3-D stack of images should be converted to 2-D projection (e.g. maximum projection) along the z-axis separately before conducting further analysis with this program. 

This program is designed to either measure radius and fluorescent intensity of macropinosomes in one-channel image (mode "volume") or pH of macropinosomes (output is radius and fluorescent intensity of each macropinosome on pH-sensitive and pH-insensitive wavelengths) therefore mode "pH" requires 2-channel image.

You can run the code using either an IPython kernel (e.g. in VSCode or Spyder or Pycharm) or command line.

Open up the file `run_analysis.py` in your IDE of choice or call it from the command line and then provide

- the path to a folder with images
- analysis type ("volume" or "pH")
- last and first frame numbers
- number of conditions
- and enter all conditions in one line separated with spaces. 

If you selected "pH" as your analysis type, you will additionally be asked to provide

- the background percentile value (for our images we use 2, that means that the background value will be an intensity value that corresponds to 2-nd percentile of intensity distribution for each frame individually)
- Number ID of the pH-sensitive and pH-insensitive wavelength (in 2-channel image one wavelength will correspond to ID 0 and another to ID 1). Macropinosome search will be performed on images at pH-sensitive wavelenght, so if in your case macropinosomes are better visible in pH-insensitive wavelength, input the corresponding number. 

In the csv file output for "volume" mode the sequence of colums is radii for each time step, mean fluorescent intensities at each time step.

In the csv file output for "pH" mode the sequence of colums is:

1) radii for each time step; 
2) mean fluorescent intensities at pH-sensitive wavelength at each time step; 
3) background values for pH-sensitive wavelength at each time step; 
4) mean fluorescent intensities at pH-insensitive wavelength at each time step; 
5) background values for pH-insensitive wavelength at each time step.

Every step in image analysis has a plotting function included, which is commented out.