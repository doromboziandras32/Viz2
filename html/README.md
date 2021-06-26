# Viz2
 
1. Install the requirements from the attached txt:
    ```
    pip install -r requirements.txt
    ```
2. Open jupyter notebook **LDA_viz.ipynb**

3. Click on **Run All Cells**

4. After all cells are executed, open the following link: [Dash App](http://127.0.0.1:8051/)

## If problem appears

If the upon solution doesn't work: 
Clone the following git repository:
   ```
   !git clone https://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@github.com/doromboziandras32/Viz2.git
   ```

Then follow step 1-4 

### Jupyter kernel doesn't start
If the the following error appears (in case of PyCharm):
   ```
   ImportError: DLL load failed while importing win32api: The specified module could not be found.
   ```

pywin32_postinstall.py has to be installed as Administrator ([Stackoverflow reference](https://stackoverflow.com/questions/58612306/how-to-fix-importerror-dll-load-failed-while-importing-win32api))
   ```
   python <path_of_the_venv>\Scripts\pywin32_postinstall.py -install
   ```

Restart Kernel then run the jupyter notebook