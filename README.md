# Exploratory Data Analysis and Feature Extraction

This project involves exploratory data analysis (EDA) and feature extraction on various datasets, including white noise series, random walk series, temperature anomalies, sequential sine wave signals, ERP conditions, and EEG signal data.

## Project Structure

The project is organized into the following files:

- `task1-1.py`: Analysis of white noise series.
- `task1-2.py`: Analysis of random walk series.
- `task1-3.py`: Analysis of temperature anomalies.
- `task2-1.py`: Analysis of sequential sine wave signals.
- `task2-2.py`: Analysis of ERP conditions.
- `task2-3.py`: Analysis of EEG signal data.
- `task2-3-discussion.py`: Validate manual calculations, and draw the ACF graph for the time series.
- `requirements.txt`: List of dependencies required to run the project.
- `README.md`: This file, providing an overview and instructions.

## Dependencies

To run this project, you need the following Python libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scipy`
- `statsmodels`

You can install these dependencies using pip:

```bash
pip install -r requirements.txt

## Running the Code

To run the analysis and generate the plots, execute the following command for each script:

```bash
python task1-1.py
python task1-2.py
python task1-3.py
python task2-1.py
python task2-2.py
python task2-3.py
python task2-3-discussion.py
## Obtaining the Results

Each script will generate and save various plots in the `pics` directory. The plots are saved with descriptive filenames, making it easy to identify the content of each plot. For example:

- `1-1-line_plot.png`: Line plot of the white noise series.
- `1-2-histogram.png`: Histogram of the random walk series.
- `1-3-acf_plot.png`: ACF plot of the temperature anomalies.
- `2-1-sine_wave_plot.png`: Plot of the sequential sine wave signals.
- `2-2-erp_plot.png`: Plot of the ERP conditions.
- `2-3-power_spectrum.png`: Power spectrum of the EEG signal.

