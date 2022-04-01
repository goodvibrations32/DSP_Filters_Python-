# DSP_Filters_Python-
This is a repository for filtering noise above 100Hz from wind speed measurment signal due to interferences. 
The method used for the process is a simple long FIR filter. The size of the filter varies in two ways:

1. Lazy_Approach branch : Here the most of the dirt is removed but there is a significant amount of noise at 0.1 - 1 {kHz}. The file "create_dataset_low_res_.py" handles the opperations needed to construct a new file with the filtered data.

2. High_analysis branch : Here all the interference is removed from the output signal. The respected file constructs a new file with the clean data for comparison of two methods.

3. main branch : In this branch the files in the folder 'Filtering a signal' provide a general walkthrough in digital and analog both FIR (analog) and IIR (digital) type filters with some basic plots for understanding frequency responce. The files in the folder 'signal_process_plots' are providing plots in both time and frequency domain as well as the power spectrum of the raw and processed signals.

All the data used in this project was provided by an organization. Minor adjustments may be needed for file paths in OS.
Dependencies : python >=3.6 (f-strings), numpy, scipy, matplotlib

--
GoodVibrations32
