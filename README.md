!git clone https://github.com/kimjuju123/radchem-analysis.git
%cd radchem-analysis
import radchem_script

run the above in colab and to use

aldf1 = rc.ExponentialFitter(al['a_thick1']*2.54, #array or data frace , ignore that 2.54 its me being messy
                             al['cpm1'], # array or dataframe
                             xaxis_label='Thickness (cm)', #can change this
                             yaxis_label='CPM', #axis title
                             guess = (8000,-5), # Modified guess, can change
                             background_subtract = background_counts, 
                             dead_time = dt_min
                             )
aldf1.plot_fit(title = 'Aluminum Shelf 2') # Call the plot_fit method to generate the graph
aldf1.plot_residuals() # Call the plot_residuals method to generate the residuals plot
plt.show()

run that. 

can open the script to call other things as well. 

this uses python
