import pickle
import pandas as pd
import matplotlib.pyplot as plt

file_path = '/home/ale/Downloads/aegnn_results/flops.pkl'

with open(file_path, 'rb') as file:
    flops_data = pickle.load(file)
flops_df = pd.DataFrame(flops_data)
save_to_excel = True
if save_to_excel == True:    
    # Specify the output file path
    output_file_path = '/home/ale/Downloads/aegnn_results/flops_data_ncaltech101.xlsx'

    # Save the DataFrame to the Excel file
    flops_df.to_excel(output_file_path, index=False, engine='openpyxl')
flops_df['Mflops/ev'] = flops_df['flops'] / 1e6 / 25000
average_runtime = flops_df.groupby(['layer', 'model'])['runtime'].mean()
print(average_runtime)
average_flops = flops_df.groupby(['layer', 'model'])[['flops', 'Mflops/ev']].mean()
print(average_flops)
ave_to_excel = True

if save_to_excel:
    average_flops.to_excel('/home/ale/Downloads/aegnn_results/avg_flops_ncaltech101.xlsx', index=True, engine='openpyxl')
    average_runtime.to_excel('/home/ale/Downloads/aegnn_results/avg_runtime_ncaltech101.xlsx', index=True, engine='openpyxl')
df = pd.read_excel('/home/ale/Downloads/aegnn_results/avg_flops_ncaltech101.xlsx')
df['layer'] = df['layer'].fillna(method='ffill')
print(df)
import matplotlib.pyplot as plt
import pandas as pd
# Assuming the DataFrame is named average_flops

df = pd.read_excel('/home/ale/Downloads/aegnn_results/avg_flops_ncaltech101.xlsx')
df['layer'] = df['layer'].fillna(method='ffill')

# Reset the index to make 'Layer' and 'Model' regular columns
#df.reset_index(inplace=True)

# Create a figure and axes
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the data
bar_width = 0.35
opacity = 0.8

# Filter the data for each model
gnn_dense_data = df[df['model'] == 'gnn_dense']
ours_data = df[df['model'] == 'ours']

# Create the bars for each model
gnn_dense_bars = ax.bar(gnn_dense_data['layer'], gnn_dense_data['Mflops/ev'], bar_width,
                        alpha=opacity, color='b', label='gnn_dense')
ours_bars = ax.bar(ours_data['layer'], ours_data['Mflops/ev'],
                   alpha=opacity, color='g', label='ours', align='center', width=0.5)

# Set up the axes labels, title, and legend
ax.set_ylabel('Mflops/ev')
ax.set_xlabel('Layer')
ax.set_yscale('log')

ax.set_title('Mflops/ev per Layer for Each Model')
ax.legend()

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Show the plot

plt.savefig('/home/ale/Downloads/aegnn_results/bar_plot_Mflops_ev_ncaltech101.pdf', format='pdf', bbox_inches='tight')

plt.show()