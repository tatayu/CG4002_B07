import numpy as np

def plot_graphs(processed_df, j, axs):
    x = [i for i in range(len(processed_df))]
    colors = np.tile(np.array(['red', 'orange', 'green']), math.ceil(len(processed_df.columns)/3))
    for i, col in enumerate(processed_df.columns):
        axs[j, i].plot(x, processed_df[col], 'tab:'+colors[i])
        axs[j, i].set_axis_off()
        axs[j, i].axis([0, len(processed_df), min(processed_df[col]), max(processed_df[col])])
        if i == 15:
            break
    return processed_df